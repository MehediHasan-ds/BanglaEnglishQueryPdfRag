# main.py - OPTIMIZED VERSION WITH PROPER FLOW AND CONTEXT MANAGEMENT

import os
import tempfile
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import cv2
from pdf2image import convert_from_path
import numpy as np
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
from datetime import datetime
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
def create_directories():
    """Create all required directories if they don't exist."""
    directories = ["static", "templates", "uploads", "extracted_text", "chroma_db"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory '{directory}' ensured to exist")

# Initialize directories
create_directories()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MINIMAL GLOBAL STATE - Only for session management
current_vectorstore_info = {
    "collection_name": None,
    "language": None,
    "pdf_filename": None,
    "persist_directory": "./chroma_db"
}

# Add this function for token counting
def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text. Using GPT-3.5 tokenizer as approximation for Llama."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(text) // 4

def initialize_llm():
    """Initialize Groq LLM - Called fresh each time."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables")
    
    return ChatGroq(
        api_key=groq_api_key,
        model="llama3-8b-8192",
        temperature=0.1
    )

def get_embeddings():
    """Get embeddings model - Created fresh each time."""
    return HuggingFaceEmbeddings(
        model_name="shihab17/bangla-sentence-transformer",
        model_kwargs={'device': 'cpu'}
    )

def extract_text_from_image(image, language="bengali"):
    """Extract text from a single image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply image preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    
    if language == "bengali":
        config = r'--oem 3 --psm 6 -l ben'
    else:
        config = r'--oem 3 --psm 6 -l eng'
    
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    
    text_lines = []
    current_line = []
    prev_line_num = -1
    
    for i in range(len(data['text'])):
        if data['text'][i].strip() and int(data['conf'][i]) > 30:
            line_num = data['line_num'][i]
            if line_num != prev_line_num and current_line:
                text_lines.append(' '.join(current_line).strip())
                current_line = []
            current_line.append(data['text'][i].strip())
            prev_line_num = line_num
    
    if current_line:
        text_lines.append(' '.join(current_line).strip())
    
    if language == "bengali":
        filtered_lines = []
        for line in text_lines:
            cleaned_line = re.sub(r'[^\u0980-\u09FF\u09E6-\u09EF।॥\s]', '', line)
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
            if cleaned_line:
                filtered_lines.append(cleaned_line)
        return '\n\n'.join(filtered_lines)
    
    return '\n\n'.join(text_lines)

def extract_text_from_pdf(pdf_path, language="bengali"):
    """Convert PDF pages to images and extract text."""
    try:
        images = convert_from_path(pdf_path)
        all_text = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            text = extract_text_from_image(image_cv, language)
            if text.strip():
                all_text.append(text)
        return '\n\n'.join(all_text)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return f"Error processing PDF: {str(e)}"

def save_uploaded_pdf(file: UploadFile, language: str):
    """Save the uploaded PDF file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{language}_{file.filename}"
        pdf_path = os.path.join("uploads", filename)
        
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"PDF saved to: {pdf_path}")
        return pdf_path, filename
    except Exception as e:
        logger.error(f"Error saving PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving PDF: {str(e)}")

def save_extracted_text(text: str, original_filename: str, language: str):
    """Save extracted text."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(original_filename)[0]
        txt_filename = f"{timestamp}_{language}_{base_name}.txt"
        txt_path = os.path.join("extracted_text", txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Extracted text saved to: {txt_path}")
        return txt_path, txt_filename
    except Exception as e:
        logger.error(f"Error saving extracted text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving extracted text: {str(e)}")

def create_vectorstore_only(txt_file_path, language):
    """
    OPTIMIZED: Create vectorstore with better chunking strategy
    """
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found")
        
        logger.info(f"Creating vectorstore from text length: {len(text)} characters")
        
        # OPTIMIZED CHUNKING STRATEGY
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # REDUCED from 1000 - smaller chunks
            chunk_overlap=100,     # REDUCED from 200 - less overlap
            length_function=len,
            separators=["\n\n", "\n", "।", ".", " ", ""]
        )
        
        documents = [Document(page_content=text, metadata={"source": txt_file_path, "language": language})]
        texts = text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(texts)} optimized text chunks")
        
        # Log chunk sizes for debugging
        avg_chunk_size = sum(len(chunk.page_content) for chunk in texts[:5]) / min(5, len(texts))
        logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
        
        # Create embeddings and vectorstore
        embeddings = get_embeddings()
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        collection_name = f"pdf_collection_{uuid.uuid4().hex[:8]}"
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        vectorstore.persist()
        logger.info(f"✅ OPTIMIZED VECTORSTORE CREATED: Collection {collection_name}")
        
        current_vectorstore_info.update({
            "collection_name": collection_name,
            "language": language,
            "persist_directory": persist_directory
        })
        
        return collection_name
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating vectorstore: {str(e)}")

def load_vectorstore(collection_name, persist_directory):
    """
    Load existing vectorstore from disk.
    Called fresh for each chat request.
    """
    try:
        embeddings = get_embeddings()
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        logger.info(f"✅ VECTORSTORE LOADED: Collection {collection_name}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error loading vectorstore: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading vectorstore: {str(e)}")

def create_prompt_template(language):
    """
    OPTIMIZED: Shorter, more concise prompt templates
    """
    if language == "bengali":
        # SHORTENED Bengali prompt
        system_prompt = """PDF থেকে উত্তর দিন। প্রসঙ্গে না পেলে "উত্তর পাওয়া যায়নি" বলুন।

প্রসঙ্গ:
{context}

প্রশ্ন: {question}
উত্তর:"""
    else:
        # SHORTENED English prompt
        system_prompt = """Answer from PDF content. Say "Answer not found" if not in context.

Context:
{context}

Question: {question}
Answer:"""
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt
    )

def smart_context_selection(retriever, question, max_context_tokens=1500):
    """
    SMART CONTEXT MANAGEMENT: Select best chunks within token limit
    """
    try:
        # Get more candidates initially
        candidate_docs = retriever.get_relevant_documents(question)
        
        if not candidate_docs:
            logger.warning("No documents retrieved!")
            return "", []
        
        logger.info(f"Retrieved {len(candidate_docs)} candidate documents")
        
        # INTELLIGENT SELECTION STRATEGY
        selected_chunks = []
        total_tokens = 0
        
        # Sort by relevance (assuming first docs are most relevant)
        for i, doc in enumerate(candidate_docs):
            chunk_text = doc.page_content
            chunk_tokens = count_tokens(chunk_text)
            
            # Skip if chunk alone exceeds limit
            if chunk_tokens > max_context_tokens:
                logger.warning(f"Chunk {i} too large ({chunk_tokens} tokens), truncating...")
                # Truncate large chunk
                chunk_text = chunk_text[:max_context_tokens * 3]  # Rough char limit
                chunk_tokens = count_tokens(chunk_text)
            
            # Add if within total limit
            if total_tokens + chunk_tokens <= max_context_tokens:
                selected_chunks.append(chunk_text)
                total_tokens += chunk_tokens
                logger.info(f"✅ Added chunk {i}: {chunk_tokens} tokens (Total: {total_tokens})")
            else:
                logger.info(f"❌ Skipped chunk {i}: would exceed limit")
                break
        
        # Combine selected chunks
        if selected_chunks:
            context = "\n\n".join(selected_chunks)
            logger.info(f"✅ FINAL CONTEXT: {total_tokens} tokens, {len(selected_chunks)} chunks")
            return context, candidate_docs[:len(selected_chunks)]
        else:
            # Fallback: use first chunk only, truncated
            first_chunk = candidate_docs[0].page_content
            truncated = first_chunk[:max_context_tokens * 3]  # Rough truncation
            logger.warning(f"⚠️ FALLBACK: Using truncated first chunk")
            return truncated, [candidate_docs[0]]
            
    except Exception as e:
        logger.error(f"Error in smart context selection: {str(e)}")
        return "", []

def create_qa_chain_with_context_management(collection_name, persist_directory, language):
    """
    OPTIMIZED QA CHAIN: Custom implementation with context length management
    """
    try:
        logger.info(f"🔄 CREATING CONTEXT-AWARE QA CHAIN")
        
        # Load vectorstore and create retriever
        vectorstore = load_vectorstore(collection_name, persist_directory)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Get more candidates for smart selection
        )
        
        # Create components
        prompt_template = create_prompt_template(language)
        llm = initialize_llm()
        
        logger.info("✅ CONTEXT-MANAGED QA CHAIN CREATED")
        
        return {
            "retriever": retriever,
            "prompt_template": prompt_template,
            "llm": llm,
            "language": language
        }
        
    except Exception as e:
        logger.error(f"Error creating context-managed QA chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating QA chain: {str(e)}")

def process_question_with_context_management(qa_components, question):
    """
    CUSTOM QA PROCESSING: Manual context management instead of RetrievalQA
    """
    try:
        logger.info(f"🔄 PROCESSING QUESTION WITH CONTEXT MANAGEMENT")
        
        # Step 1: Smart context selection
        context, source_docs = smart_context_selection(
            qa_components["retriever"], 
            question, 
            max_context_tokens=1500  # Adjust based on your model's limit
        )
        
        if not context:
            logger.warning("No context retrieved!")
            if qa_components["language"] == "bengali":
                return "প্রশ্নের জন্য কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি।", []
            else:
                return "No relevant information found for the question.", []
        
        # Step 2: Create final prompt
        final_prompt = qa_components["prompt_template"].format(
            context=context,
            question=question
        )
        
        # Step 3: Token count check
        prompt_tokens = count_tokens(final_prompt)
        logger.info(f"📊 FINAL PROMPT: {prompt_tokens} tokens")
        
        # Safety check - if still too long, truncate context further
        if prompt_tokens > 3000:  # Conservative limit for Llama3-8B
            logger.warning(f"⚠️ Prompt still too long ({prompt_tokens} tokens), truncating context...")
            
            # Drastically reduce context
            context_lines = context.split('\n\n')
            reduced_context = '\n\n'.join(context_lines[:2])  # Keep only first 2 chunks
            
            final_prompt = qa_components["prompt_template"].format(
                context=reduced_context,
                question=question
            )
            
            new_token_count = count_tokens(final_prompt)
            logger.info(f"📊 REDUCED PROMPT: {new_token_count} tokens")
        
        # Step 4: Get LLM response
        logger.info("🤖 SENDING TO LLM...")
        response = qa_components["llm"].invoke(final_prompt)
        
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        logger.info(f"✅ LLM RESPONSE: {answer[:100]}...")
        
        return answer, source_docs
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        
        # Return appropriate error message
        if qa_components["language"] == "bengali":
            return f"প্রশ্ন প্রক্রিয়াকরণে ত্রুটি: {str(e)}", []
        else:
            return f"Error processing question: {str(e)}", []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    language: str = Form(...)
):
    """
    UPLOAD FLOW: PDF → Extract Text → Create Vectorstore ONLY
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    await file.seek(0)
    
    try:
        logger.info("🚀 STARTING PDF UPLOAD PROCESS")
        
        # Step 1: Save uploaded PDF
        pdf_path, saved_filename = save_uploaded_pdf(file, language)
        current_vectorstore_info["pdf_filename"] = saved_filename
        logger.info(f"✅ PDF SAVED: {pdf_path}")
        
        # Step 2: Extract text from PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            await file.seek(0)
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        logger.info("🔄 EXTRACTING TEXT FROM PDF")
        extracted_text = extract_text_from_pdf(temp_path, language)
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        if extracted_text.startswith("Error"):
            raise HTTPException(status_code=500, detail=extracted_text)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted")
        
        logger.info(f"✅ TEXT EXTRACTED: {len(extracted_text)} characters")
        
        # Step 3: Save extracted text
        txt_path, txt_filename = save_extracted_text(extracted_text, file.filename, language)
        logger.info(f"✅ TEXT SAVED: {txt_path}")
        
        # Step 4: Create vectorstore ONLY (NO QA chain here!)
        collection_name = create_vectorstore_only(txt_path, language)
        
        logger.info("🎉 UPLOAD PROCESS COMPLETED")
        
        return JSONResponse({
            "status": "success",
            "message": f"PDF processed successfully in {language}",
            "text_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "text_length": len(extracted_text),
            "collection_name": collection_name,
            "files_saved": {
                "pdf_filename": saved_filename,
                "txt_filename": txt_filename
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat")
async def chat(request: Request):
    """
    OPTIMIZED CHAT: Using custom context management
    """
    if not current_vectorstore_info["collection_name"]:
        raise HTTPException(status_code=400, detail="Please upload a PDF first")
    
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"🚀 PROCESSING QUESTION: {question}")
        
        # Create QA components with context management
        qa_components = create_qa_chain_with_context_management(
            collection_name=current_vectorstore_info["collection_name"],
            persist_directory=current_vectorstore_info["persist_directory"],
            language=current_vectorstore_info["language"]
        )
        
        # Process question with smart context management
        answer, source_docs = process_question_with_context_management(qa_components, question)
        
        logger.info("🎉 QUESTION PROCESSED SUCCESSFULLY")
        
        response = {
            "answer": answer,
            "sources": len(source_docs),
            "language": current_vectorstore_info["language"],
            "pdf_filename": current_vectorstore_info["pdf_filename"],
            "collection_name": current_vectorstore_info["collection_name"]
        }
        
        return JSONResponse(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vectorstore_ready": current_vectorstore_info["collection_name"] is not None,
        "current_session": current_vectorstore_info,
        "directories": {
            "uploads": os.path.exists("uploads"),
            "extracted_text": os.path.exists("extracted_text"),
            "chroma_db": os.path.exists("chroma_db"),
        }
    }

@app.get("/files")
async def list_files():
    """List all files for debugging."""
    files_info = {}
    directories = ["uploads", "extracted_text", "chroma_db"]
    for directory in directories:
        if os.path.exists(directory):
            files_info[directory] = os.listdir(directory)
        else:
            files_info[directory] = []
    return files_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



