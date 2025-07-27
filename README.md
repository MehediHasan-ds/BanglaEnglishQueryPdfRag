# PDF Chatbot - Bengali & English OCR with LangChain

A complete Docker-containerized solution for creating a chatbot that can process both Bengali and English PDFs using OCR and answer questions using Groq AI.

## ✨ Features

- 📄 **PDF Processing**: Upload and process PDF files with OCR
- 🌐 **Multi-language Support**: Bengali and English text extraction
- 🤖 **AI-Powered Chat**: Query your PDFs using Groq's LLM
- 🐳 **Fully Dockerized**: No manual installation of Tesseract or Poppler
- 🎨 **Modern UI**: Clean and responsive web interface
- ⚡ **FastAPI Backend**: High-performance async API

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd pdf-chatbot
```

### 2. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Build and Run with Docker Compose
```bash
# Build and start the application
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### 4. Access the Application
Open your browser and go to: `http://localhost:8000`

## 🛠️ Manual Setup (Alternative)

If you prefer to run without Docker:

### Prerequisites
- Python 3.8+
- Tesseract OCR
- Poppler Utils

### Installation Steps

1. **Install System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng poppler-utils
   
   # macOS
   brew install tesseract tesseract-lang poppler
   
   # Windows
   # Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
   # Download and install Poppler from: https://blog.alivate.com.au/poppler-windows/
   ```

2. **Clone and Install Python Dependencies**
   ```bash
   git clone <your-repo-url>
   cd pdf-chatbot
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the Application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📁 Project Structure

```
pdf-chatbot/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── .env                    # Environment variables
├── templates/
│   └── index.html          # Main HTML template
├── static/
│   ├── style.css           # CSS styles
│   └── script.js           # JavaScript functionality
├── uploads/                # Temporary PDF uploads
├── extracted_text          # stores the extracted text
└── chroma_db/              # Vector database storage
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | Required |
| `GROQ_MODEL` | Groq model to use | `mixtral-8x7b-32768` |
| `MAX_FILE_SIZE` | Max upload size in bytes | `10485760` (10MB) |
| `CHUNK_SIZE` | Text chunk size for embeddings | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

### Supported Languages
- **Bengali**: Full OCR support with English character filtering
- **English**: Standard OCR processing

## 🎯 Usage Guide

### 1. Upload PDF
- Click "Choose PDF File" and select your PDF
- Select the language (Bengali or English)
- Click "Upload & Process"

### 2. Chat with PDF
- Once processed, the chat interface will appear
- Ask questions in Bengali or English
- The AI will respond based on the PDF content

### Example Questions
**English:**
- "What is the main topic of this document?"
- "Summarize the key points"
- "What does page 1 contain?"

**Bengali:**
- "এই ডকুমেন্টের মূল বিষয় কি?"
- "মূল পয়েন্টগুলো সংক্ষেপে বলুন"
- "প্রথম পাতায় কি আছে?"

## 🐳 Docker Details

### Building the Image
```bash
docker build -t pdf-chatbot .
```

### Running with Custom Port
```bash
docker run -p 3000:8000 -e GROQ_API_KEY=your_key pdf-chatbot
```

### Volume Mounting for Persistence
```bash
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  pdf-chatbot
```

## 🧪 Health Check

Check if the application is running:
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "vectorstore_ready": false
}
```

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/upload` | POST | Upload and process PDF |
| `/chat` | POST | Chat with processed PDF |
| `/health` | GET | Health check |

### API Usage Examples

**Upload PDF:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "language=bengali"
```

**Chat:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## 🎨 Customization

### Styling
Modify `static/style.css` to change the appearance.

### OCR Configuration
Adjust OCR settings in the `extract_text_from_image()` function:
```python
config = r'--oem 3 --psm 6 -l ben+eng'  # OCR Engine Mode + Page Segmentation Mode
```

### LLM Model
Change the Groq model in `main.py`:
```python
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama2-70b-4096",  # or other available models
    temperature=0.1
)
```

## 🐛 Troubleshooting

### Common Issues

**1. GROQ_API_KEY not found**
- Ensure your `.env` file contains the correct API key
- Restart the Docker container after adding the key

**2. OCR not working**
- Check if the PDF contains text or images
- Bengali text requires proper font rendering

**3. Memory issues**
- Large PDFs may require more memory
- Adjust Docker memory limits if needed

**4. Permission denied (uploads directory)**
```bash
chmod 755 uploads/
```

**5. Port already in use**
```bash
# Change port in docker-compose.yml
ports:
  - "3000:8000"  # Use port 3000 instead
```

### Logs
View application logs:
```bash
docker-compose logs -f
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Create an issue on GitHub

## 🔄 Updates

To update the application:
```bash
git pull origin main
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

**Made with ❤️ for multilingual PDF processing**