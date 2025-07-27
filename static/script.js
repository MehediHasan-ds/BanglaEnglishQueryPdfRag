// static/script.js

// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const uploadBtn = document.getElementById('uploadBtn');
const uploadLoader = document.getElementById('uploadLoader');
const uploadStatus = document.getElementById('uploadStatus');
const chatSection = document.getElementById('chatSection');
const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const chatLoader = document.getElementById('chatLoader');
const chatMessages = document.getElementById('chatMessages');
const chatInfo = document.getElementById('chatInfo');
const fileInput = document.getElementById('pdfFile');
const btnText = document.querySelector('.btn-text');

// State
let isProcessing = false;
let currentLanguage = 'bengali';

// File input handler
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    const label = document.querySelector('.file-text');
    if (file) {
        label.textContent = file.name;
    } else {
        label.textContent = 'Choose PDF File';
    }
});

// Upload form handler
uploadForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    if (isProcessing) return;
    
    const formData = new FormData(uploadForm);
    const file = formData.get('file');
    const language = formData.get('language');
    
    if (!file || file.size === 0) {
        showStatus('Please select a PDF file', 'error');
        return;
    }
    
    currentLanguage = language;
    setProcessingState(true);
    
    try {
        const response = await fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus(result.message, 'success');
            showChatSection(language);
            updateChatInfo(language, file.name);
        } else {
            showStatus(result.detail || 'Upload failed', 'error');
        }
    } catch (error) {
        showStatus('Network error. Please try again.', 'error');
        console.error('Upload error:', error);
    } finally {
        setProcessingState(false);
    }
});

// Chat form handler
chatForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const question = questionInput.value.trim();
    if (!question || isProcessing) return;
    
    // Add user message
    addMessage(question, 'user');
    questionInput.value = '';
    setChatProcessingState(true);
    
    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            addMessage(result.answer, 'bot', result.sources);
        } else {
            addMessage(result.detail || 'Sorry, I encountered an error processing your question.', 'bot');
        }
    } catch (error) {
        addMessage('Network error. Please check your connection and try again.', 'bot');
        console.error('Chat error:', error);
    } finally {
        setChatProcessingState(false);
    }
});

// Utility functions
function setProcessingState(processing) {
    isProcessing = processing;
    uploadBtn.disabled = processing;
    
    if (processing) {
        btnText.style.display = 'none';
        uploadLoader.classList.add('show');
    } else {
        btnText.style.display = 'inline';
        uploadLoader.classList.remove('show');
    }
}

function setChatProcessingState(processing) {
    isProcessing = processing;
    sendBtn.disabled = processing;
    questionInput.disabled = processing;
    
    if (processing) {
        chatLoader.classList.add('show');
        document.querySelector('.send-icon').style.display = 'none';
    } else {
        chatLoader.classList.remove('show');
        document.querySelector('.send-icon').style.display = 'inline';
    }
}

function showStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-message ${type}`;
    uploadStatus.style.display = 'block';
}

function showChatSection(language) {
    chatSection.style.display = 'block';
    chatSection.scrollIntoView({ behavior: 'smooth' });
}

function updateChatInfo(language, filename) {
    const langText = language === 'bengali' ? 'Bengali (বাংলা)' : 'English';
    chatInfo.textContent = `Language: ${langText} | File: ${filename}`;
}

function addMessage(content, sender, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = `${sender}-avatar`;
    avatar.textContent = sender === 'user' ? '👤' : '🤖';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (sender === 'bot' && sources) {
        const sourcesText = sources > 0 ? ` (${sources} source${sources > 1 ? 's' : ''})` : '';
        messageContent.innerHTML = `<p>${content}</p><small style="opacity: 0.7;">Retrieved from PDF${sourcesText}</small>`;
    } else {
        messageContent.innerHTML = `<p>${content}</p>`;
    }
    
    if (sender === 'user') {
        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(avatar);
    } else {
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Enter key handler for chat input
questionInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Health check on load
window.addEventListener('load', async function() {
    try {
        const response = await fetch('http://localhost:8000/health');
        const health = await response.json();
        console.log('Application health:', health);
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
