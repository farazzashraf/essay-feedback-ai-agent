let currentInputMethod = 'paste';
let selectedFile = null;
let isEssaySubmitted = false;

// function toggleInput(method, event) {
//     event.preventDefault(); // 👈 Prevents default behavior
//     event.stopPropagation(); // 👈 Stops bubbling
//     currentInputMethod = method;
    
//     // Update button states
//     document.querySelectorAll('.toggle-btn').forEach(btn => {
//         btn.classList.remove('active');
//     });
//     event.target.classList.add('active');
    
//     // Show/hide input areas
//     const pasteArea = document.getElementById('pasteArea');
//     const uploadArea = document.getElementById('uploadArea');
    
//     if (method === 'paste') {
//         pasteArea.style.display = 'block';
//         uploadArea.classList.remove('active');
//     } else {
//         pasteArea.style.display = 'none';
//         uploadArea.classList.add('active');
//     }
// }

function toggleInput(method, event) {
    event.preventDefault(); // 👈 Prevents default behavior
    event.stopPropagation(); // 👈 Stops bubbling
    currentInputMethod = method;

    // Reset file input and selected file if switching to paste
    if (method === 'paste') {
        selectedFile = null;
        const fileInput = document.getElementById('fileInput');
        fileInput.value = ''; // Reset file input
        document.getElementById('selectedFile').innerHTML = '';
    }

    // Update button states
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Show/hide input areas
    const pasteArea = document.getElementById('pasteArea');
    const uploadArea = document.getElementById('uploadArea');

    if (method === 'paste') {
        pasteArea.style.display = 'block';
        uploadArea.style.display = 'none';
    } else {
        pasteArea.style.display = 'none';
        uploadArea.style.display = 'block';
    }
}


function updateWordCount() {
    const textarea = document.getElementById('essayText');
    const wordCount = document.getElementById('wordCount');
    const text = textarea.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    wordCount.textContent = `${words} words`;
}

function handleFileSelect(event) {
    event.stopPropagation();
    event.preventDefault();

    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        displaySelectedFile(file);

        // Remove focus so file picker doesn’t reopen due to weird focus event
        event.target.blur();
    }
}

// console.log("handleFileSelect triggered!");

// function handleFileSelect(event) {
//     const file = event.target.files[0];
//     if (file) {
//         selectedFile = file;
//         displaySelectedFile(file);
//     }
// }

function handleDrop(event) {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        selectedFile = file;
        displaySelectedFile(file);
        document.getElementById('fileInput').files = files;
    }
    event.target.classList.remove('dragover');
}

function handleDragOver(event) {
    event.preventDefault();
    event.target.classList.add('dragover');
}

function handleDragLeave(event) {
    event.target.classList.remove('dragover');
}

function displaySelectedFile(file) {
    const selectedFileDiv = document.getElementById('selectedFile');
    selectedFileDiv.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>📄</span>
            <span>${file.name}</span>
            <span style="color: #6b7280; font-size: 12px;">(${formatFileSize(file.size)})</span>
        </div>
    `;
    selectedFileDiv.style.display = 'block';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showError(message) {
    const feedbackContent = document.getElementById('feedbackContent');
    feedbackContent.innerHTML = `<div class="error">${message}</div>`;
}

function showLoading(message = 'Analyzing essay...') {
    const feedbackContent = document.getElementById('feedbackContent');
    feedbackContent.innerHTML = `<div class="loading">${message}</div>`;
}

async function submitEssay() {
    const submitBtn = document.querySelector('.submit-btn');
    const originalText = submitBtn.textContent;
    
    try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Analyzing...';
        showLoading();

        const formData = new FormData();
        formData.append('input_method', currentInputMethod);
        
        if (currentInputMethod === 'paste') {
            const essayText = document.getElementById('essayText').value.trim();
            if (!essayText) {
                showError('Please enter your essay text.');
                return;
            }
            formData.append('essay_text', essayText);
        } else {
            if (!selectedFile) {
                showError('Please select a file to upload.');
                return;
            }
            formData.append('essay_file', selectedFile);
        }

        const response = await fetch('/submit_essay', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.success) {
            const htmlFeedback = marked.parse(result.feedback);
            document.getElementById('feedbackContent').innerHTML = `<div class="feedback-content">${htmlFeedback}</div>`;
            isEssaySubmitted = true;
            enableChat();
        } else {
            showError(result.error || 'An error occurred while analyzing your essay.');
        }
    } catch (error) {
        showError('Network error. Please try again.');
        console.error('Error:', error);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
}

function enableChat() {
    const chatInput = document.getElementById('chatInput');
    const chatSendBtn = document.getElementById('chatSendBtn');
    const chatMessages = document.getElementById('chatMessages');
    
    chatInput.disabled = false;
    chatSendBtn.disabled = false;
    chatInput.placeholder = 'Ask about your essay feedback...';
    
    // Clear empty state
    chatMessages.innerHTML = '';
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

function addMessageToChat(message, isUser = false) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    if (isUser) {
        // User messages: display as plain text
        messageDiv.textContent = message;
    } else {
        // Assistant messages: render Markdown
        const html = DOMPurify.sanitize(marked.parse(message));
        messageDiv.innerHTML = html;
    }
    // messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <span>Writing tutor is typing</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

async function sendChatMessage() {
    const chatInput = document.getElementById('chatInput');
    const chatSendBtn = document.getElementById('chatSendBtn');
    const message = chatInput.value.trim();
    
    if (!message || !isEssaySubmitted) return;
    
    // Add user message
    addMessageToChat(message, true);
    chatInput.value = '';
    chatSendBtn.disabled = true;
    addTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const result = await response.json();
        
        if (result.success) {
            addMessageToChat(result.response, false);
        } else {
            addMessageToChat('Sorry, I encountered an error. Please try again.', false);
        }
    } catch (error) {
        addMessageToChat('Network error. Please try again.', false);
        console.error('Chat error:', error);
    } finally {
        removeTypingIndicator();
        chatSendBtn.disabled = false;
    }
}

async function clearChat() {
    try {
        const response = await fetch('/clear_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            document.getElementById('chatMessages').innerHTML = '';
        }
    } catch (error) {
        console.error('Error clearing chat:', error);
    }
}

// Auto-resize chat input
document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
});