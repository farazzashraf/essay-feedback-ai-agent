# Essay Feedback App

An intelligent essay feedback system powered by AI that provides detailed, constructive feedback on student essays and enables interactive conversations about writing improvement.

## Features

### 📝 Essay Analysis
- **Multiple Input Methods**: Submit essays by pasting text or uploading files (.txt, .pdf, .docx, .tex)
- **Smart Validation**: Automatically validates if submitted text is a proper essay (minimum 150 words, proper structure)
- **Detailed Feedback**: Provides structured feedback covering strengths, areas for improvement, and specific suggestions
- **Scoring System**: Assigns scores out of 10 with reasoning based on learned patterns

### 💬 Interactive Chat
- **Conversational Support**: Chat with an AI writing tutor about your essay and feedback
- **Context-Aware**: Maintains conversation history and references your specific essay
- **Writing Guidance**: Get additional tips, clarifications, and writing advice
- **Session Management**: Maintains chat history throughout your session

### 📊 Session Management
- **Essay History**: Keeps track of all essays submitted in the current session
- **Persistent Sessions**: Uses Redis for robust session management with filesystem fallback
- **Chat History**: Maintains conversation context for meaningful interactions

## Technology Stack

- **Backend**: Flask (Python web framework)
- **AI/ML**: AutoGen with Groq API (Llama 3.3 70B model)
- **Vector Database**: ChromaDB for document retrieval
- **Session Management**: Redis with Flask-Session
- **Document Processing**: PyPDF2, python-docx, pandas
- **Frontend**: HTML/CSS/JavaScript (templates)

## Prerequisites

- Python 3.8+
- Redis server (optional, will fallback to filesystem sessions)
- Groq API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd essay-feedback-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   SECRET_KEY=your_secure_secret_key_here
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=your_redis_password  # Optional
   ```

5. **Prepare the training data**
   Ensure you have `essay_feedback_scores.csv` in the root directory with columns:
   - `essay_topic`: The topic/title of the essay
   - `essay_content`: The full essay content
   - `feedback`: Detailed feedback provided
   - `score`: Numerical score (out of 10)

## Required Dependencies

Create a `requirements.txt` file with:

```txt
Flask==2.3.3
Flask-Session==0.5.0
redis==5.0.1
python-dotenv==1.0.0
pandas==2.1.1
PyPDF2==3.0.1
python-docx==0.8.11
pyautogen==0.2.0
chromadb==0.4.15
sentence-transformers==2.2.2
```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the application**
   Open your browser and go to `http://localhost:5000`

3. **Submit an essay**
   - Choose to paste text or upload a file
   - Submit your essay for analysis
   - Receive detailed feedback and scoring

4. **Chat with the AI tutor**
   - Ask questions about the feedback
   - Get writing tips and suggestions
   - Discuss specific aspects of your essay

## API Endpoints

### POST `/submit_essay`
Submit an essay for analysis and feedback.

**Parameters:**
- `input_method`: "paste" or "upload"
- `essay_text`: Essay content (if pasting)
- `essay_file`: File upload (if uploading)
- `reset_chat`: Boolean to reset chat history

**Response:**
```json
{
  "success": true,
  "feedback": "Detailed feedback text..."
}
```

### POST `/chat`
Interactive chat with the AI writing tutor.

**Parameters:**
```json
{
  "message": "Your question or message"
}
```

**Response:**
```json
{
  "success": true,
  "response": "AI tutor's response..."
}
```

### POST `/clear_chat`
Clear the current chat history.

**Response:**
```json
{
  "success": true
}
```

## Configuration

### Redis Setup (Optional)
The app uses Redis for session management but will fallback to filesystem sessions if Redis is unavailable.

**Redis Configuration:**
- Host: Set via `REDIS_HOST` environment variable
- Port: Set via `REDIS_PORT` environment variable
- Password: Set via `REDIS_PASSWORD` environment variable (optional)

### Model Configuration
The app uses Groq's Llama 3.3 70B model. You can modify the model configuration in the `config_list` sections of the code.

## File Structure

```
essay-feedback-app/
├── app.py                 # Main Flask application
├── essay_feedback_scores.csv  # Training data
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
│   └── index.html
└── README.md           # This file
```

## Error Handling

The application includes comprehensive error handling for:
- File upload errors
- Redis connection issues
- Session management problems
- AI model API errors
- Invalid essay submissions

## Security Considerations

- Environment variables for sensitive data
- Secure session management
- File upload validation
- Input sanitization
- Error logging without exposing sensitive information

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis server is running
   - Check Redis configuration in `.env`
   - App will fallback to filesystem sessions automatically

2. **Groq API Issues**
   - Verify your API key is correct
   - Check your API quota/limits
   - Ensure stable internet connection

3. **File Upload Problems**
   - Check file format (supported: .txt, .pdf, .docx, .tex)
   - Ensure file is not corrupted
   - Check file size limits

4. **Missing Training Data**
   - Ensure `essay_feedback_scores.csv` exists
   - Verify CSV has required columns
   - Check data format and encoding

### Logging

The application uses Python's logging module. Check console output for detailed error messages and debugging information.

## Acknowledgments

- Built with AutoGen for AI agent orchestration
- Powered by Groq's Llama 3.3 70B model
- Uses ChromaDB for efficient document retrieval
- Flask framework for web application structure
