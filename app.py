from flask import Flask, request, render_template, jsonify, session
import os
import tempfile
import pandas as pd
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import uuid
import json
import redis
from flask_session import Session
import logging
from logging.handlers import RotatingFileHandler
import pickle
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from pathlib import Path
from autogen import ConversableAgent
from werkzeug.middleware.proxy_fix import ProxyFix
import sys
from functools import wraps
import time
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Configure logging for production
def setup_logging(app):
    """Setup production logging"""
    if not app.debug and not app.testing:
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        # File handler for errors
        file_handler = RotatingFileHandler('logs/essay_app.log', maxBytes=10240000, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s'
        ))
        console_handler.setLevel(logging.INFO)
        app.logger.addHandler(console_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Essay application startup')

def create_app():
    """Application factory pattern"""
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    
    # Production configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True
    app.config['SESSION_KEY_PREFIX'] = 'essay_app:'
    app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_ENV') == 'production'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
    
    # Trust proxy headers (important for Railway)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Setup logging
    setup_logging(app)
    
    # Redis configuration with Railway support
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        # Parse Redis URL for Railway
        parsed = urlparse(redis_url)
        redis_config = {
            'host': parsed.hostname,
            'port': parsed.port,
            'password': parsed.password,
            'decode_responses': False,
            'socket_connect_timeout': 30,
            'socket_timeout': 30,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        if parsed.scheme == 'rediss':
            redis_config['ssl'] = True
            redis_config['ssl_cert_reqs'] = None
    else:
        # Fallback configuration
        redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD', None),
            'decode_responses': False,
            'socket_connect_timeout': 30,
            'socket_timeout': 30,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
    
    try:
        redis_client = redis.Redis(**redis_config)
        redis_client.ping()
        app.config['SESSION_REDIS'] = redis_client
        app.logger.info("Successfully connected to Redis")
    except Exception as e:
        app.logger.error(f"Redis connection failed: {str(e)}")
        app.logger.info("Falling back to filesystem sessions")
        app.config['SESSION_TYPE'] = 'filesystem'
        app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()
    
    # Initialize Flask-Session
    try:
        Session(app)
        app.logger.info("Flask-Session initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing Flask-Session: {str(e)}")
        raise
    
    return app

# Create app instance
app = create_app()
logger = app.logger

# Environment validation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY not found in environment. Set it before running.")

# Rate limiting decorator
def rate_limit(max_requests=10, window=60):
    """Simple rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple in-memory rate limiting (use Redis for production scaling)
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            current_time = time.time()
            
            # This is a simplified rate limiter - in production, use Redis
            if not hasattr(app, 'rate_limit_storage'):
                app.rate_limit_storage = {}
            
            if client_ip in app.rate_limit_storage:
                requests = app.rate_limit_storage[client_ip]
                # Clean old requests
                requests = [req_time for req_time in requests if current_time - req_time < window]
                
                if len(requests) >= max_requests:
                    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
                
                requests.append(current_time)
                app.rate_limit_storage[client_ip] = requests
            else:
                app.rate_limit_storage[client_ip] = [current_time]
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def process_csv_to_documents():
    """Process CSV file to create document embeddings"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Check if CSV file exists
        csv_path = 'essay_feedback_scores.csv'
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Processing {len(df)} rows from CSV file")
        
        for index, row in df.iterrows():
            filename = f"feedback_{int(float(str(index))) + 1}.txt"
            file_path = os.path.join(temp_dir, filename)
            document_content = f"""
Essay Topic: {row['essay_topic']}

Student Essay: {row['essay_content']}

Feedback: {row['feedback']}

Score: {row['score']}/10
            """
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(document_content.strip())
        
        logger.info(f"Successfully processed {len(df)} documents")
        return temp_dir, len(df)
    
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        raise Exception(f"Error processing CSV file: {str(e)}")

# Initialize document processing
try:
    docs_path, num_documents = process_csv_to_documents()
    logger.info(f"Documents initialized successfully: {num_documents} documents")
except Exception as e:
    logger.error(f"Failed to initialize documents: {str(e)}")
    docs_path = None
    num_documents = 0

def create_chat_agent():
    """Create a specialized chat agent for conversational interactions"""
    config_list = [{
        "model": "llama-3.3-70b-versatile",
        "api_key": GROQ_API_KEY,
        "api_type": "groq"
    }]
    
    chat_agent = AssistantAgent(
        name="essay_chat_assistant",
        system_message="""You are an expert writing tutor and teacher engaged in a helpful conversation with a student about their essay and writing in general. You have access to the student's essay and the detailed feedback that was provided.

Your role is to:
1. Answer questions about the feedback provided
2. Clarify any points the student doesn't understand
3. Provide additional writing tips and techniques
4. Help the student improve their writing skills
5. Engage in natural, supportive conversation about writing

Guidelines:
- Be conversational, supportive, and encouraging
- Reference the specific essay and feedback when relevant
- Provide concrete, actionable advice
- Ask follow-up questions to better understand the student's needs
- Keep responses concise but informative (2-4 sentences typically)
- Be patient and explain concepts clearly
- Encourage the student to ask more questions
- If asked about something not related to writing/essays, gently redirect to writing topics

Remember: You're having a conversation, not writing a formal analysis. Be natural, helpful, and engaging.""",
        llm_config={"config_list": config_list, "temperature": 0.5},
    )
    
    return chat_agent

def safe_session_get(key, default=None):
    """Safely get session data with error handling"""
    try:
        return session.get(key, default)
    except Exception as e:
        logger.warning(f"Error accessing session key {key}: {e}")
        return default

def safe_session_set(key, value):
    """Safely set session data with error handling"""
    try:
        session[key] = value
        return True
    except Exception as e:
        logger.warning(f"Error setting session key {key}: {e}")
        return False

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    try:
        # Check Redis connection if configured
        if app.config.get('SESSION_REDIS'):
            app.config['SESSION_REDIS'].ping()
        
        # Check if documents are loaded
        if docs_path is None:
            return jsonify({'status': 'unhealthy', 'error': 'Documents not loaded'}), 503
        
        return jsonify({
            'status': 'healthy',
            'documents_loaded': num_documents,
            'session_type': app.config.get('SESSION_TYPE'),
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.route('/')
def index():
    """Main page route"""
    try:
        # Initialize session variables if not present
        if 'chat_history' not in session:
            session['chat_history'] = []
        if 'current_essay' not in session:
            session['current_essay'] = ''
        if 'current_feedback' not in session:
            session['current_feedback'] = ''
        if 'essay_history' not in session:
            session['essay_history'] = []
            
        return render_template('index.html')
    
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        # Clear session and try again
        session.clear()
        session['chat_history'] = []
        session['current_essay'] = ''
        session['current_feedback'] = ''
        session['essay_history'] = []
        return render_template('index.html')

@app.route('/submit_essay', methods=['POST'])
@rate_limit(max_requests=5, window=300)  # 5 requests per 5 minutes
def submit_essay():
    """Submit essay for analysis"""
    try:
        if docs_path is None:
            return jsonify({'error': 'Service temporarily unavailable. Please try again later.'}), 503
        
        user_essay = ""
        
        # Handle pasted essay or uploaded file
        input_method = request.form.get('input_method')
        if input_method == "paste":
            user_essay = request.form.get('essay_text', '').strip()
        else:
            uploaded_file = request.files.get('essay_file')
            if uploaded_file and uploaded_file.filename:
                file_ext = uploaded_file.filename.split('.')[-1].lower()
                
                # File size check (limit to 10MB)
                uploaded_file.seek(0, 2)  # Seek to end
                file_size = uploaded_file.tell()
                uploaded_file.seek(0)  # Reset to beginning
                
                if file_size > 10 * 1024 * 1024:  # 10MB limit
                    return jsonify({'error': 'File too large. Maximum size is 10MB.'})
                
                try:
                    if file_ext in ['txt', 'tex']:
                        user_essay = uploaded_file.read().decode('utf-8')
                    elif file_ext == 'pdf':
                        pdf_reader = PyPDF2.PdfReader(uploaded_file.stream)
                        user_essay = ""
                        for page in pdf_reader.pages:
                            user_essay += page.extract_text() or ""
                    elif file_ext == 'docx':
                        doc = Document(uploaded_file.stream)
                        user_essay = "\n".join([para.text for para in doc.paragraphs])
                    else:
                        return jsonify({'error': 'Unsupported file format. Please upload a .txt, .pdf, .docx, or .tex file.'})
                except Exception as e:
                    logger.error(f"Error reading uploaded file: {str(e)}")
                    return jsonify({'error': f'Error reading uploaded file: {str(e)}'})
            else:
                return jsonify({'error': 'No file uploaded.'})

        if not user_essay:
            return jsonify({'error': 'Please provide an essay by pasting text or uploading a file.'})

        # Word count check
        word_count = len(user_essay.split())
        if word_count > 5000:
            return jsonify({'error': 'Essay too long. Maximum 5000 words allowed.'})

        # Configure agents
        config_list = [{
            "model": "llama-3.3-70b-versatile",
            "api_key": GROQ_API_KEY,
            "api_type": "groq"
        }]

        assistant = AssistantAgent(
            name="writing_teacher",
            system_message="""You are a helpful writing teacher who gives feedback to school students. who first evaluates whether a submitted text is a valid essay and then provides detailed, structured feedback if appropriate. Your feedback is constructive, honest, and educational, avoiding harsh grading while providing clear areas for improvement.

                **Step 1: Essay Validation**
                Before generating feedback, assess if the submitted text is a valid essay by checking:
                - Length: At least 150 words to ensure sufficient content.
                - Structure: Presence of an introduction, body, and conclusion (or similar organization).
                - Coherence: Logical flow and relevance to a clear topic or argument.
                - Content: Academic or argumentative focus, not random text, notes, or unrelated content.
                If the text is not a valid essay, provide a brief explanation of why and suggest how to correct it. Do not proceed to feedback in this case.

                **Step 2: Feedback Generation (if valid)**
                You are provided with previous essay feedback examples in a CSV file. Study the feedback patterns, scoring criteria, and how essay titles and content influence the feedback and scores. Use these examples as a reference for the tone, style, and structure of your response. Your goal is to mimic the teacher's review style shown in the examples while adapting it to the student's essay.

                You are a helpful writing teacher who gives feedback to school students. 

                Write in simple, easy-to-understand English. Use short sentences and common words that students can easily understand. Avoid big, complicated words.

                When giving feedback:
                - Be encouraging and positive
                - Use simple language 
                - Give clear examples
                - Make suggestions easy to follow
                - Be patient and kind

                Remember: You're helping students learn to write better, so keep everything simple and clear.

                **Important Rules:**
                - For validation, clearly state if the text is a valid essay or not and why.
                - For feedback, analyze how essay titles and content in the examples correlate with their feedback and scores, and apply similar reasoning.
                - Be respectful, clear, and informative — your tone should be professional but supportive.
                - Avoid repeating the student's full essay.
                - Do not invent unrelated facts or feedback.
                - Focus on helping the student grow as a writer.
                - Write the feedback in a way that is easy to understand and actionable.
                - Focus on main points without excessive detail.
                Only reply with the validation result and (if applicable) the feedback.
            """,
            llm_config={"config_list": config_list, "temperature": 0.5},
        )

        ragproxyagent = RetrieveUserProxyAgent(
            name="essay_analyzer",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            retrieve_config={
                "task": "qa",
                "docs_path": docs_path,
                "chunk_token_size": 2000,
                "model": config_list[0]["model"],
                "api_key": GROQ_API_KEY,
                "api_type": "groq",
                "get_or_create": True,
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_db": "chroma",
                "update_context": True,
                "distance_threshold": 0.3,
            },
            code_execution_config=False,
        )

        feedback_prompt = f"""
First, evaluate whether the following text is a valid essay by checking:
- Length: At least 150 words.
- Structure: Presence of an introduction, body, and conclusion (or similar organization).
- Coherence: Logical flow and relevance to a clear topic or argument.
- Content: Academic or argumentative focus, not random text, notes, or unrelated content.

If the text is not a valid essay, explain why and suggest how to correct it. If it is a valid essay, provide structured feedback based on the essay feedback examples in the provided CSV. Study the CSV examples to understand how essay titles and content influence the feedback structure and scores. Base your feedback and score on the patterns, scoring criteria, and feedback styles in those examples, focusing on clarity, structure, depth, and persuasiveness.

STUDENT ESSAY:
"{user_essay}"

If the essay is valid, provide feedback in the following structure:
1. **Strengths:** What the student did well
2. **Areas for Improvement:** What needs work
3. **Specific Suggestions:** Concrete recommendations for improvement
4. **Suggested Score:** A score out of 10 with brief reasoning, consistent with the examples

If the essay is not valid, provide only the validation result and correction suggestions.
        """

        response = ragproxyagent.initiate_chat(
            assistant,
            message=feedback_prompt,
            max_turns=3,
            silent=False
        )

        feedback = None
        if hasattr(assistant, 'last_message') and assistant.last_message:
            last_message = assistant.last_message()
            feedback = last_message.get('content', '') if isinstance(last_message, dict) else ''
        elif hasattr(assistant, 'chat_messages') and assistant.chat_messages:
            for msg in reversed(assistant.chat_messages):
                if hasattr(msg, 'name') and msg.name == 'writing_teacher':
                    feedback = getattr(msg, 'content', '')
                    break

        if feedback and feedback.strip():
            # Store in session for chat context
            safe_session_set('current_essay', user_essay)
            safe_session_set('current_feedback', feedback)
            
            reset_chat = request.form.get('reset_chat') == 'true'
            if reset_chat:
                safe_session_set('chat_history', [])
            
            # Save to essay history
            essay_history = safe_session_get('essay_history', [])
            if essay_history is None:
                essay_history = []
            essay_history.append({
                'essay': user_essay,
                'feedback': feedback
            })
            safe_session_set('essay_history', essay_history)
            
            logger.info(f"Essay submitted successfully. Word count: {word_count}")
            return jsonify({
                'success': True,
                'feedback': feedback
            })
        else:
            return jsonify({'error': 'No analysis or feedback generated. Please try again.'})

    except Exception as e:
        logger.error(f"Error in submit_essay: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error generating feedback. Please try again later.'}), 500

@app.route('/chat', methods=['POST'])
@rate_limit(max_requests=20, window=300)  # 20 requests per 5 minutes
def chat():
    """Chat endpoint for conversational interaction"""
    try:
        if docs_path is None:
            return jsonify({'error': 'Service temporarily unavailable. Please try again later.'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'})
        
        if len(user_message) > 1000:
            return jsonify({'error': 'Message too long. Maximum 1000 characters allowed.'})
        
        # Get context from session
        current_essay = safe_session_get('current_essay', '')
        current_feedback = safe_session_get('current_feedback', '')
        chat_history = safe_session_get('chat_history', [])
        essay_history = safe_session_get('essay_history', [])
        
        if not current_essay or not current_feedback:
            return jsonify({'error': 'Please submit an essay first to start chatting'})
        
        # Create chat agent
        chat_agent = create_chat_agent()
        
        # Build conversation context
        conversation_context = f"""
CONTEXT:
Student's Essay: {current_essay[:500]}...  # Truncated for context

Current Feedback Provided: {current_feedback}

Previous Essay History: {essay_history[-3:] if essay_history else []}  # Last 3 essays

Chat History: {chat_history[-10:] if chat_history else []}  # Last 10 messages
"""
        
        # Add current user message
        conversation_context += f"\nSTUDENT: {user_message}\n\nPlease respond as the writing tutor, considering the essay, feedback, and conversation history above."
        
        # Create proxy agent
        config_list = [{
            "model": "llama-3.3-70b-versatile",
            "api_key": GROQ_API_KEY,
            "api_type": "groq"
        }]
        
        proxy_agent = RetrieveUserProxyAgent(
            name="chat_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            retrieve_config={
                "task": "qa",
                "docs_path": docs_path,
                "chunk_token_size": 2000,
                "model": config_list[0]["model"],
                "api_key": GROQ_API_KEY,
                "api_type": "groq",
                "get_or_create": True,
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_db": "chroma",
                "update_context": True,
                "distance_threshold": 0.3,
            },
            code_execution_config=False,
        )
        
        # Generate response
        response = proxy_agent.initiate_chat(
            chat_agent,
            message=conversation_context,
            max_turns=3,
            silent=False
        )
        
        # Extract AI response
        ai_response = None
        if hasattr(chat_agent, 'last_message') and chat_agent.last_message:
            last_message = chat_agent.last_message()
            ai_response = last_message.get('content', '') if isinstance(last_message, dict) else ''
        elif hasattr(chat_agent, 'chat_messages') and chat_agent.chat_messages:
            for msg in reversed(chat_agent.chat_messages):
                if hasattr(msg, 'name') and msg.name == 'essay_chat_assistant':
                    ai_response = getattr(msg, 'content', '')
                    break
        
        if not ai_response:
            ai_response = "I apologize, but I'm having trouble generating a response. Could you please try rephrasing your question?"
        
        # Update chat history
        chat_history = chat_history or []
        chat_history.append({'role': 'user', 'content': user_message})
        chat_history.append({'role': 'assistant', 'content': ai_response})
        
        # Keep only last 20 messages to prevent session bloat
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        safe_session_set('chat_history', chat_history)
        
        return jsonify({
            'success': True,
            'response': ai_response
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error in chat. Please try again later.'}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear the chat history"""
    try:
        safe_session_set('chat_history', [])
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return jsonify({'error': 'Error clearing chat. Please try again.'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500

@app.errorhandler(503)
def service_unavailable(error):
    """Handle service unavailable errors"""
    return jsonify({'error': 'Service temporarily unavailable. Please try again later.'}), 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port)