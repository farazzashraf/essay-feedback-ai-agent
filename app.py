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
import pickle
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from pathlib import Path
from autogen import ConversableAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secure-secret-key-here')
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'essay_app:'

# Redis configuration with better error handling
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD', None),
        decode_responses=False,  # Important: Set to False to handle binary data
        socket_connect_timeout=30,
        socket_timeout=30,
        retry_on_timeout=True,
        health_check_interval=30
    )
    
    # Clear any corrupted session data
    try:
        # Get all keys with the session prefix
        pattern = f"{app.config['SESSION_KEY_PREFIX']}*"
        keys = redis_client.keys(pattern)
        # if keys:
        #     logger.info(f"Found {len(keys)} existing session keys, clearing them...")
        #     redis_client.delete(*keys)
        #     logger.info("Cleared existing session data")
        logger.info(f"Found {keys}")
    except Exception as e:
        logger.warning(f"Could not clear existing sessions: {e}")
    
    # Test Redis connection
    redis_client.ping()
    app.config['SESSION_REDIS'] = redis_client
    logger.info("Successfully connected to Redis")
    
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    # Fallback to filesystem sessions
    logger.info("Falling back to filesystem sessions")
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()
    
except Exception as e:
    logger.error(f"Redis setup error: {str(e)}")
    # Fallback to filesystem sessions
    logger.info("Falling back to filesystem sessions")
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()

# Initialize Flask-Session with error handling
try:
    Session(app)
    logger.info("Flask-Session initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Flask-Session: {str(e)}", exc_info=True)
    # Try to reinitialize with filesystem sessions
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()
    Session(app)
    logger.info("Flask-Session initialized with filesystem fallback")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Set it before running.")

def process_csv_to_documents():
    temp_dir = tempfile.mkdtemp()
    try:
        df = pd.read_csv('essay_feedback_scores.csv')
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
        return temp_dir, len(df)
    except FileNotFoundError:
        raise FileNotFoundError("CSV file 'essay_feedback_scores.csv' not found.")
    except Exception as e:
        raise Exception(f"Error processing CSV file: {str(e)}")

docs_path, num_documents = process_csv_to_documents()

def create_chat_agent():
    """Create a specialized chat agent for conversational interactions"""
    config_list = [{
        "model": "llama-3.3-70b-versatile",
        "api_key": GROQ_API_KEY,
        "api_type": "groq"
    }]
    
    # rag_memory = ChromaDBVectorMemory(
    #     config=PersistentChromaDBVectorMemoryConfig(
    #         collection_name="autogen_docs",
    #         persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
    #         k=3,  # Return top 3 results
    #         score_threshold=0.3,  # Minimum similarity score
    #     )
    # )
    
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
        # memory=[rag_memory]
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

@app.route('/')
def index():
    # Initialize session variables if not present
    try:
        if 'chat_history' not in session:
            session['chat_history'] = []
        if 'current_essay' not in session:
            session['current_essay'] = ''
        if 'current_feedback' not in session:
            session['current_feedback'] = ''
        if 'essay_history' not in session:  
            session['essay_history'] = []   
            
    except Exception as e:
        logger.warning(f"Error initializing session: {e}")
        # Clear session and try again
        session.clear()
        session['chat_history'] = []
        session['current_essay'] = ''
        session['current_feedback'] = ''
        session['essay_history'] = []  # [NEW]
    
    return render_template('index.html')

@app.route('/submit_essay', methods=['POST'])
def submit_essay():
    try:
        user_essay = ""
        error_message = ""
        
        # Handle pasted essay or uploaded file
        input_method = request.form.get('input_method')
        if input_method == "paste":
            user_essay = request.form.get('essay_text', '').strip()
        else:
            uploaded_file = request.files.get('essay_file')
            if uploaded_file:
                if uploaded_file and uploaded_file.filename:
                    file_ext = uploaded_file.filename.split('.')[-1].lower()
                else:
                    return jsonify({'error': 'No file uploaded or invalid file.'})
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
                    return jsonify({'error': f'Error reading uploaded file: {str(e)}'})
            else:
                return jsonify({'error': 'No file uploaded.'})

        if not user_essay:
            return jsonify({'error': 'Please provide an essay by pasting text or uploading a file.'})

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
            # Store in session for chat context with error handling
            safe_session_set('current_essay', user_essay)
            safe_session_set('current_feedback', feedback)
            # safe_session_set('chat_history', [])  # Reset chat history for new essay
            
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
            
            return jsonify({
                'success': True,
                'feedback': feedback
            })
        else:
            return jsonify({'error': 'No analysis or feedback generated. Please try again.'})

    except Exception as e:
        logger.error(f"Error in submit_essay: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error generating feedback: {str(e)}'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'})
        
        # Get context from session with safe access
        current_essay = safe_session_get('current_essay', '')
        current_feedback = safe_session_get('current_feedback', '')
        chat_history = safe_session_get('chat_history', [])
        essay_history = safe_session_get('essay_history', [])  # [NEW]
        
        
        if not current_essay or not current_feedback:
            return jsonify({'error': 'Please submit an essay first to start chatting'})
        
        # Create chat agent
        chat_agent = create_chat_agent()
        
        # Build conversation context
        conversation_context = f"""
CONTEXT:
Student's Essay: {current_essay}  # Truncated for context

Current Feedback Provided: {current_feedback}

Previous Essay History: {essay_history}

Chat History: {chat_history}
"""
        
        # Add recent chat history (last 10 messages to keep context manageable)
        chat_history = chat_history or []  # Ensure chat_history is a list
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        for msg in recent_history:
            conversation_context += f"{msg['role'].upper()}: {msg['content']}\n"
        
        # Add current user message
        conversation_context += f"\nSTUDENT: {user_message}\n\nPlease respond as the writing tutor, considering the essay, feedback, and conversation history above."
        
        # Create a simple proxy agent for context
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
        
        # Update chat history with safe session handling
        chat_history.append({'role': 'user', 'content': user_message})
        chat_history.append({'role': 'assistant', 'content': ai_response})
        safe_session_set('chat_history', chat_history)
        
        return jsonify({
            'success': True,
            'response': ai_response
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error in chat: {str(e)}'})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear the chat history"""
    try:
        safe_session_set('chat_history', [])
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return jsonify({'error': f'Error clearing chat: {str(e)}'})

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    app.run()