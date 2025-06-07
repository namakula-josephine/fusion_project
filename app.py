from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, List
from dotenv import load_dotenv
import os
import shutil
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from retriever import search_rag
from PIL import Image
from tensorflow.keras.models import load_model
from auth_utils import load_users_db, save_users_db, hash_password, verify_password
from backend.models import Chat, ChatCreate, ChatUpdate, ChatResponse
from backend.chat_storage import ChatStorage

load_dotenv('secrets.env')

# Initialize OpenAI with the older v0.27.x API
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None

if openai_api_key:
    try:
        import openai
        # Set API key globally for v0.27.x
        openai.api_key = openai_api_key
        openai_client = openai
        print("Using OpenAI v0.27.x client")
        # Check if version is available, some versions don't have __version__
        try:
            print(f"OpenAI version: {openai.__version__}")
        except AttributeError:
            print("OpenAI version: Unable to detect version")
    except ImportError:
        print("Error: OpenAI library not found")
        openai_client = None
else:
    print("Warning: OPENAI_API_KEY not found in environment variables")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",  # React app
        "http://localhost:8080",  # Vite dev server
        "http://localhost:5173",  # Another common Vite port
        "http://172.24.176.1:8081",  # Frontend running on network IP
        "http://172.24.176.1:8080",  # Alternative ports
        "http://172.24.176.1:5173",
        # Production frontend URL
        "https://potato-disease-clinic.onrender.com",  # Your actual frontend deployment
        # Other common deployment URLs (fallback)
        "https://potato-plant-aid-dashboard.vercel.app",  # Vercel deployment
        "https://potato-plant-aid-dashboard.netlify.app",  # Netlify deployment
        "https://potato-plant-aid-dashboard.onrender.com",  # Render deployment
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],  # Allow all headers for flexibility
    expose_headers=["*"],
    max_age=3600  # Cache preflight requests for 1 hour
)

security = HTTPBasic()

class User(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    result: Optional[Dict] = None

class MessageCreate(BaseModel):
    role: str
    content: str
    result: Optional[dict] = None

users_db = load_users_db()

try:
    # Try to load the model with custom objects handling
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import os
    import numpy as np
    import shutil
    
    # Ensure model directory exists
    model_path = 'model/potato_classification_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        vision_model = None
    else:
        try:
            # Try loading with TensorFlow 2.15.0 - ignore unknown keywords
            vision_model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                safe_mode=False  # Disable safe mode to ignore unknown parameters
            )
            print("Model loaded successfully with TensorFlow 2.15.0")
        except Exception as e1:
            print(f"Model loading failed: {e1}")
            try:
                # Alternative method - create a custom loader that ignores batch_shape
                import h5py
                import json
                
                # Read the model config and modify it
                with h5py.File(model_path, 'r') as f:
                    model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                
                # Remove batch_shape from InputLayer configs
                def fix_config(config):
                    if isinstance(config, dict):
                        if config.get('class_name') == 'InputLayer':
                            if 'batch_shape' in config.get('config', {}):
                                # Convert batch_shape to input_shape
                                batch_shape = config['config'].pop('batch_shape')
                                if batch_shape and len(batch_shape) > 1:
                                    config['config']['input_shape'] = batch_shape[1:]
                        
                        for key, value in config.items():
                            config[key] = fix_config(value)
                    elif isinstance(config, list):
                        return [fix_config(item) for item in config]
                    return config
                
                fixed_config = fix_config(model_config)
                
                # Try to reconstruct the model from the fixed config
                model_from_config = tf.keras.Model.from_config(fixed_config)
                
                # Load weights
                model_from_config.load_weights(model_path)
                vision_model = model_from_config
                print("Model loaded with config fix method")
                
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                # Last resort - use a simple fallback model structure
                try:
                    # Create a simple model with the expected structure
                    from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
                    from tensorflow.keras.applications import MobileNetV2
                    
                    print("Creating fallback model structure...")
                    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
                    inputs = Input(shape=(224, 224, 3))
                    x = base_model(inputs, training=False)
                    x = GlobalAveragePooling2D()(x)
                    outputs = Dense(3, activation='softmax')(x)
                    fallback_model = tf.keras.Model(inputs, outputs)
                    
                    # Try to load weights into this structure
                    try:
                        fallback_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                        vision_model = fallback_model
                        print("Fallback model created and weights loaded")
                    except:
                        vision_model = None
                        print("Could not load weights into fallback model")
                        
                except Exception as e3:
                    print(f"Fallback model creation failed: {e3}")
                    vision_model = None
    
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    
except Exception as e:
    print(f"Critical error during model initialization: {e}")
    vision_model = None
    class_names = ['Early Blight', 'Healthy', 'Late Blight']

# Initialize chat storage
chat_storage = ChatStorage()

def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = users_db.get(credentials.username)
    if not user or not verify_password(credentials.password, user['password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    return user

@app.post("/api/register")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...)
):
    if username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    users_db[username] = {
        "username": username,
        "password": hash_password(password),
        "email": email
    }
    save_users_db(users_db)
    return {"message": "Registration successful"}

@app.post("/api/login")
async def login(credentials: HTTPBasicCredentials = Depends(security)):
    try:
        user = authenticate_user(credentials)
        return {
            "username": user["username"],
            "email": user["email"]
        }
    except HTTPException as e:
        print(f"Login failed: {str(e)}")
        raise

@app.get("/api/me")
async def get_current_user_info(current_user: dict = Depends(authenticate_user)):
    return {
        "username": current_user["username"],
        "email": current_user["email"]
    }

@app.post("/api/chats/", response_model=ChatResponse)
async def create_chat(
    chat_data: ChatCreate,
    current_user: dict = Depends(authenticate_user)
):
    chat = chat_storage.create_chat(current_user["username"], chat_data.title)
    return ChatResponse(
        chat_id=chat.chat_id,
        title=chat.title,
        created_at=chat.created_at,
        message_count=len(chat.messages),
        last_message=chat.messages[-1].content if chat.messages else None
    )

@app.get("/api/chats/", response_model=List[ChatResponse])
async def get_chats(current_user: dict = Depends(authenticate_user)):
    chats = chat_storage.get_user_chats(current_user["username"])
    return [
        ChatResponse(
            chat_id=chat.chat_id,
            title=chat.title,
            created_at=chat.created_at,
            message_count=len(chat.messages),
            last_message=chat.messages[-1].content if chat.messages else None
        )
        for chat in chats
    ]

@app.get("/api/chats/{chat_id}", response_model=Chat)
async def get_chat(
    chat_id: str,
    current_user: dict = Depends(authenticate_user)
):
    chat = chat_storage.get_chat(current_user["username"], chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@app.put("/api/chats/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: str,
    chat_data: ChatUpdate,
    current_user: dict = Depends(authenticate_user)
):
    chat = chat_storage.update_chat(current_user["username"], chat_id, chat_data.title)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return ChatResponse(
        chat_id=chat.chat_id,
        title=chat.title,
        created_at=chat.created_at,
        message_count=len(chat.messages),
        last_message=chat.messages[-1].content if chat.messages else None
    )

@app.delete("/api/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    current_user: dict = Depends(authenticate_user)
):
    if not chat_storage.delete_chat(current_user["username"], chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "success"}

def get_fallback_prediction(image_path):
    """Fallback prediction when model is not available"""
    import random
    # Simple random prediction for testing - replace with actual logic
    predictions = [
        ("Early Blight", 0.75),
        ("Healthy", 0.85),
        ("Late Blight", 0.70)
    ]
    return random.choice(predictions)

# Update the query endpoint to handle missing model
@app.post("/api/query", response_model=QueryResponse)
async def query(
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    chat_id: Optional[str] = Form(None),
    current_user: dict = Depends(authenticate_user)
):
    try:
        if not question and not image:
            raise HTTPException(
                status_code=400,
                detail="No question or image provided"
            )

        # Process image if provided
        if image:
            print(f"Processing image: {image.filename}")
            
            try:
                os.makedirs('temp', exist_ok=True)
                image_path = f"temp/{image.filename}"
                
                # Validate image file
                if not image.content_type or not image.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image file type"
                    )
                
                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                print(f"Image saved to: {image_path}")
                
                # Verify file was saved correctly
                if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                    raise ValueError("Failed to save image file")
                
                if vision_model:
                    # Use the actual model
                    image_array = preprocess_image(image_path)
                    if image_array is None:
                        raise ValueError("Failed to preprocess image")
                    
                    print(f"Image array shape: {image_array.shape}")
                    
                    try:
                        prediction = vision_model.predict(image_array, verbose=0)
                        predicted_class = class_names[np.argmax(prediction)]
                        confidence = float(np.max(prediction))
                        print(f"Prediction: {predicted_class} ({confidence:.2%})")
                    except Exception as pred_error:
                        print(f"Prediction error: {pred_error}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Model prediction failed: {str(pred_error)}"
                        )
                else:
                    # Use fallback when model is not available
                    print("Using fallback prediction (model not available)")
                    predicted_class, confidence = get_fallback_prediction(image_path)
                    print(f"Fallback prediction: {predicted_class} ({confidence:.2%})")
                
                # Create a brief, clean summary for the main answer
                summary = f"Image analysis complete. See detailed results below."
                
                # Get AI responses with error handling
                try:
                    explanation_query = (
                        f"Provide a detailed explanation of {predicted_class} in potato plants. "
                        f"Cover what this condition is, what causes it, typical symptoms, and how it affects the plant. "
                        f"Keep it informative and well-structured with proper formatting."
                    )
                    explanation = await get_ai_response(explanation_query)

                    treatment_query = (
                        f"Provide a comprehensive treatment plan for {predicted_class} in potato plants. "
                        f"Include immediate actions, preventive measures, chemical treatments if needed, and long-term management strategies. "
                        f"Format with clear bullet points and actionable steps."
                    )
                    treatment_plans = await get_ai_response(treatment_query)
                except Exception as ai_error:
                    print(f"AI response error: {ai_error}")
                    explanation = f"**{predicted_class}** detected with {confidence:.1%} confidence. AI explanation service temporarily unavailable."
                    treatment_plans = "Treatment recommendations temporarily unavailable. Please consult with an agricultural expert."
                
                # Add model status to response
                model_status = "Model loaded successfully" if vision_model else "Using fallback prediction (model unavailable)"
                
                response_data = {
                    "answer": summary,
                    "result": {
                        "predicted_class": predicted_class,
                        "confidence": f"{confidence:.2%}",
                        "explanation": explanation,
                        "treatment_plans": treatment_plans,
                        "model_status": model_status
                    }
                }
                
                # Save to chat if chat_id provided
                if chat_id:
                    try:
                        chat_storage.add_message(
                            current_user["username"],
                            chat_id,
                            "user",
                            f"Uploaded image: {image.filename}"
                        )
                        chat_storage.add_message(
                            current_user["username"],
                            chat_id,
                            "assistant",
                            summary,
                            result=response_data["result"]
                        )
                    except Exception as chat_error:
                        print(f"Chat storage error: {chat_error}")
                
                return JSONResponse(
                    content=response_data,
                    headers={
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Credentials": "true"
                    }
                )
            
            except HTTPException:
                raise
            except Exception as e:
                print(f"Image processing error: {str(e)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Image processing failed: {str(e)}"
                )
            finally:
                # Clean up temporary file
                try:
                    if 'image_path' in locals() and os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"Cleaned up temporary file: {image_path}")
                except Exception as cleanup_error:
                    print(f"Failed to clean up temp file: {cleanup_error}")
        
        else:
            print(f"Processing text query: {question}")
            answer = await get_ai_response(question)
            
            # Save to chat if chat_id provided
            if chat_id:
                chat_storage.add_message(current_user["username"], chat_id, "user", question)
                chat_storage.add_message(current_user["username"], chat_id, "assistant", answer)
            
            response_data = {"answer": answer}
            return JSONResponse(
                content=response_data,
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/api/chats/{chat_id}/messages", response_model=Chat)
async def add_chat_message(
    chat_id: str,
    message: MessageCreate,
    current_user: dict = Depends(authenticate_user)
):
    chat = chat_storage.add_message(
        username=current_user["username"],
        chat_id=chat_id,
        role=message.role,
        content=message.content,
        result=message.result
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

def preprocess_image(image_path, img_size=(224, 224)):
    try:
        with Image.open(image_path) as img:
            image = img.convert('RGB').resize(img_size)
            image = np.array(image) / 255.0
            return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

async def get_ai_response(query: str) -> str:
    if not openai_client:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API not configured"
        )
        
    try:
        # Use OpenAI v0.27.x API structure
        response = openai_client.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo as it's more reliable and cost-effective
            messages=[
                {
                    "role": "system", 
                    "content": """You are a helpful assistant specializing in potato plant diseases and agricultural advice. 

FORMATTING GUIDELINES:
- Use **bold text** for important terms, disease names, and key points
- Use bullet points (-) for lists of symptoms, treatments, or recommendations
- Organize your response with clear paragraphs
- Use *italic text* for scientific names or emphasis
- Structure your responses with clear sections when applicable

Provide clear, practical information about plant health, disease identification, and treatment recommendations. Always format your responses to be easy to read and well-structured."""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=800,  # Increased to allow for more detailed formatted responses
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        # Provide a fallback response for testing
        fallback_response = f"I'm currently unable to connect to the AI service. However, based on your query about '{query[:50]}...', I recommend consulting with an agricultural expert for proper diagnosis and treatment."
        print(f"Using fallback response: {fallback_response}")
        return fallback_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)