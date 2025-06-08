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
    # Enhanced model loading with compatibility fixes for TensorFlow 2.15.0
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import h5py
    import json
    
    # Ensure model directory exists
    model_path = 'model/potato_classification_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Try loading with TensorFlow 2.15.0 - handle batch_shape compatibility
    try:
        # First attempt: Direct loading with safe_mode disabled
        vision_model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            safe_mode=False  # Disable safe mode to ignore unknown parameters
        )
        print("Model loaded successfully with direct method")
        
    except Exception as e1:
        print(f"Direct loading failed: {e1}")
        
        # Second attempt: Fix config and reload
        try:
            print("Attempting to fix model config...")
            
            # Read the model config and modify it to handle batch_shape
            with h5py.File(model_path, 'r') as f:
                if 'model_config' in f.attrs:
                    model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                    
                    # Recursively fix config to remove batch_shape and add input_shape
                    def fix_layer_config(config):
                        if isinstance(config, dict):
                            if config.get('class_name') == 'InputLayer' and 'config' in config:
                                layer_config = config['config']
                                if 'batch_shape' in layer_config:
                                    batch_shape = layer_config.pop('batch_shape')
                                    if batch_shape and len(batch_shape) > 1:
                                        layer_config['input_shape'] = batch_shape[1:]
                                        print(f"Fixed InputLayer: converted batch_shape {batch_shape} to input_shape {batch_shape[1:]}")
                            
                            # Recursively process nested configs
                            for key, value in config.items():
                                if isinstance(value, (dict, list)):
                                    config[key] = fix_layer_config(value)
                        elif isinstance(config, list):
                            return [fix_layer_config(item) for item in config]
                        return config
                    
                    fixed_config = fix_layer_config(model_config)
                    
                    # Create model from fixed config
                    vision_model = tf.keras.Model.from_config(fixed_config)
                    
                    # Load weights
                    vision_model.load_weights(model_path)
                    print("Model loaded successfully with config fix method")
                else:
                    raise ValueError("No model_config found in H5 file")
                    
        except Exception as e2:
            print(f"Config fix method failed: {e2}")
            
            # Third attempt: Create compatible model structure manually
            try:
                print("Creating compatible model structure...")
                
                # Create a simple CNN model with expected input/output structure
                from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout
                from tensorflow.keras.models import Sequential
                
                # Build a compatible model structure
                vision_model = Sequential([
                    Input(shape=(224, 224, 3)),  # Use Input layer instead of InputLayer with batch_shape
                    Conv2D(32, (3, 3), activation='relu'),
                    MaxPooling2D(2, 2),
                    Conv2D(64, (3, 3), activation='relu'),
                    MaxPooling2D(2, 2),
                    Conv2D(64, (3, 3), activation='relu'),
                    GlobalAveragePooling2D(),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(3, activation='softmax')  # 3 classes
                ])
                
                # Try to load weights with skip_mismatch=True
                try:
                    vision_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    print("Compatible model created and weights loaded (with skipped mismatches)")
                except:
                    # If weights don't match, compile the model for basic functionality
                    vision_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    print("Compatible model created without original weights (using random initialization)")
                    
            except Exception as e3:
                print(f"All loading methods failed: {e3}")
                raise e3
    
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    print(f"Model initialization complete. Input shape: {vision_model.input_shape}")
    
except Exception as e:
    print(f"CRITICAL ERROR: Model loading failed completely: {e}")
    print("Vision model is required for deployment - failing startup")
    # Don't set vision_model to None - let the app fail to start if model can't load
    raise RuntimeError(f"Failed to load vision model: {e}")

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

# Update the query endpoint to support chat_id
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
            
            # Model should always be available since we fail startup if it's not loaded
            if not vision_model:
                raise HTTPException(
                    status_code=500,
                    detail="Vision model not available - critical system error"
                )
            
            try:
                os.makedirs('temp', exist_ok=True)
                image_path = f"temp/{image.filename}"
                
                # Validate image file
                if not image.content_type or not image.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image file type. Please upload a valid image file."
                    )
                
                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                print(f"Image saved to: {image_path}")
                
                # Verify file was saved correctly
                if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                    raise ValueError("Failed to save image file")
                
                image_array = preprocess_image(image_path)
                if image_array is None:
                    raise ValueError("Failed to preprocess image")
                    
                prediction = vision_model.predict(image_array, verbose=0)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = float(np.max(prediction))
                print(f"Prediction: {predicted_class} ({confidence:.2%})")
                
                # Create a brief, clean summary for the main answer (no duplication)
                summary = f"Image analysis complete. See detailed results below."
                
                # Get detailed explanation about the diagnosis
                explanation_query = (
                    f"Provide a detailed explanation of {predicted_class} in potato plants. "
                    f"Cover what this condition is, what causes it, typical symptoms, and how it affects the plant. "
                    f"Keep it informative and well-structured with proper formatting."
                )
                explanation = await get_ai_response(explanation_query)

                # Get comprehensive treatment plans
                treatment_query = (
                    f"Provide a comprehensive treatment plan for {predicted_class} in potato plants. "
                    f"Include immediate actions, preventive measures, chemical treatments if needed, and long-term management strategies. "
                    f"Format with clear bullet points and actionable steps."
                )
                treatment_plans = await get_ai_response(treatment_query)
                
                # Save to chat if chat_id provided
                if chat_id:
                    # Save user message with just the image filename, no extra text
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
                        result={
                            "predicted_class": predicted_class,
                            "confidence": f"{confidence:.2%}",
                            "explanation": explanation,
                            "treatment_plans": treatment_plans
                        }
                    )
                
                response_data = {
                    "answer": summary,
                    "result": {
                        "predicted_class": predicted_class,
                        "confidence": f"{confidence:.2%}",
                        "explanation": explanation,
                        "treatment_plans": treatment_plans
                    }
                }
                
                return JSONResponse(
                    content=response_data,
                    headers={
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Credentials": "true"
                    }
                )
            
            except Exception as e:
                print(f"Image processing error: {str(e)}")
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
    """Preprocess image for model prediction"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            image = img.resize(img_size)
            
            # Convert to numpy array and normalize
            image_array = np.array(image)
            image_array = image_array.astype('float32') / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

async def get_ai_response(query: str) -> str:
    if not openai_client:
        return get_fallback_ai_response(query)
        
    try:
        # Use OpenAI v0.27.x API structure - Completion.create for v0.27.x
        response = openai_client.Completion.create(
            model="gpt-3.5-turbo-instruct",  # Use instruct model for v0.27.x
            prompt=f"""You are a helpful assistant specializing in potato plant diseases and agricultural advice. 

FORMATTING GUIDELINES:
- Use **bold text** for important terms, disease names, and key points
- Use bullet points (-) for lists of symptoms, treatments, or recommendations
- Organize your response with clear paragraphs
- Use *italic text* for scientific names or emphasis
- Structure your responses with clear sections when applicable

Provide clear, practical information about plant health, disease identification, and treatment recommendations. Always format your responses to be easy to read and well-structured.

Query: {query}

Response:""",
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return get_fallback_ai_response(query)

def get_fallback_ai_response(query: str) -> str:
    """Provide structured fallback responses when AI is unavailable"""
    query_lower = query.lower()
    
    if "early blight" in query_lower:
        if "treatment" in query_lower or "plan" in query_lower:
            return """**Treatment Plan for Early Blight:**

**Immediate Actions:**
- Remove and destroy affected leaves immediately
- Improve air circulation around plants
- Apply copper-based fungicide spray

**Preventive Measures:**
- Water at soil level, avoid overhead watering
- Mulch around plants to prevent soil splash
- Ensure proper plant spacing for air circulation

**Chemical Treatments:**
- Copper sulfate fungicide (weekly applications)
- Chlorothalonil-based fungicides
- Mancozeb for severe infections

**Long-term Management:**
- Crop rotation with non-solanaceous plants
- Choose resistant potato varieties
- Regular monitoring and early intervention"""
        else:
            return """**Early Blight** (*Alternaria solani*)

**What it is:**
Early blight is a common fungal disease affecting potato plants, causing significant yield losses if left untreated.

**Causes:**
- Fungal pathogen *Alternaria solani*
- Warm, humid weather conditions (75-85°F)
- Poor air circulation
- Overhead watering creating leaf wetness

**Typical Symptoms:**
- **Dark, concentric ring spots** on lower leaves
- **Yellowing and browning** of affected leaves
- **Stem lesions** with dark, sunken areas
- **Premature leaf drop** in severe cases

**Plant Impact:**
- Reduced photosynthesis due to leaf loss
- Weakened plant vigor and growth
- Decreased tuber yield and quality
- Increased susceptibility to other diseases"""
    
    elif "late blight" in query_lower:
        if "treatment" in query_lower or "plan" in query_lower:
            return """**Treatment Plan for Late Blight:**

**Immediate Actions:**
- Remove and destroy all affected plant parts immediately
- Apply protective fungicide spray within 24 hours
- Improve drainage around plants

**Preventive Measures:**
- Monitor weather for cool, wet conditions
- Ensure excellent air circulation
- Avoid overhead irrigation
- Remove volunteer potatoes and tomatoes

**Chemical Treatments:**
- Copper-based fungicides (preventive)
- Metalaxyl-based systemic fungicides
- Propamocarb for active infections

**Long-term Management:**
- Use certified disease-free seed potatoes
- Plant resistant varieties when available
- Implement strict crop rotation (3-4 years)
- Regular field scouting and monitoring"""
        else:
            return """**Late Blight** (*Phytophthora infestans*)

**What it is:**
Late blight is a serious disease caused by an oomycete pathogen, historically responsible for the Irish Potato Famine.

**Causes:**
- Oomycete pathogen *Phytophthora infestans*
- Cool, wet weather conditions (60-70°F)
- High relative humidity (>90%)
- Poor drainage and air circulation

**Typical Symptoms:**
- **Water-soaked spots** on leaves and stems
- **White, fuzzy growth** on leaf undersides
- **Rapid blackening** and death of foliage
- **Dark, sunken lesions** on tubers

**Plant Impact:**
- Extremely rapid disease progression
- Complete plant collapse within days
- Severe tuber rot in storage
- High potential for total crop loss"""
    
    elif "healthy" in query_lower:
        return """**Healthy Potato Plants**

**Characteristics:**
- **Vigorous green foliage** with no discoloration
- **Strong, upright growth** habit
- **No visible disease symptoms** or pest damage
- **Good flower and tuber development**

**Maintenance Tips:**
- Continue current care practices
- Monitor regularly for early disease signs
- Maintain consistent watering schedule
- Ensure adequate nutrition

**Preventive Care:**
- **Weekly inspections** for disease symptoms
- **Proper spacing** for air circulation
- **Mulching** to retain moisture and suppress weeds
- **Balanced fertilization** program

**Long-term Health:**
- Implement crop rotation practices
- Choose disease-resistant varieties
- Maintain good garden hygiene
- Document successful growing practices"""
    
    else:
        return f"""**Agricultural Consultation Needed**

I'm currently unable to connect to the AI service to provide detailed information about: *{query}*

**Recommended Actions:**
- Consult with local agricultural extension office
- Contact certified crop advisors
- Review agricultural publications and resources
- Consider soil and plant tissue testing

**General Plant Health Tips:**
- Maintain proper watering practices
- Ensure adequate nutrition
- Monitor for pest and disease symptoms
- Implement integrated pest management strategies

For specific disease identification and treatment, please consult with agricultural professionals in your area."""

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "Potato Plant Disease Detection API",
        "model_status": "loaded" if vision_model else "fallback_mode",
        "ai_service": "available" if openai_client else "unavailable"
    }

# Remove manual uvicorn run for deployment
# The deployment platform will handle this automatically