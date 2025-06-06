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
        print(f"OpenAI version: {openai.__version__}")
    except ImportError:
        print("Error: OpenAI library not found")
        openai_client = None
else:
    print("Warning: OPENAI_API_KEY not found in environment variables")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's URL
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
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    
    # Try different loading strategies
    try:
        vision_model = load_model('model/potato_classification_model.h5', compile=False)
    except Exception as e1:
        print(f"First load attempt failed: {e1}")
        try:
            # Try with custom objects
            vision_model = tf.keras.models.load_model('model/potato_classification_model.h5', compile=False)
        except Exception as e2:
            print(f"Second load attempt failed: {e2}")
            # Try loading without InputLayer issues
            try:
                # Load model architecture and weights separately if needed
                vision_model = load_model('model/potato_classification_model.h5', compile=False, custom_objects={'InputLayer': tf.keras.layers.InputLayer})
            except Exception as e3:
                print(f"Third load attempt failed: {e3}")
                raise e3
    
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Vision model will be unavailable - image analysis disabled")
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
            
            if not vision_model:
                raise HTTPException(
                    status_code=500,
                    detail="Vision model not available"
                )
            
            try:
                os.makedirs('temp', exist_ok=True)
                image_path = f"temp/{image.filename}"
                
                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                print(f"Image saved to: {image_path}")
                
                image_array = preprocess_image(image_path)
                if image_array is None:
                    raise ValueError("Failed to preprocess image")
                    
                prediction = vision_model.predict(image_array, verbose=0)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = float(np.max(prediction))
                print(f"Prediction: {predicted_class} ({confidence:.2%})")
                
                query_text = (
                    f"The model predicts this potato plant has {predicted_class} "
                    f"with {confidence:.2%} confidence. Please explain what this means "
                    "and suggest specific treatment options."
                )
                explanation = await get_ai_response(query_text)

                # Get treatment plans separately
                treatment_query = f"What are the specific treatment plans for a potato plant with {predicted_class}?"
                treatment_plans = await get_ai_response(treatment_query)
                
                # Save to chat if chat_id provided
                if chat_id:
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
                        explanation,
                        result={
                            "predicted_class": predicted_class,
                            "confidence": f"{confidence:.2%}",
                            "explanation": explanation,
                            "treatment_plans": treatment_plans
                        }
                    )
                
                response_data = {
                    "answer": explanation,
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
                        "Access-Control-Allow-Origin": "http://localhost:3000",
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
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Cleaned up temporary file: {image_path}")
        
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
                    "Access-Control-Allow-Origin": "http://localhost:3000",
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
    
    with Image.open(image_path) as img:
        image = img.convert('RGB').resize(img_size)
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)

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
                    "content": "You are a helpful assistant specializing in potato plant diseases and agricultural advice. Provide clear, practical information about plant health, disease identification, and treatment recommendations."
                },
                {"role": "user", "content": query}
            ],
            max_tokens=500,
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