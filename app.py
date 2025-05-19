from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from retriever import search_rag
import os
import json
import numpy as np
from dotenv import load_dotenv
import openai
import shutil
from datetime import datetime
from typing import List, Optional, Dict
import uuid
from passlib.context import CryptContext

# Load environment variables
load_dotenv('secrets.env')

# Session management
active_sessions = {}  # Store active sessions

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User models
class User(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None

# Login model
class LoginData(BaseModel):
    username: str
    password: str

# Chat models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    result: Optional[Dict] = None

class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: str
    user_id: str

# In-memory storage
users_db = {}
chat_sessions = {}

# Load the vision model
try:
    vision_model = load_model('model/potato_classification_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    vision_model = None

# Define class names for the vision model
class_names = ['Early Blight', 'Healthy', 'Late Blight']

# Authentication functions
def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user:
        return None
    
    # Verify the password against the stored hash
    if not pwd_context.verify(password, user['password']):
        return None
        
    return user

async def get_current_user(request: Request):
    session_id = request.headers.get('authorization')
    if not session_id or session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    username = active_sessions[session_id]
    return users_db[username]

# Helper functions
def preprocess_image(image_path, img_size=(224, 224)):
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            image = img.convert('RGB').resize(img_size)
            image = np.array(image) / 255.0
            return np.expand_dims(image, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_image(image_path):
    if vision_model is None:
        raise HTTPException(status_code=500, detail="Vision model not loaded")
    try:
        image = preprocess_image(image_path)
        prediction = vision_model.predict(image, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        return predicted_class, confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Auth endpoints
@app.post("/register")
@app.post("/api/register")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...)
):
    """Register a new user"""
    print(f"Registration attempt: {username}, {email}")
    # Check if user already exists
    if username in users_db:
        print(f"Registration failed - user {username} already exists")
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash the password
    hashed_password = pwd_context.hash(password)
    
    # Store the hashed password
    users_db[username] = {
        "username": username,
        "password": hashed_password,
        "email": email
    }
    
    print(f"Registration successful for {username}")
    # Return success response
    return {"message": "Registration successful"}

@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    print(f"Login attempt for user: {username}")
    
    # Debug: Check if user exists
    if username not in users_db:
        print(f"User {username} not found in database")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Debug: Print stored user data (except password)
    user_data = users_db.get(username).copy()
    if 'password' in user_data:
        user_data['password'] = '***REDACTED***'
    print(f"User data found: {user_data}")
    
    user = authenticate_user(username, password)
    if not user:
        print("Authentication failed: password doesn't match")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = username  # Store username instead of user object
    
    print(f"Login successful, session created: {session_id}")
    return {
        "session_id": session_id,
        "username": username,
        "message": "Login successful"
    }

@app.post("/api/logout")
async def logout(request: Request):
    session_id = request.headers.get('authorization')
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
    return {"message": "Logout successful"}

# Chat endpoints
@app.post("/chat-sessions")
async def create_chat_session(
    title: str = Form(...),
    session_id: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    chat_session_id = str(uuid.uuid4())
    chat_sessions[chat_session_id] = ChatSession(
        id=chat_session_id,
        title=title,
        messages=[],
        created_at=datetime.now().isoformat(),
        user_id=current_user.username
    )
    return {"session_id": chat_session_id, "title": title}

@app.get("/chat-sessions")
async def get_chat_sessions(current_user: User = Depends(get_current_user)):
    user_sessions = [
        {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "message_count": len(session.messages)
        }
        for session in chat_sessions.values()
        if session.user_id == current_user.username
    ]
    return {"sessions": sorted(user_sessions, key=lambda x: x["created_at"], reverse=True)}

@app.post("/query")
async def query(
    current_user: User = Depends(get_current_user),
    image: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    try:
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_sessions[session_id] = ChatSession(
                id=session_id,
                title="New Chat",
                messages=[],
                created_at=datetime.now().isoformat(),
                user_id=current_user.username
            )
        elif session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        session = chat_sessions[session_id]
        
        if session.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

        if image:
            # Handle image analysis
            os.makedirs('temp', exist_ok=True)
            image_path = f"temp/{image.filename}"
            try:
                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                
                predicted_class, confidence = predict_image(image_path)
                
                explanation_query = f"The model predicts the plant has {predicted_class} with {confidence:.2%} confidence. Can you explain what this means?"
                explanation_response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides information about potato plant diseases and treatments."},
                        {"role": "user", "content": explanation_query}
                    ]
                )
                explanation = explanation_response.choices[0].message.content.strip()

                treatment_query = f"{predicted_class} treatment potato plant"
                relevant_docs = search_rag(treatment_query)
                augmented_treatment_query = f"Based on the following documents: {relevant_docs}, provide detailed treatment plans for {predicted_class} in potato plants."
                
                treatment_response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides information about potato plant diseases and treatments."},
                        {"role": "user", "content": augmented_treatment_query}
                    ]
                )
                treatment_plans = treatment_response.choices[0].message.content.strip()

                session.messages.extend([
                    ChatMessage(
                        role="user",
                        content=f"Uploaded image: {image.filename}",
                        timestamp=datetime.now().isoformat()
                    ),
                    ChatMessage(
                        role="assistant",
                        content="Here's my analysis:",
                        timestamp=datetime.now().isoformat(),
                        result={
                            "predicted_class": predicted_class,
                            "confidence": f"{confidence:.2%}",
                            "explanation": explanation,
                            "treatment_plans": treatment_plans
                        }
                    )
                ])

                return {
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence:.2%}",
                    "explanation": explanation,
                    "treatment_plans": treatment_plans,
                    "session_id": session_id,
                    "messages": [msg.dict() for msg in session.messages]
                }
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

        elif question:
            session.messages.append(ChatMessage(
                role="user",
                content=question,
                timestamp=datetime.now().isoformat()
            ))

            relevant_docs = search_rag(question)
            augmented_query = f"Based on the following documents: {relevant_docs}, answer the question: {question}"
            
            answer_response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides information about potato plant diseases and treatments."},
                    {"role": "user", "content": augmented_query}
                ]
            )
            answer = answer_response.choices[0].message.content.strip()

            session.messages.append(ChatMessage(
                role="assistant",
                content=answer,
                timestamp=datetime.now().isoformat()
            ))

            return {
                "answer": answer,
                "session_id": session_id,
                "messages": [msg.dict() for msg in session.messages]
            }

        raise HTTPException(status_code=400, detail="No valid input provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history")
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        session = chat_sessions[session_id]
        
        if session.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat history")
        
        return {
            "messages": [msg.dict() for msg in session.messages],
            "session_info": {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Replace the existing message endpoints with these corrected versions

@app.post("/{session_id}/messages")
async def send_message(
    session_id: str,
    message: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        session = chat_sessions[session_id]
        
        if session.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

        # Add user message
        session.messages.append(ChatMessage(
            role="user",
            content=message,
            timestamp=datetime.now().isoformat()
        ))

        # Get response using RAG
        relevant_docs = search_rag(message)
        augmented_query = f"Based on the following documents: {relevant_docs}, answer the question: {message}"
        
        answer_response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides information about potato plant diseases and treatments."},
                {"role": "user", "content": augmented_query}
            ]
        )
        answer = answer_response.choices[0].message.content.strip()

        # Add assistant response
        session.messages.append(ChatMessage(
            role="assistant",
            content=answer,
            timestamp=datetime.now().isoformat()
        ))

        return {
            "session_id": session_id,
            "messages": [msg.dict() for msg in session.messages]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{session_id}/messages")
async def get_messages(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        session = chat_sessions[session_id]
        
        if session.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

        return {
            "session_id": session_id,
            "messages": [msg.dict() for msg in session.messages]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path_name: str):
    try:
        body = None
        try:
            body = await request.body()
        except:
            body = "Could not read body"
            
        headers = dict(request.headers)
        client_host = request.client.host if request.client else "unknown"
        method = request.method
        url = request.url.path
        query_params = dict(request.query_params)
        
        print(f"\n==== DEBUG 404 REQUEST ====")
        print(f"Method: {method}")
        print(f"URL: {url}")
        print(f"Client: {client_host}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
        print(f"Query Params: {json.dumps(query_params, indent=2)}")
        print(f"Body: {body}")
        print(f"==========================\n")
        
        return JSONResponse(
            status_code=404,
            content={"detail": f"Endpoint not found: {method} {url}"}
        )
    except Exception as e:
        print(f"Error in catch-all handler: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error in catch-all handler"}
        )

@app.get("/api/debug/users")
async def debug_users():
    """DEBUG ONLY: View registered users (remove in production)"""
    sanitized_users = {}
    for username, user_data in users_db.items():
        sanitized_users[username] = {
            "username": user_data.get("username", ""),
            "email": user_data.get("email", ""),
            "has_password": bool(user_data.get("password"))
        }
    return {"user_count": len(users_db), "users": sanitized_users}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)