from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv
import os
import shutil
from retriever import search_rag
import openai
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from auth_utils import load_users_db, save_users_db, hash_password, verify_password

load_dotenv('secrets.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],  # Your React app's URL
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

users_db = load_users_db()

try:
    vision_model = load_model('model/potato_classification_model.h5')
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
except Exception as e:
    print(f"Error loading model: {e}")
    vision_model = None

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

@app.post("/api/query")
async def query(
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: dict = Depends(authenticate_user)
):
    try:
        if not question and not image:
            raise HTTPException(
                status_code=400,
                detail="No question or image provided"
            )

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
                
                return {
                    "answer": explanation,
                    "result": {
                        "predicted_class": predicted_class,
                        "confidence": f"{confidence:.2%}"
                    }
                }
            
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
            return {"answer": answer}
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

def preprocess_image(image_path, img_size=(224, 224)):
    
    with Image.open(image_path) as img:
        image = img.convert('RGB').resize(img_size)
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)

async def get_ai_response(query: str) -> str:
    if not openai.api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured"
        )
        
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant specializing in potato plant diseases."
                },
                {"role": "user", "content": query}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)