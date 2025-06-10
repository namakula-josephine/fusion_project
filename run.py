from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from backend.vision_model.vision_model import process_image
from backend.rag_system.retriever import generate_query, search_rag
from backend.rag_system.chatgpt_response import get_chatgpt_response

app = Flask(__name__)
CORS(app)  # Enable frontend-backend communication

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    image_path = "temp.jpg"
    image.save(image_path)

    # Step 1: Get label from vision model
    predicted_label = process_image(image_path)

    # Step 2: Retrieve related documents
    query = generate_query(predicted_label)
    retrieved_docs = search_rag(query)

    # Step 3: Get AI response
    final_response = get_chatgpt_response(predicted_label, retrieved_docs)

    return jsonify({
        "label": predicted_label,
        "documents": retrieved_docs,
        "response": final_response
    })

if __name__ == '__main__':
    print("WARNING: This Flask app conflicts with the main FastAPI app.")
    print("Use 'python app.py' to run the main application instead.")
    print("If you need to run this for testing, modify the port.")
    # app.run(debug=True, port=5002)  # Use different port if needed
