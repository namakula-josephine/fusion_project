from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ✅ Load the Keras model correctly
model_path = "backend/vision_model/potato_classification_model.h5"
model = load_model(model_path)  # Load as a Keras model

# ✅ Preprocessing function for input images
def process_image(image_path):
    image = Image.open(image_path).resize((224, 224))  # Resize image
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # ✅ Get model prediction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions)  # Get class with highest probability

    return predicted_label

print(process_image("./backend/vision_model/image.png"))