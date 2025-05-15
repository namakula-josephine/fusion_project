import openai
import os
from dotenv import load_dotenv

# ✅ Load OpenAI API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Create OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

def get_chatgpt_response(predicted_label, retrieved_docs):
    """Generate a response using OpenAI's new ChatCompletion API format."""
    context = "\n".join(retrieved_docs)
    prompt = f"Based on the following documents:\n{context}\n\nExplain how '{predicted_label}' relates to 'The Late and Early Bright or Healthy'."

    try:
        # ✅ Use OpenAI's new API format
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in The Late and Early Bright topics."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content  # ✅ Extract response correctly

    except openai.APIConnectionError as e:
        print(f"🔴 OpenAI API Connection Error: {e}")
        return "Error connecting to OpenAI."

    except openai.OpenAIError as e:
        print(f"🔴 OpenAI API Error: {e}")
        return "Error generating response."
