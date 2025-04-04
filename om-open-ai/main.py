import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Call OpenAI with token and temperature limits
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": request.message}
            ],
            max_tokens=100,
            temperature=0.7
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}")
        return {"error": str(e)}
