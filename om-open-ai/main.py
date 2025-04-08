import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import logging
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from dateutil import parser as date_parser

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

def normalize_date(value: str) -> str:
    today = datetime.today()

    if not value:
        return ""

    value_lower = value.strip().lower()

    try:
        if "today" in value_lower:
            return today.strftime('%Y-%m-%d')
        elif "tomorrow" in value_lower:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # Append current year if it's missing
            try:
                parsed = date_parser.parse(value, default=today.replace(month=1, day=1))
                return parsed.strftime('%Y-%m-%d')
            except Exception:
                return ""
    except Exception as e:
        logging.warning(f"Failed to parse date: {value} -> {e}")
        return ""

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        system_prompt = """
        You are an assistant that helps users create task entries in a task management system.
        Based on the user's message, extract the following:
        - title: short title of the task
        - details: full task description
        - due_date: the due date of the task in natural language (e.g., April 10, today, tomorrow)
        - effective_date: the start date of the task in natural language (e.g., April 7, today, tomorrow)

        If either date is not mentioned, return an empty string for that field.
        Always respond only in pure JSON format like this:
        {
          "title": "Cashier Task",
          "details": "Manage cash transactions, provide customer service, and maintain records.",
          "due_date": "April 10",
          "effective_date": "today"
        }
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            max_tokens=200,
            temperature=0.5
        )

        ai_message = response.choices[0].message.content.strip()
        logging.info(f"AI Response: {ai_message}")

        task_data = json.loads(ai_message)

        # Normalize date fields and default to today if empty
        today_str = datetime.today().strftime('%Y-%m-%d')

        raw_due = task_data.get("due_date", "")
        raw_effective = task_data.get("effective_date", "")

        normalized_due = normalize_date(raw_due)
        normalized_effective = normalize_date(raw_effective)

        task_data["due_date"] = normalized_due if normalized_due else today_str
        task_data["effective_date"] = normalized_effective if normalized_effective else today_str

        return {"response": task_data}

    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}")
        return {"error": str(e)}
