import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import logging
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import parsedatetime as pdt

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

tagalog_weekdays = {
    "lunes": "monday",
    "martes": "tuesday",
    "miyerkules": "wednesday",
    "huwebes": "thursday",
    "biyernes": "friday",
    "sabado": "saturday",
    "linggo": "sunday"
}

cal = pdt.Calendar()

def normalize_date(value: str) -> str:
    today = datetime.today()

    if not value:
        return ""

    value_lower = value.strip().lower()

    for tagalog, english in tagalog_weekdays.items():
        if tagalog in value_lower:
            value_lower = value_lower.replace(tagalog, english)

    try:
        if "today" in value_lower:
            return today.strftime('%Y-%m-%d')
        elif "tomorrow" in value_lower:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            time_struct, parse_status = cal.parse(value_lower)
            if parse_status:
                parsed_date = datetime(*time_struct[:6])
                return parsed_date.strftime('%Y-%m-%d')
            return ""
    except Exception as e:
        logging.warning(f"Failed to parse date: {value} -> {e}")
        return ""

@app.post("/taskChat")
async def chat(request: ChatRequest):
    try:
        system_prompt = """
        You are a multilingual assistant that helps users create task entries in a task management system.

        The user might speak in English, Tagalog, Bisaya, Ilocano, Bicolano, Korean, Japanese, Chinese, German, Russian or a mix of languages.

        Your task:
        - Detect the language of the user input.
        - Respond in the **same language** used by the user — do NOT translate to a different language.
        - Extract the following fields:
          - title: short title of the task
          - details: full task description
          - due_date: the due date of the task in natural language (e.g., April 10, today, tomorrow)
          - effective_date: the start date of the task in natural language (e.g., April 7, today, tomorrow)
          - assigned_to: the full name of the person the task is assigned to, if mentioned. If not mentioned, use an empty string ""

        Respond ONLY in this exact pure JSON format:
        {
          "title": "Cashier Task",
          "details": "Manage cash transactions, provide customer service, and maintain records.",
          "due_date": "April 10",
          "effective_date": "today",
          "assigned_to": "Juan Dela Cruz"
        }
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            max_tokens=300,
            temperature=0.5
        )

        ai_message = response.choices[0].message.content.strip()
        logging.info(f"AI Response: {ai_message}")

        task_data = json.loads(ai_message)
      
        task_data.setdefault("title", "")
        task_data.setdefault("details", "")
        task_data.setdefault("due_date", "")
        task_data.setdefault("effective_date", "")
        task_data.setdefault("assigned_to", "")

        today_str = datetime.today().strftime('%Y-%m-%d')

        raw_due = task_data["due_date"]
        raw_effective = task_data["effective_date"]

        normalized_due = normalize_date(raw_due)
        normalized_effective = normalize_date(raw_effective)

        if not raw_due and not raw_effective:
            task_data["due_date"] = today_str
            task_data["effective_date"] = today_str
        elif normalized_due and not normalized_effective:
            task_data["due_date"] = normalized_due
            task_data["effective_date"] = normalized_due
        elif normalized_effective and not normalized_due:
            task_data["due_date"] = normalized_effective
            task_data["effective_date"] = normalized_effective
        else:
            task_data["due_date"] = normalized_due or today_str
            task_data["effective_date"] = normalized_effective or today_str

        return {"response": task_data}

    except Exception as e:
        logging.error(f"Error during OpenAI API call or processing: {e}")
        return {"error": str(e)}

