import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import logging
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import parsedatetime as pdt
import pytz

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

roles = ["cashier", "manager", "driver", "engineer", "waiter", "supervisor", "accountant"]

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
PH_TIMEZONE = pytz.timezone("Asia/Manila")

user_memory = {}  # Store last task per user_id

def normalize_date(value: str) -> str:
    today = datetime.now(PH_TIMEZONE)

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
                parsed_date = PH_TIMEZONE.localize(parsed_date)
                return parsed_date.strftime('%Y-%m-%d')
            return ""
    except Exception as e:
        logging.warning(f"Failed to parse date: {value} -> {e}")
        return ""

class ChatRequest(BaseModel):
    message: str
    user_id: str  # Added for per-user memory

@app.post("/taskChat")
async def chat(request: ChatRequest):
    try:
        user_id = request.user_id
        previous_task = user_memory.get(user_id)

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
          - assigned_to: the full name of the person the task is assigned to, if mentioned. If no name is mentioned, assign an empty string "" to this field.
          - If no specific person is mentioned, check if the role is mentioned (e.g., cashier, manager, etc.). If it is, leave `assigned_to` empty.
          - If the word "assign" is used or a person's name is included in the input, assign the task to that person.

        Respond ONLY in this exact pure JSON format:
        {
          "title": "",
          "details": "",
          "due_date": "",
          "effective_date": "",
          "assigned_to": ""
        }
        """

        messages = [{"role": "system", "content": system_prompt}]

        if previous_task:
            messages.append({
                "role": "user",
                "content": f"Previously, the task was: {json.dumps(previous_task)}"
            })

        messages.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
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

        today_str = datetime.now(PH_TIMEZONE).strftime('%Y-%m-%d')

        raw_due = task_data["due_date"]
        raw_effective = task_data["effective_date"]

        normalized_due = normalize_date(raw_due)
        normalized_effective = normalize_date(raw_effective)

        if "assign" in request.message.lower():
            if task_data["assigned_to"] == "":
                task_data["assigned_to"] = "Unknown Assignee"

        elif any(role in request.message.lower() for role in roles):
            task_data["assigned_to"] = ""

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

        # Update the user's memory
        user_memory[user_id] = task_data

        return {"response": task_data}

    except Exception as e:
        logging.error(f"Error during OpenAI API call or processing: {e}")
        return {"error": str(e)}
