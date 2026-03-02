import getpass
import os

from dotenv import load_dotenv

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")

os.environ["LANGSMITH_TRACING"] = "true"

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

# Send test message "What were the top 5 WRs of the 2022 season?"
messages = [
    ("system", "You are a fantasy football expert in a fantasy football draft app."),
    ("human", "What were the top 5 WRs of the 2022 season?"),
]

ai_msg = llm.invoke(messages)
ai_msg

print(ai_msg.content)
