import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tavily import TavilyClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MAX_STEPS = 8
MAX_HISTORY_MESSAGES = 40  # sliding window: keep the last N non-system messages
MEMORY_FILE = Path("memory.json")
AUDIT_LOG_FILE = Path("audit_log.jsonl")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ---------- Audit trail ----------

def log_audit(session_id: str, step: int, event: str, **details) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "step": step,
        "event": event,
        **details,
    }
    with AUDIT_LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------- Memory (persists across runs) ----------

def load_memory() -> list[str]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text()).get("facts", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_fact(fact: str) -> str:
    facts = load_memory()
    facts.append(fact)
    MEMORY_FILE.write_text(json.dumps({"facts": facts}, indent=2))
    return f"Remembered: {fact}"


def build_system_prompt() -> str:
    base = (
        "You help someone evaluate a job opportunity. Given a company name and a role "
        "description, research the company and role using web_search before answering. "
        "Cover: what the company does and recent news, its product/tech stack or engineering "
        "practices if relevant, team structure or work culture, and how the role's stated "
        "responsibilities map to daily work. Search enough to ground your answer in specifics "
        "you found, not generic assumptions. Once you have enough, describe a concrete, "
        "realistic day-to-day workflow for someone in this role at this company: likely tasks, "
        "tools, meetings, and collaborators. If the user asks a follow-up instead, just answer it "
        "directly using what's already in context, without re-researching unless it's genuinely new "
        "ground. Use the remember tool only for durable facts worth keeping across sessions "
        "(e.g. stated user preferences), not for search results or one-off answers."
    )
    facts = load_memory()
    if facts:
        base += "\n\nKnown facts from previous sessions:\n" + "\n".join(f"- {f}" for f in facts)
    return base


# ---------- Tools ----------

def web_search(query: str) -> str:
    result = tavily.search(query=query, max_results=3)
    return "\n\n".join(f"{r['title']}: {r['content']}" for r in result["results"])


def get_datetime() -> str:
    return datetime.now(timezone.utc).isoformat()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when the question needs facts you're not confident about, or anything time-sensitive.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current UTC date and time",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a durable fact that should persist across sessions (e.g. a stated user preference or a recurring detail about their setup). Do not use this for search results or answers that only matter for the current question.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
]

DISPATCH = {"web_search": web_search, "get_datetime": get_datetime, "remember": save_fact}


# ---------- Guardrails ----------

def validate_input(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if len(question) > 6000:
        raise ValueError("Input is too long (max 6000 characters).")
    return question


def filter_output(text: str | None) -> str:
    if not text or not text.strip():
        return "The agent didn't produce a usable answer."
    if len(text) > 8000:
        text = text[:8000] + "... [truncated]"
    return text


def trim_history(messages: list) -> list:
    """Keep the system message plus the most recent MAX_HISTORY_MESSAGES entries."""
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    return system_msgs[:1] + other_msgs[-MAX_HISTORY_MESSAGES:]


# ---------- Agent loop ----------

def run_agent(messages: list, question: str) -> tuple[str, list]:
    question = validate_input(question)
    session_id = uuid.uuid4().hex[:8]
    log_audit(session_id, 0, "prompt", question=question)
    messages.append({"role": "user", "content": question})

    for step in range(MAX_STEPS):
        if step > 0:
            messages.append({
                "role": "system",
                "content": f"You have {MAX_STEPS - step} step(s) left. If you can answer now, do so.",
            })

        response = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS,
            extra_headers={"X-Title": "research-agent"},
        )
        reply = response.choices[0].message
        print(f"[step {step}] tool_calls={bool(reply.tool_calls)} content={reply.content!r}")
        log_audit(
            session_id, step, "model_response",
            has_tool_calls=bool(reply.tool_calls),
            content=reply.content,
        )

        if reply.tool_calls:
            messages.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [tc.model_dump() for tc in reply.tool_calls],
            })
            for call in reply.tool_calls:
                fn = DISPATCH[call.function.name]
                args = json.loads(call.function.arguments)
                log_audit(session_id, step, "tool_call", tool=call.function.name, args=args)
                try:
                    result = fn(**args)
                    log_audit(session_id, step, "tool_result", tool=call.function.name, result=str(result))
                except Exception as e:
                    result = f"Tool error: {e}"
                    log_audit(session_id, step, "tool_error", tool=call.function.name, error=str(e))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })
            continue

        answer = filter_output(reply.content)
        messages.append({"role": "assistant", "content": answer})
        log_audit(session_id, step, "final_answer", answer=answer)
        return answer, trim_history(messages)

    log_audit(session_id, MAX_STEPS, "max_steps_reached")
    return "Max steps reached without a final answer", trim_history(messages)


# ---------- Entry point ----------

def read_input() -> str:
    """Read one line, or — if it's not a short command — keep reading pasted
    lines until a line containing only END, so a full job description
    (which typically has blank lines between paragraphs) can be pasted safely."""
    first_line = input("you: ")
    if first_line.strip().lower() in ("exit", "quit") or first_line.strip() == "":
        return first_line.strip()
    print("(paste as many lines as you need — including blank ones — then type END on its own line to submit)")
    lines = [first_line]
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    messages = [{"role": "system", "content": build_system_prompt()}]
    print("Day-in-the-Life Research Agent")
    print("Paste a company name + role description, or ask a follow-up. Type 'exit' to quit.\n")

    while True:
        user_input = read_input()
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            user_input = validate_input(user_input)
        except ValueError as e:
            print(f"Invalid input: {e}")
            continue

        answer, messages = run_agent(messages, user_input)
        print(f"\nagent:\n{answer}\n")


if __name__ == "__main__":
    main()
