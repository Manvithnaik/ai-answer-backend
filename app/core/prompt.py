def build_prompt(user_question: str) -> str:
    system_prompt = "You are a helpful AI assistant."

    return f"""
SYSTEM:
{system_prompt}

USER:
{user_question}

ASSISTANT:
""".strip()
