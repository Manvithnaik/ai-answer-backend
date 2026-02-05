from app.core.llm_interface import BaseLLM

class MockLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        p = prompt.lower()

        if "what is ai" in p:
            return "AI stands for Artificial Intelligence. It enables machines to mimic human intelligence."

        if "fastapi" in p:
            return "FastAPI is a modern Python framework for building APIs quickly."

        if "react" in p:
            return "React is a JavaScript library for building user interfaces."

        return "This is a mock AI response. Replace MockLLM with a real LLM later."
