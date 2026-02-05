from fastapi import APIRouter, Response
from app.schemas import AskRequest, AskResponse
from app.core.prompt import build_prompt
from app.llms.mock_llm import MockLLM

router = APIRouter()
llm = MockLLM()

# ✅ Explicit OPTIONS handler (THIS FIXES THE ISSUE)
@router.options("/ask")
def options_ask():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "http://localhost:5173",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )

@router.post("/ask", response_model=AskResponse)
def ask_ai(request: AskRequest):
    prompt = build_prompt(request.question)
    answer = llm.generate(prompt)
    return AskResponse(answer=answer)
