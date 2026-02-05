from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ask import router as ask_router

app = FastAPI(title="AI Answer Hub Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for internship demo
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router)
