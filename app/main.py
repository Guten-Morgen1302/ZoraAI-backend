import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine
from app.models import Base
from app.auth.router import router as auth_router
from app.text_analysis.router import router as text_analysis_router
from app.middleware.auth_logging import AuthLoggingMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mail.google.com"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app.add_middleware(AuthLoggingMiddleware)

Base.metadata.create_all(bind=engine)

app.include_router(auth_router)
app.include_router(text_analysis_router)

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.get("/hello")
def hello():
    return {"message": "Hello Pratham"}