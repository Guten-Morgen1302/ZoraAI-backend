from fastapi import FastAPI
from app.database import engine
from app.models import Base
from app.auth.router import router as auth_router


app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(auth_router)

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.get("/hello")
def hello():
    return {"message": "Hello Pratham"}