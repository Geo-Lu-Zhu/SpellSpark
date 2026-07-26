from fastapi import FastAPI
from src.api.endpoints import router as api_router

app = FastAPI(
    title="SpellSpark Backend",
    description="API for SpellSpark: A Spelling Growth Recommender System for Kids",
    version="1.0.0",
)

# Include the API router from endpoints.py
app.include_router(api_router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the SpellSpark API!"}