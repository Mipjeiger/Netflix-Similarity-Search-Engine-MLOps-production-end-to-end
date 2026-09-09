from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

# Define FastAPI app
app = FastAPI(title='Netflix Content Recommendation API', version='1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

# Endpoint checked up
@app.get("/")
async def root():
    return {"message": "Netflix Content Recommendation API is running.",
            "status": "✅ healthy"}