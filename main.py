from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router, initialize_ml_service
from config.settings import CORS_ORIGINS

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    initialize_ml_service()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
