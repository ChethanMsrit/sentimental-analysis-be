from fastapi import FastAPI, HTTPException
from sentiment import analyze_sentiment_with_generate
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sentiment")
def sentiment_analysis(request: dict):
    try:
        print(request)

        result = analyze_sentiment_with_generate(request.get("reviewText"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
