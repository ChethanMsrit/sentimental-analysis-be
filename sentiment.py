import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_URL = "https://api.cohere.ai/v1/generate"


def analyze_sentiment_with_generate(review_text: str):
    prompt = f"""Analyze the sentiment of the following movie review. Classify it as "positive", "negative", or "neutral". 
    Also provide a confidence score from 1-100.
    
    Movie review: "{review_text}"
    
    Format your response as a valid JSON object with the following structure:
    {{
      "sentiment": "positive|negative|neutral",
      "confidence": 75,
      "reasoning": "brief explanation of your classification"
    }}

    Return ONLY the JSON object with no other text."""

    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "command",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.3,
        "k": 0,
        "stop_sequences": [],
        "return_likelihoods": "NONE"
    }

    response = requests.post(COHERE_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    output_text = response.json()["generations"][0]["text"].strip()

    try:
        result = json.loads(output_text)
        return result
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON returned by Cohere: " + output_text)
