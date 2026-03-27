import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import humanizer
import uvicorn
from dotenv import load_dotenv
from mangum import Mangum

load_dotenv()

app = FastAPI(title="Stateless AI Humanizer")
handler = Mangum(app)

class HumanizeRequest(BaseModel):
    text: str
    tone: str

@app.post("/api/humanize")
async def api_humanize(req: HumanizeRequest):
    if len(req.text.split()) > 500:
        raise HTTPException(status_code=400, detail="Max 500 words allowed.")
    
    try:
        # 1. Humanize the text using Gemini
        humanized = await humanizer.humanize_text(req.text, req.tone)
        
        # 2. Calculate the "Human Score" of the output
        score = humanizer.calculate_human_score(humanized)
        
        return {
            "humanized": humanized,
            "score": score,
            "tone": req.tone
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing text.")

@app.post("/api/analyze")
async def api_analyze(req: HumanizeRequest):
    try:
        score = humanizer.calculate_human_score(req.text)
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error analyzing text.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

