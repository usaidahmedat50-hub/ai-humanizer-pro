import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
import math

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def calculate_human_score(text):
    """
    Simulated human score based on perplexity (sentence variety) 
    and burstiness (word choice variety).
    """
    if not text or len(text.strip()) == 0:
        return 0
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    if not sentences:
        return 0
    
    # Perplexity proxy: Variance in sentence lengths
    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)
    variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    perplexity_score = min(math.sqrt(variance) * 5, 50) # Max 50 points
    
    # Burstiness proxy: Unique words vs total words
    words = re.findall(r'\w+', text.lower())
    unique_words = set(words)
    if not words:
        return 0
    burstiness_score = (len(unique_words) / len(words)) * 50 # Max 50 points
    
    total_score = int(perplexity_score + burstiness_score)
    return min(max(total_score, 10), 100) # Range 10-100%

async def humanize_text(text, tone="natural"):
    """
    Humanizes text using Gemini API with specific tone instructions.
    """
    prompts = {
        "natural": (
            "Rewrite the following text to sound natural and human. "
            "Increase perplexity and burstiness by varying sentence lengths and structures. "
            "Use subtle idioms, conversational flow, and remove common AI transition words "
            "like 'Furthermore', 'Moreover', or 'In conclusion'. Aim for a simple, clear fix."
        ),
        "academic": (
            "Rewrite the following text for an academic audience. "
            "Maintain formal tone but vary the complexity of sentence structures to avoid robotic patterns. "
            "Use precise vocabulary, ensure logical flow without over-relying on standard transition markers, "
            "and increase sentence variety for higher perplexity."
        ),
        "storyteller": (
            "Rewrite the following text with a storyteller's personality. "
            "Infuse vivid descriptions, rare idioms, and a dynamic rhythm. "
            "Vary the sentence structure heavily—short punchy sentences mixed with descriptive ones. "
            "Remove all AI-typical robotic phrasing and make it feel like a human wrote it with passion."
        )
    }
    
    instruction = prompts.get(tone, prompts["natural"])
    
    full_prompt = f"{instruction}\n\nInput Text: {text}\n\nHumanized Output:"
    
    response = model.generate_content(full_prompt)
    return response.text.strip()
