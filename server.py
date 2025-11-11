print("Focus each studio on one task. Serve your model here.")


# server.py (Simplified for deployment)
import torch
from unsloth import FastLanguageModel
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
app = FastAPI(title="Unsloth GPT-OSS 20B API")
generator = None

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.on_event("startup")
async def load_model():
    global generator
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_NAME,
            max_seq_length = 2048, 
            dtype = None,           
            load_in_4bit = True,    # CRITICAL for loading this specific model
            device_map = "auto",    
        )
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            #device=model.device, 
        )
    except Exception as e:
        # If loading fails, log the error and let the app fail gracefully
        print(f"FATAL ERROR LOADING MODEL: {e}")
        raise e 


@app.post("/generate")
def generate_text(request: InferenceRequest):
    if not generator:
        raise HTTPException(status_code=503, detail="Model service is not ready.")
    
    # 1. Prepare the input messages
    # The pipeline should handle the chat template automatically
    messages = [{"role": "user", "content": request.prompt}]
    
    outputs = generator(
        messages,
        max_new_tokens=request.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        # Set to True/False based on your preference, but False is common for chat
        return_full_text=False, 
    )
    
    # 2. Extract the generated text as a simple string
    # The output structure is typically: [{'generated_text': '...string content...'}]
    try:
        # Safely access the first (and usually only) generated text result
        response_text = outputs[0]['generated_text']
    except (KeyError, IndexError):
        # Fallback in case of unexpected output format
        raise HTTPException(status_code=500, detail="Failed to parse model output.")
    
    return {"generated_text": response_text}

# Add a simple health check, which the studio often expects
@app.get("/")
def read_root():
    return {"status": "Model service is operational and running Unsloth loader."}

# If the studio doesn't have a specific entrypoint, this is how you start:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080)

