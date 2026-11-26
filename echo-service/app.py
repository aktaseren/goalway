import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Echo Model")

class ChatRequest(BaseModel):
    model: str
    messages: list

@app.get("/health")
def health():
    return {"status": "healthy", "model": "echo"}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt = req.messages[-1]["content"]
    
    start = time.time()

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "model": req.model,
        "backend": "echo-service",
        "latency_ms": latency_ms,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"Echo: {prompt}"
                }
            }
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)