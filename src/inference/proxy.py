from fastapi import FastAPI
import uvicorn
import requests
from pydantic import BaseModel
from typing import List, Optional, Any

# ======= 你的 Azure GPT-5 客户端 =======
from run_ms_api import get_client

client, resolved_model = get_client(model_name="gpt-5")

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.0

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    out = client.chat.completions.create(
        model=resolved_model,
        messages=[m.dict() for m in req.messages],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    return {
        "id": "chatcmpl-azure-proxy",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": out.choices[0].message.content
                },
                "finish_reason": out.choices[0].finish_reason
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
