from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Union
import uvicorn
import logging
import time
import json

from run_ms_api import get_client

# ======= 配置 =======
MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    client, resolved_model = get_client(model_name=MODEL)
    logger.info(f"Client initialized: model={MODEL}, resolved={resolved_model}")
except Exception as e:
    logger.error(f"Failed to initialize client: {e}")
    raise

app = FastAPI(title="Azure OpenAI Proxy", version="1.0.0")

# ======= 数据模型 =======
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(None, gt=0, le=128000)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10)
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # 函数调用相关
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    # JSON 模式
    response_format: Optional[Dict[str, str]] = None

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

# ======= 错误处理 =======
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """返回 OpenAI 格式的错误"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": exc.status_code
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理的异常"""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_error",
                "code": 500
            }
        }
    )

# ======= 流式响应生成器 =======
async def generate_stream(response):
    """生成 SSE 格式的流式响应"""
    try:
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                chunk_data = {
                    "id": chunk.id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": resolved_model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": delta.role if hasattr(delta, 'role') and delta.role else None,
                            "content": delta.content if hasattr(delta, 'content') else None,
                        },
                        "finish_reason": chunk.choices[0].finish_reason
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_data = {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"

# ======= API 端点 =======
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    """OpenAI 兼容的 chat completions 端点"""
    try:
        # 准备请求参数
        params = {
            "model": resolved_model,
            "messages": [m.model_dump(exclude_none=True) for m in req.messages],
        }
        
        # 添加可选参数
        if req.max_tokens is not None:
            params["max_completion_tokens"] = req.max_tokens
        
        # GPT-5 不支持某些参数
        if MODEL != "gpt-5":
            if req.temperature is not None:
                params["temperature"] = req.temperature
            if req.top_p is not None:
                params["top_p"] = req.top_p
            if req.presence_penalty is not None:
                params["presence_penalty"] = req.presence_penalty
            if req.frequency_penalty is not None:
                params["frequency_penalty"] = req.frequency_penalty
        
        if req.n is not None:
            params["n"] = req.n
        if req.stop is not None:
            params["stop"] = req.stop
        if req.logit_bias is not None:
            params["logit_bias"] = req.logit_bias
        if req.user is not None:
            params["user"] = req.user
        if req.tools is not None:
            params["tools"] = req.tools
        if req.tool_choice is not None:
            params["tool_choice"] = req.tool_choice
        if req.response_format is not None:
            params["response_format"] = req.response_format
        
        # 流式响应
        if req.stream:
            params["stream"] = True
            response = client.chat.completions.create(**params)
            return StreamingResponse(
                generate_stream(response),
                media_type="text/event-stream"
            )
        
        # 非流式响应
        response = client.chat.completions.create(**params)
        
        # 构建完整响应
        return {
            "id": response.id or f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resolved_model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        **({"tool_calls": choice.message.tool_calls} 
                           if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls 
                           else {})
                    },
                    "logprobs": None,
                    "finish_reason": choice.finish_reason
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            },
            "system_fingerprint": getattr(response, 'system_fingerprint', None)
        }
    
    except AttributeError as e:
        logger.error(f"Invalid response structure: {e}")
        raise HTTPException(status_code=502, detail="Invalid API response format")
    except Exception as e:
        logger.error(f"API call failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": resolved_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "azure"
            }
        ]
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "model": resolved_model}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        log_level="info"
    )
