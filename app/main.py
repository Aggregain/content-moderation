from fastapi import FastAPI, Body, HTTPException, Header
from app.schemas import InputData
from app.pii import moderate_text 

app = FastAPI()


@app.post("/api/shai/moderation")
async def dify_receive(
        data: InputData = Body(...),
        authorization: str = Header(None)
):
    expected_api_key = "123456" 
    if not authorization:
        raise HTTPException(status_code=401, detail="Unauthorized")
    auth_scheme, _, api_key = authorization.partition(' ')
    if auth_scheme.lower() != "bearer" or api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if data.point == "ping":
        return {"result": "pong"}
    elif data.point in (
            "APP_MODERATION_INPUT", "APP_MODERATION_OUTPUT",
            "app.moderation.input", "app.moderation.output"
    ):
        return handle_moderation(data.params)
    else:
        return {
            "flagged": True,
            "action": "direct_output",
            "preset_response": f"Extension point '{data.point}' not implemented."
        }


def handle_moderation(params: dict):
    text = ""
    lang = "ru"
    if "query" in params and params.get("query"):
        text = params["query"]
    elif "inputs" in params and params["inputs"].get("text"):
        text = params["inputs"]["text"]
        lang = params["inputs"].get("lang", "ru")
    elif "text" in params and params.get("text"):
        text = params.get("text", "")
    
  
    has_pii, is_toxic = moderate_text(text, lang) 
    
    if not has_pii and not is_toxic:
        return {
            "flagged": False,
            "action": "direct_output",
            "preset_response": ""
        }
    else:
        if has_pii and is_toxic:
            msg = "В вашем сообщении обнаружены персональные данные и высокая токсичность. Пожалуйста, отправьте корректный текст."
        elif has_pii:
            msg = "В вашем сообщении обнаружены персональные данные. Пожалуйста, удалите их и попробуйте снова."
        else:
            msg = "В вашем сообщении обнаружена высокая токсичность. Пожалуйста, переформулируйте ваш текст."
        return {
            "flagged": True,
            "action": "direct_output",
            "preset_response": msg
        }