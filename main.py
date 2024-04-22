import os
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
from starlette.websockets import WebSocketDisconnect
import asyncio
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Указываем абсолютный путь к директории, содержащей файл index.html
templates_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates")
app.mount("/static", StaticFiles(directory=templates_dir), name="static")

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny",
                       generate_kwargs={"language": "russian"})


class TranscriptionRequest(BaseModel):
    audio_data: str


class TranscriptionResponse(BaseModel):
    transcription: str


connected_websockets = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()
            transcription = transcriber(data).get('text')  # Обработка бинарных данных

            response = TranscriptionResponse(transcription=transcription)
            await websocket.send_json(response.dict())
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        connected_websockets.remove(websocket)


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(templates_dir, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)