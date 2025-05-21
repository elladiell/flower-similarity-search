from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision
from PIL import Image
import io
import numpy as np
from typing import Dict, List
import os
from app.model import FlowerSimilarityModel

app = FastAPI(title="Flower Similarity Search API",
              description="API для поиска похожих изображений растений",
              version="1.0.0")

#Инициализация модели
model = FlowerSimilarityModel()

@app.get("/")
def read_root():
    return {"message": "Flower Similarity Search API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Загрузка изображения и поиск похожих изображений растений
    """
    try:
        #Чтение и преобразование изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        #Поиск похожих изображений
        similar_images = model.find_similar_images(image)
        
        return JSONResponse(content=similar_images)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)