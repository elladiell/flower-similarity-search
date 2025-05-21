from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
import time
import json
from typing import Dict, List, Optional
from model import FlowerSimilarityModel

#Определение путей к файлам моделей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Проверка наличия файла признаков в различных местах
potential_model_paths = [
    os.path.join(BASE_DIR, "models"),                #./models
    os.path.join(BASE_DIR, "..", "models"),          #../models
    os.path.join(os.path.dirname(BASE_DIR), "models")  #Соседняя директория models
]

MODEL_PATH = None
for path in potential_model_paths:
    features_path = os.path.join(path, "library_features.pkl")
    if os.path.exists(features_path):
        MODEL_PATH = path
        print(f"Найден файл с признаками: {features_path}")
        break

if MODEL_PATH is None:
    print("Файл с признаками не найден. Используем папку по умолчанию.")
    MODEL_PATH = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_PATH, exist_ok=True)

#Создание директории для статических файлов, если она не существует
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(
    title="Flower Similarity Search API",
    description="API для поиска похожих изображений растений",
    version="1.0.0"
)

#Добавляем CORS middleware для разрешения запросов с разных источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Монтируем статические файлы
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#Инициализация модели
model = FlowerSimilarityModel(model_path=MODEL_PATH)

@app.get("/")
def read_root():
    """
    Корневой путь API. Возвращает информацию о статусе сервиса.
    """
    feature_dim = 0
    if model.image_features is not None and len(model.image_features) > 0:
        feature_dim = model.image_features.shape[1]
    
    return {
        "message": "Flower Similarity Search API",
        "status": "active",
        "model_info": {
            "feature_dim": feature_dim,
            "library_size": len(model.image_paths) if model.image_paths else 0
        },
        "endpoints": {
            "predict": "/predict/",
            "info": "/info/",
            "healthcheck": "/healthcheck",
            "create_library": "/create_library/",
            "endpoints": "/endpoints/"
        }
    }

@app.get("/healthcheck")
def healthcheck():
    """
    Проверка работоспособности API.
    """
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/endpoints/")
def list_endpoints():
    """
    Список всех доступных эндпоинтов API
    """
    return {
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Получение информации о статусе сервиса"
            },
            {
                "path": "/healthcheck",
                "method": "GET",
                "description": "Проверка работоспособности API"
            },
            {
                "path": "/predict/",
                "method": "POST",
                "description": "Загрузка изображения и поиск похожих изображений растений",
                "parameters": {
                    "file": "Загружаемое изображение (multipart/form-data)",
                    "top_n": "Количество возвращаемых похожих изображений (опционально)"
                }
            },
            {
                "path": "/info/",
                "method": "GET",
                "description": "Получение информации о модели и библиотеке изображений"
            },
            {
                "path": "/create_library/",
                "method": "POST",
                "description": "Создание библиотеки признаков из директории с изображениями",
                "parameters": {
                    "image_dir": "Путь к директории с изображениями",
                    "save_path": "Путь для сохранения файла признаков (опционально)"
                }
            }
        ]
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...), top_n: int = Query(5, description="Количество возвращаемых похожих изображений")):
    """
    Загрузка изображения и поиск похожих изображений растений
    file: Загружаемое изображение
    top_n: Количество возвращаемых похожих изображений
    Возвращаем:
    Dict: Словарь с путями к похожим изображениям и их показателями сходства
    """
    #Проверка типа файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        #Чтение и преобразование изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        #Сохраняем загруженное изображение для отладки
        upload_dir = os.path.join(STATIC_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        upload_filename = f"uploaded_{int(time.time())}_{file.filename}"
        upload_path = os.path.join(upload_dir, upload_filename)
        image.save(upload_path)
        
        #Поиск похожих изображений
        start_time = time.time()
        similar_images = model.find_similar_images(image, top_n=top_n)
        process_time = time.time() - start_time
        
        #Проверяем, есть ли ошибки в результате
        if isinstance(similar_images, dict) and "error" in similar_images:
            return JSONResponse(
                content={
                    "error": similar_images["error"],
                    "process_time": process_time
                },
                status_code=404
            )
        
        #Подготовка ответа
        response = {
            "result": similar_images,
            "uploaded_image": {
                "filename": file.filename,
                "saved_path": f"/static/uploads/{upload_filename}",
                "content_type": file.content_type
            },
            "process_time": process_time
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        import traceback
        print(f"Ошибка при обработке изображения: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")

@app.get("/info/")
def get_info():
    """
    Получение информации о модели и библиотеке изображений
    """
    feature_dim = 0
    if model.image_features is not None and len(model.image_features) > 0:
        feature_dim = model.image_features.shape[1]
    
    #Получение распределения классов
    class_distribution = {}
    if model.image_labels and model.classes:
        for i, cls in enumerate(model.classes):
            class_distribution[cls] = model.image_labels.count(i)
    
    return {
        "model_info": {
            "feature_dim": feature_dim,
            "device": str(model.device),
            "model_path": MODEL_PATH,
            "model_type": "EfficientNet B0"
        },
        "library_info": {
            "images_count": len(model.image_paths) if model.image_paths else 0,
            "classes": model.classes,
            "class_distribution": class_distribution,
            "features_path": model.features_path,
            "has_metadata": len(model.metadata) > 0 if model.metadata else False
        }
    }

@app.post("/create_library/")
def create_library(image_dir: str, save_path: Optional[str] = None):
    """
    Создание библиотеки признаков из директории с изображениями
    
    Args:
        image_dir: Путь к директории с изображениями
        save_path: Путь для сохранения файла признаков (опционально)
        
    Returns:
        Dict: Информация о созданной библиотеке
    """
    try:
        if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
            return JSONResponse(
                content={"error": f"Директория {image_dir} не существует"},
                status_code=400
            )
        
        #Если путь сохранения не указан, используем стандартный
        if save_path is None:
            save_path = model.features_path
        
        start_time = time.time()
        
        #Создаем библиотеку векторов признаков
        model.create_features_library(image_dir, save_path)
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "library_info": {
                "images_count": len(model.image_paths),
                "feature_dim": model.image_features.shape[1] if model.image_features is not None and len(model.image_features) > 0 else 0,
                "classes": model.classes,
                "saved_path": save_path
            },
            "process_time": process_time
        }
    except Exception as e:
        import traceback
        print(f"Ошибка при создании библиотеки признаков: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            content={"error": f"Ошибка при создании библиотеки: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)