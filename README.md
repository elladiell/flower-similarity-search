# Flower Similarity Search

Система поиска наиболее похожих растений на основе входного изображения среди изображений из заданной библиотеки.

## Описание проекта

Проект представляет собой систему для поиска похожих изображений растений с использованием глубокого обучения. Система анализирует входное изображение и находит 5 наиболее похожих изображений из библиотеки на основе косинусного сходства векторов признаков.

## Структура проекта

flower-similarity-search/
├── app/                  # Код для API
│   ├── __init__.py
│   ├── main.py           # FastAPI приложение
│   └── model.py          # Код модели
├── data/                 # Папка для данных
│   └── .gitkeep
├── models/               # Сохраненные веса модели
│   └── .gitkeep
├── notebooks/            # Jupyter notebooks
│   └── flower_similarity_model.ipynb
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

## Установка и запуск

### Локальная установка

1. Клонировать репозиторий:
git clone https://github.com/[ваш_логин]/flower-similarity-search.git
cd flower-similarity-search

2. Создать виртуальное окружение и установить зависимости:
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Запустить API:
cd app
uvicorn main:app --reload

### Docker

Собрать и запустить контейнер:
docker-compose up --build

### Docker
Пример запроса к API:

import requests

#Загрузка изображения
url = "http://localhost:8000/predict/"
files = {"file": open("path/to/flower_image.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())

### Jupiter Notebook

Ноутбук notebooks/flower_similarity_model.ipynb содержит код для подготовки данных, обучения и тестирования модели.

### Данные

Проект использует датасет Flowers Recognition для обучения и тестирования модели.