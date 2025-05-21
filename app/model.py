import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from typing import Dict, List, Tuple
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FlowerSimilarityModel:
    def __init__(self, model_path: str = "models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Загрузка модели и преобразований
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
        #Пути к файлам модели
        self.model_path = model_path
        self.features_path = os.path.join(model_path, "library_features.pkl")
        
        #Загрузка библиотеки векторов признаков
        self.image_paths = []
        self.image_features = None
        self.image_labels = []
        self.classes = []
        self.load_features()
    
    def _load_model(self):
        """Загрузка предобученной модели efficientnet b0"""
        model = models.efficientnet_b0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=weights)
        #Удалим последний полносвязанный слой, не нужна классификация по исходным классам
        self.model.classifier = torch.nn.Identity()
        model.eval()
        return model.to(self.device)
    
    def _get_transforms(self):
        """Получение преобразований для изображений"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def load_features(self):
        """Загрузка предварительно извлеченных признаков"""
        try:
            with open(self.features_path, 'rb') as f:
                library_data = pickle.load(f)
                
            self.image_paths = library_data['paths']
            self.image_features = library_data['features']
            self.image_labels = library_data['labels']
            self.classes = library_data['classes']
            
            print(f"Загружены признаки для {len(self.image_paths)} изображений библиотеки")
        except Exception as e:
            print(f"Ошибка при загрузке признаков: {e}")
            #Создаем пустые структуры данных
            self.image_paths = []
            self.image_features = np.array([])
            self.image_labels = []
            self.classes = []
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Извлечение вектора признаков из изображения
        image: PIL изображение
        Возвращает:
        np.ndarray: Вектор признаков
        """
        #Применяем трансформации
        image_tensor = self.transform(image).unsqueeze(0)  #Добавляем размерность batch
        
        #Извлекаем признаки
        with torch.no_grad():
            features = self.model(image_tensor.to(self.device))
            features = features.squeeze().cpu().numpy()
            
        return features
    
    def find_similar_images(self, image: Image.Image, top_n: int = 5) -> Dict[str, float]:
        """
        Поиск похожих изображений на основе входного изображения
        image: Входное изображение
        top_n: Количество похожих изображений для возврата
        Возвращает:
        Dict[str, float]: словарь с путями к изображениям и их показателями сходства
        """
        #Проверка загрузки признаков
        if len(self.image_paths) == 0 or self.image_features is None:
            return {"error": "Библиотека изображений не загружена"}
        
        #Извлекаем признаки из входного изображения
        query_feature = self.extract_features(image)
        
        #Вычисляем косинусное сходство между запросом и всеми изображениями
        similarities = cosine_similarity(query_feature.reshape(1, -1), self.image_features)[0]
        
        #Находим индексы top_n наиболее похожих изображений
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        #Создаем словарь результатов
        result = {}
        for idx in top_indices:
            image_path = self.image_paths[idx]
            result[image_path] = float(similarities[idx])
        
        return result