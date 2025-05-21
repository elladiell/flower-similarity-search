import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from typing import Dict, List, Tuple, Union, Any
import pickle
from sklearn.metrics.pairwise import cosine_similarity
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
        self.metadata = {}  #Добавляем словарь для метаданных
        self.load_features()
    
    def _load_model(self):
        """Загрузка предобученной модели efficientnet b0"""
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        #Удалим последний полносвязанный слой, не нужна классификация по исходным классам
        model.classifier = torch.nn.Identity()
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
            
            #Загружаем метаданные, если они есть
            if 'metadata' in library_data:
                self.metadata = library_data['metadata']
            
            print(f"Загружены признаки для {len(self.image_paths)} изображений библиотеки")
            
            #Проверка размерности признаков
            if len(self.image_features) > 0:
                feature_dim = self.image_features.shape[1]
                print(f"Размерность векторов признаков: {feature_dim}")
                
        except Exception as e:
            print(f"Ошибка при загрузке признаков: {e}")
            #Создаем пустые структуры данных
            self.image_paths = []
            self.image_features = np.array([])
            self.image_labels = []
            self.classes = []
            self.metadata = {}
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Извлечение вектора признаков из изображения
        image: PIL изображение
        Возвращает:
        np.ndarray: Вектор признаков
        """
        # Применяем трансформации
        image_tensor = self.transform(image).unsqueeze(0)  # Добавляем размерность batch
        
        # Извлекаем признаки
        with torch.no_grad():
            features = self.model(image_tensor.to(self.device))
            features = features.squeeze().cpu().numpy()
            
        return features
    
    def find_similar_images(self, image: Image.Image, top_n: int = 5) -> Dict[str, Any]:
        """
        Поиск похожих изображений на основе входного изображения
        image: Входное изображение
        top_n: Количество похожих изображений для возврата
        Возвращает:
        Dict[str, Any]: словарь с путями к изображениям, их показателями сходства и метаданными
        """
        #Проверка загрузки признаков
        if len(self.image_paths) == 0 or self.image_features is None or len(self.image_features) == 0:
            return {"error": "Библиотека изображений не загружена или пуста"}
        
        try:
            #Извлекаем признаки из входного изображения
            query_feature = self.extract_features(image)
            
            #Вычисляем косинусное сходство между запросом и всеми изображениями
            similarities = cosine_similarity(query_feature.reshape(1, -1), self.image_features)[0]
            
            #Находим индексы top_n наиболее похожих изображений
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            #Создаем словарь результатов с дополнительной информацией
            result = {
                "similar_images": [],
                "feature_dim": query_feature.shape[0]
            }
            
            for idx in top_indices:
                image_path = self.image_paths[idx]
                similarity_score = float(similarities[idx])
                
                #Определяем класс изображения
                class_id = self.image_labels[idx] if idx < len(self.image_labels) else None
                class_name = self.classes[class_id] if class_id is not None and class_id < len(self.classes) else "unknown"
                
                #Добавляем метаданные, если они есть
                image_metadata = self.metadata.get(image_path, {}) if self.metadata else {}
                
                image_info = {
                    "path": image_path,
                    "similarity": similarity_score,
                    "class_id": int(class_id) if class_id is not None else None,
                    "class_name": class_name,
                    "metadata": image_metadata
                }
                
                result["similar_images"].append(image_info)
            
            return result
            
        except Exception as e:
            import traceback
            print(f"Ошибка при поиске похожих изображений: {e}")
            print(traceback.format_exc())
            return {"error": f"Ошибка обработки: {str(e)}"}
    
    def create_features_library(self, image_dir: str, save_path: str = None):
        """
        Создание библиотеки векторов признаков из директории с изображениями
        
        Args:
            image_dir: Путь к директории с изображениями
            save_path: Путь для сохранения файла признаков
        """
        if save_path is None:
            save_path = self.features_path
        
        #Собираем пути к изображениям
        image_paths = []
        labels = []
        classes = []
        
        #Сканируем директорию, считая, что подпапки - это классы
        for class_id, class_name in enumerate(sorted(os.listdir(image_dir))):
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            classes.append(class_name)
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    image_paths.append(img_path)
                    labels.append(class_id)
        
        #Извлекаем признаки из всех изображений
        features_list = []
        valid_paths = []
        valid_labels = []
        metadata = {}
        
        print(f"Обработка {len(image_paths)} изображений...")
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            try:
                img = Image.open(img_path).convert("RGB")
                features = self.extract_features(img)
                
                features_list.append(features)
                valid_paths.append(img_path)
                valid_labels.append(label)
                
                #Добавляем базовые метаданные
                metadata[img_path] = {
                    "filename": os.path.basename(img_path),
                    "class": classes[label],
                    "size": img.size
                }
                
                if (i + 1) % 50 == 0:
                    print(f"Обработано {i + 1} изображений из {len(image_paths)}")
                    
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
        
        #Преобразуем список признаков в массив NumPy
        features_array = np.array(features_list)
        
        #Сохраняем извлеченные признаки и метаданные
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        library_data = {
            'paths': valid_paths,
            'features': features_array,
            'labels': valid_labels,
            'classes': classes,
            'metadata': metadata
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(library_data, f)
        
        print(f"Библиотека признаков создана и сохранена в {save_path}")
        print(f"Размерность признаков: {features_array.shape}")
        
        #Обновляем текущую модель
        self.image_paths = valid_paths
        self.image_features = features_array
        self.image_labels = valid_labels
        self.classes = classes
        self.metadata = metadata