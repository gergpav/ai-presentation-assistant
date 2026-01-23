# app/core/image_generator.py
import os
import logging
import asyncio
import torch
from pathlib import Path
from typing import Optional
from uuid import uuid4
from PIL import Image
from app.config import settings

logger = logging.getLogger(__name__)

STORAGE_DIR = Path("storage") / "images"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Ленивая загрузка модели
_pipeline = None
_model_lock = asyncio.Lock()


class ImageGenerator:
    """Генератор изображений через локальный Stable Diffusion на GPU"""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
        """
        Инициализация генератора изображений.
        
        Args:
            model_id: ID модели Stable Diffusion из HuggingFace
            device: Устройство для генерации ('cuda', 'cpu' или None для автоопределения)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._initialized = False
        
        logger.info(f"Инициализация ImageGenerator с моделью {model_id} на устройстве {self.device}")
        
        # Проверяем доступность GPU
        if self.device == "cuda":
            if torch.cuda.is_available():
                try:
                    device_name = torch.cuda.get_device_name(0)
                    device_capability = torch.cuda.get_device_capability(0)
                    logger.info(f"Обнаружен GPU: {device_name}")
                    logger.info(f"CUDA capability: {device_capability[0]}.{device_capability[1]}")
                    
                    # Проверяем совместимость (sm_120 и выше требуют специальных версий PyTorch)
                    # См. https://developer.nvidia.com/cuda/gpus - RTX 50xx серия имеет Compute Capability 12.0
                    if device_capability[0] >= 12:
                        # Если PyTorch видит GPU с sm_120, значит установлена версия с поддержкой
                        logger.info(f"✅ GPU {device_name} с CUDA capability sm_{device_capability[0]}.{device_capability[1]} (Blackwell) обнаружен")
                        logger.info(f"Используется GPU: {device_name}")
                        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                        # Не переключаемся на CPU - используем GPU
                    else:
                        logger.info(f"Используется GPU: {device_name}")
                        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                except Exception as e:
                    logger.warning(f"Не удалось получить информацию о GPU: {e}. Переключаемся на CPU")
                    self.device = "cpu"
            else:
                logger.warning("CUDA запрошена, но недоступна (PyTorch скомпилирован без CUDA). Переключаемся на CPU")
                logger.warning("Для использования GPU установите PyTorch с поддержкой CUDA:")
                logger.warning("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                self.device = "cpu"
        else:
            logger.info(f"Используется устройство: {self.device}")
    
    async def _load_model(self):
        """Ленивая загрузка модели Stable Diffusion"""
        global _pipeline
        
        async with _model_lock:
            if _pipeline is not None:
                self.pipeline = _pipeline
                self._initialized = True
                return
            
            try:
                logger.info(f"Загрузка модели Stable Diffusion: {self.model_id}")
                logger.info("Это может занять несколько минут при первом запуске...")
                
                # Импортируем diffusers только при необходимости
                from diffusers import StableDiffusionPipeline
                
                # Загружаем модель в отдельном потоке, чтобы не блокировать event loop
                def load():
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,  # Отключаем safety checker для ускорения
                        requires_safety_checker=False,
                    )
                    pipeline = pipeline.to(self.device)
                    
                    # Оптимизация для GPU
                    if self.device == "cuda":
                        # Включаем attention slicing для экономии памяти
                        pipeline.enable_attention_slicing()
                        # Оптимизация памяти
                        pipeline.enable_vae_slicing()
                    
                    return pipeline
                
                self.pipeline = await asyncio.to_thread(load)
                _pipeline = self.pipeline
                self._initialized = True
                
                logger.info("Модель успешно загружена и готова к использованию")
                
            except Exception as e:
                logger.error(f"Ошибка загрузки модели Stable Diffusion: {e}")
                raise
    
    def generate_image(self, prompt: str, output_path: Optional[Path] = None, 
                      num_inference_steps: int = 30, guidance_scale: float = 7.5,
                      width: int = 512, height: int = 512) -> Optional[str]:
        """
        Синхронная генерация изображения по текстовому описанию.
        
        Args:
            prompt: Текстовое описание изображения
            output_path: Путь для сохранения изображения (опционально)
            num_inference_steps: Количество шагов генерации (больше = качественнее, но медленнее)
            guidance_scale: Сила следования промпту (7.5 - стандартное значение)
            width: Ширина изображения
            height: Высота изображения
        
        Returns:
            Путь к сохраненному изображению или None в случае ошибки
        """
        if not prompt or not prompt.strip():
            logger.warning("Пустой промпт для генерации изображения")
            return None
        
        try:
            # Загружаем модель если еще не загружена
            if not self._initialized:
                # Для синхронного метода используем asyncio.run
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self._load_model())
            
            return self._generate_image_sync(prompt, output_path, num_inference_steps, 
                                            guidance_scale, width, height)
        except Exception as e:
            logger.error(f"Ошибка генерации изображения через Stable Diffusion: {e}", exc_info=True)
            return None
    
    def _generate_image_sync(self, prompt: str, output_path: Optional[Path] = None,
                            num_inference_steps: int = 30, guidance_scale: float = 7.5,
                            width: int = 512, height: int = 512) -> str:
        """Синхронная генерация изображения через локальную модель"""
        logger.info(f"Генерация изображения через локальный Stable Diffusion: {prompt[:50]}...")
        
        # Генерируем изображение
        with torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]
        
        # Сохраняем изображение
        if output_path is None:
            image_id = str(uuid4())
            output_path = STORAGE_DIR / f"{image_id}.png"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, "PNG")
        logger.info(f"Изображение сохранено: {output_path}")
        
        return str(output_path)
    
    async def generate_image_async(self, prompt: str, output_path: Optional[Path] = None,
                                   num_inference_steps: int = 30, guidance_scale: float = 7.5,
                                   width: int = 512, height: int = 512) -> Optional[str]:
        """
        Асинхронная версия генерации изображения через локальный Stable Diffusion.
        
        Args:
            prompt: Текстовое описание изображения
            output_path: Путь для сохранения изображения (опционально)
            num_inference_steps: Количество шагов генерации (больше = качественнее, но медленнее)
            guidance_scale: Сила следования промпту (7.5 - стандартное значение)
            width: Ширина изображения
            height: Высота изображения
        
        Returns:
            Путь к сохраненному изображению или None в случае ошибки
        """
        if not prompt or not prompt.strip():
            logger.warning("Пустой промпт для генерации изображения")
            return None
        
        try:
            # Загружаем модель если еще не загружена
            if not self._initialized:
                await self._load_model()
            
            # Генерируем изображение в отдельном потоке, чтобы не блокировать event loop
            return await asyncio.to_thread(
                self._generate_image_sync,
                prompt,
                output_path,
                num_inference_steps,
                guidance_scale,
                width,
                height
            )
        except Exception as e:
            logger.error(f"Ошибка генерации изображения через Stable Diffusion: {e}", exc_info=True)
            return None
    


# Глобальный экземпляр генератора изображений
# Инициализируется лениво при первом использовании
_image_generator_instance = None


def get_image_generator() -> ImageGenerator:
    """Получить глобальный экземпляр генератора изображений"""
    global _image_generator_instance
    
    if _image_generator_instance is None:
        model_id = settings.STABLE_DIFFUSION_MODEL_ID
        device = settings.STABLE_DIFFUSION_DEVICE
        _image_generator_instance = ImageGenerator(model_id=model_id, device=device)
    
    return _image_generator_instance


# Создаем объект-прокси для обратной совместимости
class ImageGeneratorProxy:
    """Прокси для обратной совместимости с существующим кодом"""
    
    async def generate_image_async(self, prompt: str, output_path: Optional[Path] = None, **kwargs):
        """Асинхронная генерация с параметрами по умолчанию из настроек"""
        generator = get_image_generator()
        # Используем параметры из настроек, если не указаны явно
        num_inference_steps = kwargs.get("num_inference_steps", settings.STABLE_DIFFUSION_STEPS)
        guidance_scale = kwargs.get("guidance_scale", settings.STABLE_DIFFUSION_GUIDANCE_SCALE)
        width = kwargs.get("width", settings.STABLE_DIFFUSION_WIDTH)
        height = kwargs.get("height", settings.STABLE_DIFFUSION_HEIGHT)
        
        return await generator.generate_image_async(
            prompt=prompt,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )
    
    def generate_image(self, prompt: str, output_path: Optional[Path] = None, **kwargs):
        """Синхронная генерация с параметрами по умолчанию из настроек"""
        generator = get_image_generator()
        # Используем параметры из настроек, если не указаны явно
        num_inference_steps = kwargs.get("num_inference_steps", settings.STABLE_DIFFUSION_STEPS)
        guidance_scale = kwargs.get("guidance_scale", settings.STABLE_DIFFUSION_GUIDANCE_SCALE)
        width = kwargs.get("width", settings.STABLE_DIFFUSION_WIDTH)
        height = kwargs.get("height", settings.STABLE_DIFFUSION_HEIGHT)
        
        return generator.generate_image(
            prompt=prompt,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )


image_generator = ImageGeneratorProxy()
