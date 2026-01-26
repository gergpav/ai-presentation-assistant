# app/core/llm_generator.py
import os

# Убеждаемся, что CUDA настроена правильно ДО импорта torch
# (на случай если переменные не были установлены в main.py)
force_cpu_env = os.getenv("FORCE_CPU", "false").lower() == "true"  # По умолчанию false - используем GPU
if force_cpu_env and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Отключаем GPU только если FORCE_CPU=true

# Настройки HuggingFace Hub для надежной загрузки моделей
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # Отключаем hf_transfer для совместимости
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")  # 10 минут таймаут для загрузки
os.environ.setdefault("REQUESTS_TIMEOUT", "600")  # Таймаут для requests

# Подавляем предупреждение о kernels CPU ядре (не критично, bitsandbytes работает без него)
import warnings
import logging
warnings.filterwarnings("ignore", message=".*Failed to load CPU gemm_4bit_forward.*")
warnings.filterwarnings("ignore", message=".*Cannot install kernel from repo kernels-community.*")
# Подавляем предупреждения от bitsandbytes/kernels
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("kernels").setLevel(logging.ERROR)

import torch
from torch import dtype
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import settings
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)

if force_cpu_env:
    logger.info("CUDA принудительно отключен через FORCE_CPU=true")

# Оптимизация потоков для CPU (можно настроить через переменные окружения)
omp_threads = int(os.getenv("OMP_NUM_THREADS", "4"))  # Уменьшено с 8 до 4 для экономии ресурсов
mkl_threads = int(os.getenv("MKL_NUM_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(mkl_threads))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch.set_num_threads(omp_threads)
torch.set_num_interop_threads(2)


class ContentGenerator:
    def __init__(self):
        self.is_loaded = False
        self.tokenizer = None
        self.model = None
        self._loading = False
        # Ленивая загрузка - загружаем модель только при первом использовании
        # Это позволяет избежать загрузки модели в app контейнере, если она не нужна
        # Модель будет загружена при первом вызове generate_from_prompt или health_check
    
    def _ensure_loaded(self):
        """Обеспечивает загрузку модели при первом использовании"""
        if not self.is_loaded and not self._loading:
            self._loading = True
            try:
                self._load_model()
            finally:
                self._loading = False

    def _load_model(self):
        try:
            logger.info(f"Загрузка модели: {settings.LLM_MODEL}")

            # Настройка HuggingFace для работы с сетью
            # Можно использовать токен через переменную окружения HF_TOKEN
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                from huggingface_hub import login
                try:
                    login(token=hf_token)
                    logger.info("✅ Авторизация в HuggingFace выполнена")
                except Exception as e:
                    logger.warning(f"⚠️  Не удалось авторизоваться в HuggingFace: {e}")

            # Загружаем токенайзер с обработкой сетевых ошибок
            max_retries = 5  # Увеличено с 3 до 5
            retry_count = 0
            tokenizer_loaded = False
            
            while retry_count < max_retries and not tokenizer_loaded:
                try:
                    logger.info(f"Загрузка токенайзера (попытка {retry_count + 1}/{max_retries})...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        settings.LLM_MODEL,
                        trust_remote_code=True,
                        token=hf_token if hf_token else None,
                        timeout=600,  # 10 минут таймаут
                    )
                    tokenizer_loaded = True
                    logger.info("✅ Токенайзер загружен успешно")
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * retry_count  # Увеличена задержка: 10, 20, 30, 40 секунд
                        logger.warning(f"⚠️  Попытка {retry_count}/{max_retries} загрузки токенайзера не удалась: {e}")
                        logger.info(f"⏳ Повторная попытка через {wait_time} секунд...")
                        import time
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ Не удалось загрузить токенайзер после {max_retries} попыток")
                        raise

            # Определяем, использовать ли GPU (по умолчанию используем GPU если доступен)
            force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"  # По умолчанию false - используем GPU
            use_cuda = False
            
            if not force_cpu and torch.cuda.is_available():
                try:
                    device_capability = torch.cuda.get_device_capability(0)
                    device_name = torch.cuda.get_device_name(0)
                    # Поддерживаем все версии CUDA capability (включая sm_120 для RTX 50xx)
                    use_cuda = True
                    logger.info(f"✅ Используем GPU {device_name} с compute capability {device_capability[0]}.{device_capability[1]}")
                except Exception as e:
                    logger.warning(f"Ошибка проверки CUDA: {e}, используем CPU")
                    use_cuda = False
            else:
                if force_cpu:
                    logger.info("FORCE_CPU=true, используем CPU для генерации")
                elif not torch.cuda.is_available():
                    logger.info("CUDA недоступна, используем CPU для генерации")
            
            if not use_cuda:
                logger.info("Используем CPU для генерации")
            
            # Проверяем, нужно ли использовать квантование для CPU
            use_quantization = settings.USE_QUANTIZATION and not use_cuda
            
            if use_quantization:
                # Пробуем использовать 8-bit квантование для экономии памяти
                try:
                    # Подавляем предупреждения от bitsandbytes при импорте
                    import sys
                    from io import StringIO
                    old_stderr = sys.stderr
                    sys.stderr = StringIO()
                    try:
                        from transformers import BitsAndBytesConfig
                    finally:
                        sys.stderr = old_stderr
                    
                    # Для CPU нужны специальные настройки квантования
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        llm_int8_enable_fp32_cpu_offload=True,  # Разрешаем offload на CPU
                    )
                    
                    logger.info("Загрузка модели с 8-bit квантованием (уменьшает память в 2 раза)")
                    # Загрузка модели с retry логикой
                    max_retries = 5  # Увеличено с 3 до 5
                    retry_count = 0
                    model_loaded = False
                    
                    while retry_count < max_retries and not model_loaded:
                        try:
                            logger.info(f"Загрузка модели с квантованием (попытка {retry_count + 1}/{max_retries})...")
                            # Для CPU используем явный device_map="cpu" с правильными настройками
                            if not use_cuda:
                                # Для CPU квантования нужен явный device_map="cpu"
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    settings.LLM_MODEL,
                                    trust_remote_code=True,
                                    quantization_config=quantization_config,
                                    device_map="cpu",
                                    token=hf_token if hf_token else None,
                                )
                            else:
                                # Для GPU используем "auto"
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    settings.LLM_MODEL,
                                    trust_remote_code=True,
                                    quantization_config=quantization_config,
                                    device_map="auto",
                                    token=hf_token if hf_token else None,
                                )
                            model_loaded = True
                            logger.info("✅ Модель загружена с квантованием")
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = 15 * retry_count  # Увеличена задержка: 15, 30, 45, 60 секунд
                                logger.warning(f"⚠️  Попытка {retry_count}/{max_retries} загрузки модели не удалась: {e}")
                                logger.info(f"⏳ Повторная попытка через {wait_time} секунд...")
                                import time
                                time.sleep(wait_time)
                            else:
                                logger.error(f"❌ Не удалось загрузить модель после {max_retries} попыток")
                                raise
                except ImportError:
                    logger.warning("bitsandbytes не установлен. Установите: pip install bitsandbytes")
                    logger.info("Загружаем модель без квантования")
                    use_quantization = False
                except Exception as e:
                    logger.warning(f"Ошибка загрузки с квантованием: {e}. Загружаем без квантования")
                    use_quantization = False
            
            if not use_quantization:
                # Явно указываем CPU для избежания CUDA ошибок
                if use_cuda:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        settings.LLM_MODEL,
                        trust_remote_code=True,
                        dtype=torch.float16,  # Используем dtype вместо torch_dtype
                        device_map="auto",
                    )
                else:
                    # Принудительно загружаем на CPU с retry логикой
                    max_retries = 5  # Увеличено с 3 до 5
                    retry_count = 0
                    model_loaded = False
                    
                    while retry_count < max_retries and not model_loaded:
                        try:
                            logger.info(f"Загрузка модели на CPU (попытка {retry_count + 1}/{max_retries})...")
                            self.model = AutoModelForCausalLM.from_pretrained(
                                settings.LLM_MODEL,
                                trust_remote_code=True,
                                dtype=torch.float32,  # Используем dtype вместо torch_dtype
                                device_map="cpu",  # Явно указываем CPU
                                token=hf_token if hf_token else None,
                            )
                            model_loaded = True
                            logger.info("✅ Модель загружена на CPU")
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = 15 * retry_count  # Увеличена задержка: 15, 30, 45, 60 секунд
                                logger.warning(f"⚠️  Попытка {retry_count}/{max_retries} загрузки модели не удалась: {e}")
                                logger.info(f"⏳ Повторная попытка через {wait_time} секунд...")
                                import time
                                time.sleep(wait_time)
                            else:
                                logger.error(f"❌ Не удалось загрузить модель после {max_retries} попыток")
                                raise
                    # Дополнительно перемещаем модель на CPU (на случай если device_map не сработал)
                    if not use_quantization:  # Квантованные модели нельзя перемещать
                        self.model = self.model.to(torch.device("cpu"))

            # eval() обязательно (для не-квантованных моделей)
            if not use_quantization:
                self.model.eval()

            # pad_token_id иногда отсутствует → ставим EOS
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.is_loaded = True
            logger.info("✅ LLM модель успешно загружена")
            logger.info(
                f"LLM device={getattr(self.model, 'device', None)}, "
                f"cuda={torch.cuda.is_available()}, dtype={dtype}"
            )

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели {settings.LLM_MODEL}: {e}")
            self.is_loaded = False  # не падаем, просто помечаем как незагруженную

    # -------- вспомогательная логика под аудитории --------

    @staticmethod
    def audience_instructions(audience: str) -> str:
        """
        Возвращает текст-инструкцию для LLM в зависимости от аудитории.
        Аудитории: инвесторы, топ-менеджеры, эксперты.
        """
        a = (audience or "").lower()

        if "инвест" in a:
            return (
                "Аудитория: инвесторы. Делай акцент на следующих аспектах:\n"
                "- размер и рост рынка, конкурентные преимущества;\n"
                "- финансовые показатели: выручка, прибыль, EBITDA, денежный поток;\n"
                "- ключевые метрики эффективности инвестиций: ROI, IRR, срок окупаемости;\n"
                "- основные риски и способы их снижения.\n"
                "Избегай излишних технических деталей, формулируй чётко и по делу."
            )

        if "топ" in a or "менедж" in a:
            return (
                "Аудитория: топ-менеджмент компании. Делай акцент на:\n"
                "- стратегическом эффекте для компании и влиянии на ключевые KPI;\n"
                "- влиянии на выручку, маржинальность, операционные расходы;\n"
                "- рисках внедрения, ресурсоёмкости и сроках реализации;\n"
                "- интеграции с текущими процессами и системами.\n"
                "Формулировки должны быть управленческими, без глубоких технических деталей."
            )

        if "эксперт" in a:
            return (
                "Аудитория: технические и предметные эксперты. Делай акцент на:\n"
                "- архитектуре решения, ключевых технологиях и стеке;\n"
                "- алгоритмах, подходах, ограничениях и допущениях;\n"
                "- метриках качества, производительности и надёжности;\n"
                "- потенциальных рисках реализации и техническом долге.\n"
                "Можно использовать профессиональные термины и более детальные формулировки."
            )

        # дефолт: сбалансированный стиль
        return (
            "Аудитория: смешанная. Сбалансируй бизнес-аспекты и технические детали.\n"
            "Избегай слишком сложных терминов без необходимости."
        )


    def generate_from_prompt(
        self,
        user_prompt: str,
        context: str,
        audience: str = "инвесторы",
        visual_type: str = "text",
        max_chars: int = 800,
    ) -> Dict[str, Any]:
        """
        Генерация содержимого ОДНОГО слайда по пользовательскому промпту,
        с учётом аудитории и контекста из документов.
        
        Использует chat template Qwen для корректного формата запросов и
        ограничивает длину генерируемого текста через settings.MAX_NEW_TOKENS.
        """
        # Ленивая загрузка модели при первом использовании
        self._ensure_loaded()
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")

        # Инструкция по аудитории
        audience_instr = self.audience_instructions(audience)

        # Определяем тип макета на основе промпта
        layout_type = self._determine_layout(user_prompt, visual_type)
        
        # Для титульного слайда генерируем заголовок и подзаголовок из промпта
        if layout_type == "title":
            # Генерируем заголовок и подзаголовок для титульного слайда
            title_instruction = "Ты – эксперт по презентациям. Сгенерируй заголовок и подзаголовок для титульного слайда презентации.\n\nФормат ответа:\nЗАГОЛОВОК: [название презентации или проекта]\nПОДЗАГОЛОВОК: [дополнительная информация: автор, дата, организация и т.д.]\n\nВАЖНО: Используй информацию из промпта пользователя. Если в промпте указан только заголовок, подзаголовок можно не указывать (оставь пустым)."
            
            messages_title = [
                {
                    "role": "system",
                    "content": title_instruction,
                },
                {
                    "role": "user",
                    "content": f"{user_prompt.strip()}\n\nСгенерируй заголовок и подзаголовок для титульного слайда.",
                },
            ]
            
            # Применяем chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                text_title = self.tokenizer.apply_chat_template(
                    messages_title,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text_title = "\n\n".join([msg["content"] for msg in messages_title])
            
            # Генерируем заголовок и подзаголовок
            model_inputs_title = self.tokenizer([text_title], return_tensors="pt")
            try:
                model_device = next(self.model.parameters()).device
                device_title = model_device
                if force_cpu_env:
                    device_title = torch.device('cpu')
            except (StopIteration, AttributeError):
                device_title = torch.device('cpu')
            
            model_inputs_title = {k: v.to(device_title) for k, v in model_inputs_title.items()}
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id or eos_id
            
            with torch.inference_mode():
                out_ids_title = self.model.generate(
                    **model_inputs_title,
                    max_new_tokens=min(settings.MAX_NEW_TOKENS, 100),  # Ограничиваем для титульного слайда
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            
            gen_ids_title = out_ids_title[:, model_inputs_title["input_ids"].shape[1]:]
            title_content = self.tokenizer.batch_decode(gen_ids_title, skip_special_tokens=True)[0].strip()
            
            # Парсим заголовок и подзаголовок
            title_lines = [line.strip() for line in title_content.split('\n') if line.strip()]
            title_text = ""
            subtitle_text = ""
            
            for line in title_lines:
                if line.startswith("ЗАГОЛОВОК:") or line.startswith("Заголовок:"):
                    title_text = line.split(":", 1)[1].strip() if ":" in line else line
                elif line.startswith("ПОДЗАГОЛОВОК:") or line.startswith("Подзаголовок:"):
                    subtitle_text = line.split(":", 1)[1].strip() if ":" in line else line
                elif not title_text:
                    title_text = line
                elif not subtitle_text:
                    subtitle_text = line
            
            # Если не нашли структурированный формат, используем первую строку как заголовок
            if not title_text and title_lines:
                title_text = title_lines[0]
                if len(title_lines) > 1:
                    subtitle_text = title_lines[1]
            
            # Формируем контент для титульного слайда
            title_content_result = f"ЗАГОЛОВОК: {title_text}"
            if subtitle_text:
                title_content_result += f"\nПОДЗАГОЛОВОК: {subtitle_text}"
            
            return {
                "content": title_content_result,
                "audience": audience,
                "status": "success",
                "layout": layout_type,
                "visual_type": visual_type,
                "title": title_text,
                "subtitle": subtitle_text
            }
        
        # Формируем инструкции для генерации в зависимости от типа визуализации
        # Нормализуем visual_type: если это enum или объект, берем строковое значение
        visual_type_str = visual_type.value if hasattr(visual_type, 'value') else str(visual_type)
        visual_instructions = self._get_visual_instructions(visual_type_str, user_prompt)
        
        # Логируем тип визуализации для отладки
        logger.info(f"Генерация контента для типа визуализации: {visual_type_str}")
        logger.debug(f"Инструкции для визуализации: {visual_instructions[:200]}...")
        
        # Формируем инструкции по ограничению символов
        char_limit_instruction = f"\n\nВАЖНО: Общий объем текста должен быть не более {max_chars} символов. Текст должен быть законченным, предложения не должны обрываться. Если достигнут лимит символов, заверши текущее предложение и закончи генерацию."

        # Инструкции по разделению заголовка и контента
        title_instruction = "\n\nВАЖНО: НЕ включай заголовок слайда в сгенерированный контент. Генерируй ТОЛЬКО содержимое слайда (текст, таблицу, данные для графика и т.д.), без заголовка."

        # Формируем чат-сообщения: system описывает роль и формат, user содержит
        # сам запрос пользователя и контекст из документов (обрезаем до 600 символов для ускорения)
        # Уменьшаем длину контекста для более быстрой генерации
        context_trimmed = context[:600] if context else ""
        messages = [
            {
                "role": "system",
                "content": f"Ты – эксперт по презентациям. {audience_instr}\n\n{visual_instructions}{char_limit_instruction}{title_instruction}\n\nВажно: НЕ используй markdown форматирование (**жирный**, *курсив*). Используй только обычный текст. Для выделения важных моментов используй заглавные буквы или структурируй текст списками.",
            },
            {
                "role": "user",
                "content": f"{user_prompt.strip()}\n\nКонтекст:\n{context_trimmed}\n\nСгенерируй содержимое слайда согласно требованиям выше. НЕ включай заголовок в ответ.",
            },
        ]

        # Применяем chat template, который добавляет необходимую обёртку для Qwen
        # и позволяет корректно отделять вопрос от ответа
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # fallback: склеиваем строки напрямую, если чат template отсутствует
            text = "\n\n".join([msg["content"] for msg in messages])

        # Токенизируем текст и переносим входные тензоры на устройство модели
        model_inputs = self.tokenizer([text], return_tensors="pt")
        
        # Определяем устройство модели (GPU если доступен и не принудительно CPU)
        try:
            model_device = next(self.model.parameters()).device
            device = model_device
            # Если принудительно CPU, перемещаем на CPU
            if force_cpu_env:
                device = torch.device('cpu')
                if model_device.type != 'cpu':
                    logger.info(f"Перемещаем модель с {model_device} на CPU (FORCE_CPU=true)")
                    self.model = self.model.to(device)
            else:
                # Используем устройство модели (скорее всего GPU)
                device = model_device
                logger.debug(f"Используем устройство модели: {device}")
        except (StopIteration, AttributeError):
            # Fallback на CPU если не удалось определить устройство
            device = torch.device('cpu')
        
        # Перемещаем входные тензоры на то же устройство, что и модель
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        # Генерируем текст, ограничивая длину settings.MAX_NEW_TOKENS
        # Используем inference_mode для оптимизации и более быструю генерацию
        import time
        start_time = time.time()
        logger.info(f"Начинаем генерацию с max_new_tokens={settings.MAX_NEW_TOKENS} на устройстве {device}")
        
        # Оптимизации для GPU
        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": settings.MAX_NEW_TOKENS,
            "do_sample": False,  # Greedy decoding для скорости
            "num_beams": 1,  # Одиночный beam для максимальной скорости
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
            "use_cache": True,  # KV cache для ускорения
            "output_attentions": False,
            "output_hidden_states": False,
            "repetition_penalty": 1.1,
        }
        
        # Для GPU добавляем дополнительные оптимизации
        if device.type == 'cuda':
            # Используем torch.inference_mode для GPU (быстрее чем no_grad)
            generation_kwargs["temperature"] = None  # Отключаем temperature для greedy
            logger.debug("Используем GPU оптимизации для генерации")
        
        try:
            # Используем torch.inference_mode() для ускорения (работает и на CPU и на GPU)
            with torch.inference_mode():
                out_ids = self.model.generate(**generation_kwargs)
            
            elapsed = time.time() - start_time
            logger.info(f"Генерация завершена за {elapsed:.2f} секунд, сгенерировано токенов: {out_ids.shape[1] - model_inputs['input_ids'].shape[1]}")
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                # Если возникла CUDA ошибка или OOM, пробуем переместить модель на CPU и повторить
                logger.warning(f"Обнаружена CUDA ошибка: {e}. Перемещаем модель на CPU и повторяем...")
                cpu_device = torch.device("cpu")
                self.model = self.model.to(cpu_device)
                model_inputs = {k: v.to(cpu_device) for k, v in model_inputs.items()}
                generation_kwargs_cpu = {**generation_kwargs}
                generation_kwargs_cpu.update(model_inputs)
                generation_kwargs_cpu.pop("temperature", None)  # Убираем temperature для CPU
                with torch.inference_mode():
                    out_ids = self.model.generate(**generation_kwargs_cpu)
                elapsed = time.time() - start_time
                logger.info(f"Генерация завершена после CUDA ошибки за {elapsed:.2f} секунд, сгенерировано токенов: {out_ids.shape[1] - model_inputs['input_ids'].shape[1]}")
            else:
                raise

        # Извлекаем только сгенерированную часть (новые токены) после prompt
        gen_ids = out_ids[:, model_inputs["input_ids"].shape[1]:]
        content = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # Приводим ответ к аккуратному виду с учетом типа визуализации и макета
        content = self._clean_content(content, layout_type=layout_type, visual_type=visual_type, max_chars=max_chars)

        return {
            "content": content, 
            "audience": audience, 
            "status": "success",
            "layout": layout_type,
            "visual_type": visual_type
        }

    # -------- определение макета слайда --------
    
    @staticmethod
    def _determine_layout(prompt: str, visual_type: str) -> str:
        """
        Определяет тип макета слайда на основе промпта и типа визуализации.
        Работает автоматически, не требуя явного указания типа слайда в промпте.
        
        Возвращает: "title", "two_content", "title_only", "title_and_content"
        """
        prompt_lower = prompt.lower().strip()
        words = prompt_lower.split()
        
        # Титульный слайд - если промпт очень короткий и содержит ключевые слова
        title_keywords = ["титульный", "титул", "обложка", "cover", "title slide", "начало", "название проекта"]
        if any(kw in prompt_lower for kw in title_keywords) and len(words) <= 10:
            return "title"
        
        # Сравнение - если есть слова сравнения ИЛИ упоминаются два объекта для сравнения
        comparison_keywords = ["сравнение", "сравни", "против", "versus", "vs", "разница", "отличия", 
                              "преимущества и недостатки", "плюсы и минусы", "за и против"]
        comparison_patterns = ["против", "vs", "versus", "или", "либо"]
        
        # Проверяем наличие ключевых слов сравнения
        if any(kw in prompt_lower for kw in comparison_keywords):
            return "two_content"
        
        # Проверяем паттерны сравнения (два объекта через "против", "vs" и т.д.)
        for pattern in comparison_patterns:
            if pattern in prompt_lower:
                # Проверяем, что есть два объекта для сравнения
                parts = prompt_lower.split(pattern)
                if len(parts) >= 2:
                    # Если обе части содержат существительные/объекты
                    if len(parts[0].split()) >= 2 and len(parts[1].split()) >= 2:
                        return "two_content"
        
        # Только заголовок - если промпт очень короткий (1-3 слова) ИЛИ явно указано "без текста"
        no_content_keywords = ["без текста", "только название", "только заголовок", "title only", "без контента"]
        if any(kw in prompt_lower for kw in no_content_keywords):
            return "title_only"
        
        # Если промпт очень короткий (1-3 слова) и это не титульный слайд
        if len(words) <= 3 and visual_type == "text":
            # Проверяем, не является ли это просто названием
            if not any(word in ["создай", "сделай", "покажи", "опиши", "расскажи"] for word in words):
                return "title_only"
        
        # По умолчанию - заголовок и контент
        return "title_and_content"
    
    @staticmethod
    def _get_visual_instructions(visual_type: str, prompt: str) -> str:
        """
        Возвращает инструкции для генерации в зависимости от типа визуализации.
        """
        prompt_lower = prompt.lower()
        
        if visual_type == "table":
            return "Тип визуализации: ТАБЛИЦА. Сгенерируй данные ТОЛЬКО в формате таблицы с разделителем |. Используй СТРОГО такой формат:\nСтолбец1 | Столбец2 | Столбец3\nЗначение1 | Значение2 | Значение3\nЗначение2.1 | Значение2.2 | Значение2.3\n...\n\nКРИТИЧНО ВАЖНО:\n1. Каждая строка должна содержать разделитель | между значениями.\n2. Первая строка - это заголовки столбцов.\n3. Последующие строки - это данные.\n4. НЕ добавляй никакого текста до или после таблицы.\n5. НЕ используй markdown форматирование (``` или другие символы).\n6. Только строки таблицы с разделителями |, без дополнительного текста.\n7. Пример правильного формата:\nФинансовые показатели за последние 3 квартала | Значение\nВыручка | $500,000\nПрибыль | $150,000"
        
        elif visual_type == "chart":
            return "Тип визуализации: ГРАФИК. Сгенерируй данные ТОЛЬКО в формате:\nНазвание показателя: Числовое значение\nНазвание показателя 2: Числовое значение\n...\n\nКРИТИЧНО ВАЖНО:\n1. Каждая строка должна содержать название показателя, двоеточие и числовое значение.\n2. Значения должны быть числами (можно с десятичными точками, например: 150, 160.5, 175.25).\n3. НЕ добавляй никакого текста до или после данных графика.\n4. НЕ используй markdown форматирование или другие символы.\n5. Только данные в формате 'Название: Число', каждая строка на новой строке.\n6. Пример правильного формата:\nQ1 2024: 150\nQ2 2024: 160\nQ3 2024: 175\nQ4 2024: 190"
        
        elif visual_type == "image":
            return "Тип визуализации: ИЗОБРАЖЕНИЕ. Сгенерируй ТОЛЬКО описание изображения для генерации. Опиши детально, что должно быть на изображении: какие элементы, объекты, цвета, стиль, композиция.\n\nКРИТИЧНО ВАЖНО:\n1. НЕ добавляй никакого текста до или после описания.\n2. Только описание изображения на русском языке.\n3. Описание должно быть детальным и конкретным.\n4. Пример правильного формата:\nСхема технологического процесса производства с этапами: подготовка сырья, первичная обработка, вторичная обработка, упаковка, складирование. Стрелки показывают последовательность этапов. Стиль: схематичный, профессиональный, синие и зеленые цвета."
        
        else:  # text
            return "Тип визуализации: ТЕКСТ. Сгенерируй текстовое содержимое слайда в виде структурированного списка пунктов."

    # -------- очистка текста --------

    @staticmethod
    def _clean_content(text: str, layout_type: str = "title_and_content", visual_type: str = "text", max_chars: int = 800) -> str:
        """
        Приводим ответ модели к аккуратному виду с учетом:
        - типа макета (title, title_only, two_content, title_and_content)
        - типа визуализации (text, table, chart, image)
        - ограничения по символам (max_chars)
        - убираем markdown форматирование (**жирный**, *курсив*)
        """
        # Убираем markdown форматирование
        text = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "")

        # Убираем служебные параметры/шапки, которые иногда возвращает модель
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            meta_keywords = [
                "контрольные параметры",
                "объем текста",
                "смешанная аудитория",
                "баланс",
                "тип визуализации",
                "не использовать markdown",
                "содержимое слайда",
            ]
            start_from_content = False
            cleaned_lines = []
            for line in lines:
                line_lower = line.lower()
                # Если нашли явную метку "Содержимое слайда", начинаем собирать после нее
                if "содержимое слайда" in line_lower:
                    start_from_content = True
                    continue
                if not start_from_content:
                    # Пропускаем строки с мета-параметрами
                    if any(keyword in line_lower for keyword in meta_keywords):
                        continue
                # Убираем возможные префиксы вроде "Содержимое слайда:"
                cleaned_line = line
                if cleaned_line.lower().startswith("содержимое слайда"):
                    parts = cleaned_line.split(":", 1)
                    cleaned_line = parts[1].strip() if len(parts) > 1 else ""
                cleaned_line = cleaned_line.strip()
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
            if cleaned_lines:
                text = "\n".join(cleaned_lines)
        
        if layout_type == "title":
            first_line = text.split('\n')[0].strip()
            first_line = first_line.replace('"', '').replace("'", "")
            words = first_line.split()[:6]
            return ' '.join(words)
        
        if layout_type == "title_only":
            return ""  # Пустой контент для слайда только с заголовком
        
        # Обработка таблиц
        if visual_type == "table":
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                # Убираем markdown и лишние символы
                clean_line = line.replace("|", " | ").strip()
                if clean_line:
                    lines.append(clean_line)
            
            result = "\n".join(lines)
            # Обрезаем до лимита символов, но стараемся закончить строку
            if len(result) > max_chars:
                truncated = result[:max_chars]
                # Пытаемся найти последнюю полную строку
                last_newline = truncated.rfind('\n')
                if last_newline > max_chars * 0.7:  # Если нашли новую строку не слишком рано
                    result = truncated[:last_newline]
                else:
                    # Ищем последнее место, где можно закончить предложение
                    for end_char in ['.', ';', ',']:
                        last_end = truncated.rfind(end_char)
                        if last_end > max_chars * 0.7:
                            result = truncated[:last_end + 1]
                            break
                    else:
                        result = truncated
            return result
        
        # Обработка графиков
        if visual_type == "chart":
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Ищем формат "Название: Значение"
                if ':' in line:
                    clean_line = line.split(':')[0].strip() + ': ' + ':'.join(line.split(':')[1:]).strip()
                    if clean_line:
                        lines.append(clean_line)
            
            result = "\n".join(lines)
            if len(result) > max_chars:
                truncated = result[:max_chars]
                last_newline = truncated.rfind('\n')
                if last_newline > max_chars * 0.7:
                    result = truncated[:last_newline]
                else:
                    result = truncated
            return result
        
        # Обработка изображений
        if visual_type == "image":
            # Для изображений возвращаем описание
            result = text.strip()
            if len(result) > max_chars:
                truncated = result[:max_chars]
                # Пытаемся закончить предложение
                for end_char in ['.', '!', '?']:
                    last_end = truncated.rfind(end_char)
                    if last_end > max_chars * 0.7:
                        result = truncated[:last_end + 1]
                        break
                else:
                    result = truncated
            return result
        
        # Обработка текста (по умолчанию)
        lines = []
        current_length = 0
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # убираем возможный префикс вроде "• " или "- "
            if line.startswith("•") or line.startswith("-"):
                clean_line = line[1:].strip()
            else:
                clean_line = line

            # фильтр по длине
            if len(clean_line) > 10 and len(clean_line) <= 200:
                # Проверяем, не превысим ли лимит символов
                line_with_bullet = f"• {clean_line}\n"
                if current_length + len(line_with_bullet) > max_chars:
                    # Пытаемся закончить текущую строку
                    remaining = max_chars - current_length - 3  # -3 для "• \n"
                    if remaining > 20:  # Если осталось достаточно места
                        # Обрезаем строку и пытаемся закончить предложение
                        truncated_line = clean_line[:remaining]
                        for end_char in ['.', ';', ',']:
                            last_end = truncated_line.rfind(end_char)
                            if last_end > remaining * 0.7:
                                lines.append(f"• {truncated_line[:last_end + 1]}")
                                break
                        else:
                            lines.append(f"• {truncated_line}")
                    break
                
                lines.append(f"• {clean_line}")
                current_length += len(line_with_bullet)
                
                # Ограничение по количеству пунктов
                if len(lines) >= 8:
                    break

        if not lines:
            return "• Информация готовится\n• Данные анализируются\n• Результаты будут представлены"

        result = "\n".join(lines)
        # Финальная проверка лимита символов
        if len(result) > max_chars:
            truncated = result[:max_chars]
            last_newline = truncated.rfind('\n')
            if last_newline > max_chars * 0.7:
                result = truncated[:last_newline]
            else:
                result = truncated
        
        return result

    def health_check(self) -> Dict[str, Any]:
        # Для health_check не загружаем модель принудительно, чтобы не вызывать OOM
        # Модель загрузится автоматически при первом использовании generate_from_prompt
        return {
            "status": "healthy" if self.is_loaded else "not_loaded",
            "model": settings.LLM_MODEL
        }


content_generator = ContentGenerator()

