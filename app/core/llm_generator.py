# app/core/llm_generator.py
import os

# Убеждаемся, что CUDA отключена ДО импорта torch
# (на случай если переменные не были установлены в main.py)
force_cpu_env = os.getenv("FORCE_CPU", "true").lower() == "true"
if force_cpu_env and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

            # Принудительно используем CPU для sm_120 (RTX 5060 Ti) из-за несовместимости
            # Можно включить GPU через переменную окружения FORCE_CPU=false когда появится поддержка
            use_cuda = False
            force_cpu = os.getenv("FORCE_CPU", "true").lower() == "true"
            
            if not force_cpu and torch.cuda.is_available():
                try:
                    device_capability = torch.cuda.get_device_capability(0)
                    # sm_120 (12.0) пока не поддерживается - используем CPU
                    if device_capability[0] < 12:
                        use_cuda = True
                        logger.info(f"Используем GPU с compute capability {device_capability}")
                    else:
                        logger.warning(f"GPU compute capability {device_capability} не поддерживается, используем CPU")
                except Exception as e:
                    logger.warning(f"Ошибка проверки CUDA: {e}, используем CPU")
            
            if not use_cuda:
                logger.info("Используем CPU для генерации (GPU отключен или несовместим)")
            
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
        
        # Для титульного слайда не генерируем контент
        if layout_type == "title":
            return {
                "content": "", 
                "audience": audience, 
                "status": "success",
                "layout": layout_type,
                "visual_type": visual_type
            }
        
        # Формируем инструкции для генерации в зависимости от типа визуализации
        visual_instructions = self._get_visual_instructions(visual_type, user_prompt)
        
        # Формируем инструкции по ограничению символов
        char_limit_instruction = f"\n\nВАЖНО: Общий объем текста должен быть не более {max_chars} символов. Текст должен быть законченным, предложения не должны обрываться. Если достигнут лимит символов, заверши текущее предложение и закончи генерацию."

        # Инструкции по разделению заголовка и контента
        title_instruction = "\n\nВАЖНО: НЕ включай заголовок слайда в сгенерированный контент. Генерируй ТОЛЬКО содержимое слайда (текст, таблицу, данные для графика и т.д.), без заголовка."

        # Формируем чат-сообщения: system описывает роль и формат, user содержит
        # сам запрос пользователя и контекст из документов (обрезаем до 800 символов)
        messages = [
            {
                "role": "system",
                "content": f"Ты – эксперт по презентациям. {audience_instr}\n\n{visual_instructions}{char_limit_instruction}{title_instruction}\n\nВажно: НЕ используй markdown форматирование (**жирный**, *курсив*). Используй только обычный текст. Для выделения важных моментов используй заглавные буквы или структурируй текст списками.",
            },
            {
                "role": "user",
                "content": f"{user_prompt.strip()}\n\nКонтекст:\n{context[:800]}\n\nСгенерируй содержимое слайда согласно требованиям выше. НЕ включай заголовок в ответ.",
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
        # Принудительно используем CPU для избежания CUDA ошибок
        device = torch.device('cpu')
        # Убеждаемся, что модель на CPU
        try:
            model_device = next(self.model.parameters()).device
            if model_device.type != 'cpu':
                logger.warning(f"Модель на устройстве {model_device}, перемещаем на CPU")
                self.model = self.model.to(torch.device('cpu'))
        except (StopIteration, AttributeError):
            pass
        # Перемещаем входные тензоры на CPU
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        # Дополнительная проверка: убеждаемся, что модель не на CUDA
        try:
            if torch.cuda.is_available():
                # Если CUDA доступна, но мы хотим использовать CPU
                if force_cpu_env:
                    # Явно перемещаем модель на CPU еще раз
                    self.model = self.model.to(torch.device("cpu"))
                    logger.debug("Модель явно перемещена на CPU перед генерацией")
        except Exception as e:
            logger.warning(f"Ошибка при проверке устройства модели: {e}")

        # Генерируем текст, ограничивая длину settings.MAX_NEW_TOKENS
        # Используем inference_mode для оптимизации на CPU и более быструю генерацию
        import time
        start_time = time.time()
        logger.info(f"Начинаем генерацию с max_new_tokens={settings.MAX_NEW_TOKENS}")
        
        try:
            # Используем torch.inference_mode() для ускорения на CPU
            # Дополнительно отключаем градиенты глобально для CPU
            with torch.inference_mode():
                with torch.no_grad():  # Двойная защита от градиентов
                    out_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=settings.MAX_NEW_TOKENS,
                        do_sample=False,
                        num_beams=1,  # Одиночный beam для ускорения (greedy decoding)
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        use_cache=True,
                        # Оптимизации для CPU
                        output_attentions=False,
                        output_hidden_states=False,
                        # Дополнительные оптимизации
                        repetition_penalty=1.1,  # Небольшой penalty для избежания повторений и более разнообразной генерации
                    )
            
            elapsed = time.time() - start_time
            logger.info(f"Генерация завершена за {elapsed:.2f} секунд, сгенерировано токенов: {out_ids.shape[1] - model_inputs['input_ids'].shape[1]}")
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                # Если возникла CUDA ошибка, пробуем переместить модель на CPU и повторить
                logger.error(f"Обнаружена CUDA ошибка: {e}. Принудительно перемещаем модель на CPU и повторяем...")
                self.model = self.model.to(torch.device("cpu"))
                model_inputs = {k: v.to(torch.device("cpu")) for k, v in model_inputs.items()}
                with torch.inference_mode():
                    with torch.no_grad():
                        out_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=settings.MAX_NEW_TOKENS,
                            do_sample=False,
                            num_beams=1,
                            eos_token_id=eos_id,
                            pad_token_id=pad_id,
                            use_cache=True,
                            output_attentions=False,
                            output_hidden_states=False,
                            repetition_penalty=1.1,
                        )
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
            return "Тип визуализации: ТАБЛИЦА. Сгенерируй данные ТОЛЬКО в формате таблицы с разделителем |. Используй СТРОГО такой формат:\nСтолбец1 | Столбец2 | Столбец3\nЗначение1 | Значение2 | Значение3\nЗначение2.1 | Значение2.2 | Значение2.3\n...\n\nКРИТИЧНО ВАЖНО: Каждая строка должна содержать разделитель | между значениями. НЕ добавляй никакого текста до или после таблицы. Только строки таблицы с разделителями |."
        
        elif visual_type == "chart":
            if "график" in prompt_lower or "chart" in prompt_lower or "graph" in prompt_lower:
                return "Тип визуализации: ГРАФИК. Сгенерируй данные ТОЛЬКО в формате:\nНазвание показателя: Числовое значение\nНазвание показателя 2: Числовое значение\n...\n\nКРИТИЧНО ВАЖНО: Каждая строка должна содержать название показателя, двоеточие и числовое значение. Значения должны быть числами (можно с десятичными точками). НЕ добавляй никакого текста до или после данных графика."
            else:
                return "Тип визуализации: ГРАФИК. Сгенерируй данные для графика СТРОГО в формате:\nНазвание показателя: Числовое значение\nНазвание показателя 2: Числовое значение\n...\n\nКРИТИЧНО ВАЖНО: Только данные в формате 'Название: Число', без дополнительного текста."
        
        elif visual_type == "image":
            if "изображение" in prompt_lower or "image" in prompt_lower or "картинка" in prompt_lower:
                return "Тип визуализации: ИЗОБРАЖЕНИЕ. Сгенерируй описание изображения, которое должно быть создано. Опиши, что должно быть на изображении, какие элементы, цвета, стиль."
            else:
                return "Тип визуализации: ИЗОБРАЖЕНИЕ. Сгенерируй изображение. Опиши детали изображения, которое должно быть создано."
        
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

