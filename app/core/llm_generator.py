# app/core/llm_generator.py
import os

# Убеждаемся, что CUDA отключена ДО импорта torch
# (на случай если переменные не были установлены в main.py)
force_cpu_env = os.getenv("FORCE_CPU", "true").lower() == "true"
if force_cpu_env and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        self._load_model()

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
            max_retries = 3
            retry_count = 0
            tokenizer_loaded = False
            
            while retry_count < max_retries and not tokenizer_loaded:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        settings.LLM_MODEL,
                        trust_remote_code=True,
                        token=hf_token if hf_token else None,
                    )
                    tokenizer_loaded = True
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"⚠️  Попытка {retry_count}/{max_retries} загрузки токенайзера не удалась: {e}")
                        import time
                        time.sleep(5 * retry_count)  # Экспоненциальная задержка
                    else:
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
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    
                    logger.info("Загрузка модели с 8-bit квантованием (уменьшает память в 2 раза)")
                    # Загрузка модели с retry логикой
                    max_retries = 3
                    retry_count = 0
                    model_loaded = False
                    
                    while retry_count < max_retries and not model_loaded:
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                settings.LLM_MODEL,
                                trust_remote_code=True,
                                quantization_config=quantization_config,
                                device_map="auto",
                                token=hf_token if hf_token else None,
                            )
                            model_loaded = True
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                logger.warning(f"⚠️  Попытка {retry_count}/{max_retries} загрузки модели не удалась: {e}")
                                import time
                                time.sleep(10 * retry_count)  # Экспоненциальная задержка
                            else:
                                raise
                    logger.info("✅ Модель загружена с квантованием")
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
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )
                else:
                    # Принудительно загружаем на CPU с retry логикой
                    max_retries = 3
                    retry_count = 0
                    model_loaded = False
                    
                    while retry_count < max_retries and not model_loaded:
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                settings.LLM_MODEL,
                                trust_remote_code=True,
                                torch_dtype=torch.float32,
                                device_map="cpu",  # Явно указываем CPU
                                token=hf_token if hf_token else None,
                            )
                            model_loaded = True
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                logger.warning(f"⚠️  Попытка {retry_count}/{max_retries} загрузки модели не удалась: {e}")
                                import time
                                time.sleep(10 * retry_count)  # Экспоненциальная задержка
                            else:
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
    ) -> Dict[str, Any]:
        """
        Генерация содержимого ОДНОГО слайда по пользовательскому промпту,
        с учётом аудитории и контекста из документов.

        Использует chat template Qwen для корректного формата запросов и
        ограничивает длину генерируемого текста через settings.MAX_NEW_TOKENS.
        """
        if not self.is_loaded:
            raise Exception("Модель не загружена")

        # Инструкция по аудитории
        audience_instr = self.audience_instructions(audience)

        # Формируем чат-сообщения: system описывает роль и формат, user содержит
        # сам запрос пользователя и контекст из документов (обрезаем до 800 символов)
        messages = [
            {
                "role": "system",
                "content": f"Ты – эксперт по презентациям. {audience_instr}\n\nВажно: генерируй развернутый контент для слайда - минимум 5-7 информативных пунктов. Каждый пункт должен быть содержательным и полезным.",
            },
            {
                "role": "user",
                "content": f"{user_prompt.strip()}\n\nКонтекст:\n{context[:800]}\n\nСгенерируй подробное содержимое слайда с 5-7 информативными пунктами.",
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

        # Приводим ответ к аккуратному виду: оставляем 5-8 маркерованных строк
        content = self._clean_content(content, slide_type="custom")

        return {"content": content, "audience": audience, "status": "success"}

    # -------- очистка текста --------

    @staticmethod
    def _clean_content(text: str, slide_type: str) -> str:
        """
        Приводим ответ модели к аккуратному виду:
        - для title: короткий заголовок,
        - для остальных: 5-8 маркерованных пунктов для более полного контента.
        """
        if slide_type == "title":
            first_line = text.split('\n')[0].strip()
            first_line = first_line.replace('"', '').replace("'", "")
            words = first_line.split()[:6]
            return ' '.join(words)
        else:
            lines = []
            for line in text.split('\n'):
                line = line.strip()

                if not line:
                    continue

                # убираем возможный префикс вроде "• " или "- "
                if line.startswith("•") or line.startswith("-"):
                    clean_line = line[1:].strip()
                else:
                    clean_line = line

                # фильтр по длине, чтобы не тащить мусор
                if len(clean_line) > 10:
                    if len(clean_line) <= 200:  # увеличен лимит длины для более развернутых пунктов
                        lines.append(f"• {clean_line}")

                # Увеличено до 8 пунктов для более полного контента
                if len(lines) >= 8:
                    break

            if not lines:
                return "• Информация готовится\n• Данные анализируются\n• Результаты будут представлены"

            return "\n".join(lines)

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.is_loaded else "not_loaded",
            "model": settings.LLM_MODEL
        }


content_generator = ContentGenerator()

