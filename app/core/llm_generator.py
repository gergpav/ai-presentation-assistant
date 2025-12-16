# app/core/llm_generator.py
import os

import torch
from torch import dtype
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import settings
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch.set_num_threads(8)
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

            # Загружаем токенайзер и модель Qwen3-8B
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_MODEL,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            # eval() обязательно
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
                "content": f"Ты – эксперт по презентациям. {audience_instr}",
            },
            {
                "role": "user",
                "content": f"{user_prompt.strip()}\n\nКонтекст:\n{context[:800]}",
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
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        # Генерируем текст, ограничивая длину settings.MAX_NEW_TOKENS
        # Здесь мы не используем sampling (do_sample=False) для воспроизводимости
        out_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )

        # Извлекаем только сгенерированную часть (новые токены) после prompt
        gen_ids = out_ids[:, model_inputs["input_ids"].shape[1]:]
        content = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # Приводим ответ к аккуратному виду: оставляем 3–4 маркерованных строки
        content = self._clean_content(content, slide_type="custom")

        return {"content": content, "audience": audience, "status": "success"}

    # -------- очистка текста --------

    @staticmethod
    def _clean_content(text: str, slide_type: str) -> str:
        """
        Приводим ответ модели к аккуратному виду:
        - для title: короткий заголовок,
        - для остальных: 3–4 маркерованных пункта.
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
                    if len(clean_line) <= 180:  # ограничение длины
                        lines.append(f"• {clean_line}")

                if len(lines) >= 4:
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

