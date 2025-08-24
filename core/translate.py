# core/translate.py
from __future__ import annotations
import time, logging
from typing import Optional
from .config import settings

SYSTEM_INSTR = (
    "You are a high-quality translator. Translate the following Yiddish text to English. "
    "Preserve names, honorifics and formatting where possible. Return only the translation."
)

log = logging.getLogger("preprocess.translate")

def _sample(txt: str, n: int = 80) -> str:
    t = (txt or "").replace("\n", " ")
    return (t[:n] + "…") if len(t) > n else t

class Translator:
    def __init__(self):
        self.provider = settings.tr_provider
        self.model = None
        self.client = None

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = settings.tr_model_openai

        elif self.provider == "cohere":
            import cohere
            self.client = cohere.Client(api_key=settings.cohere_api_key)
            self.model = settings.tr_model_cohere

        elif self.provider == "gemini":
            import google.generativeai as genai
            self.client = genai
            self.client.configure(api_key=settings.gemini_api_key)
            self.model = settings.tr_model_gemini

        elif not self.provider or self.provider == "none":
            pass
        else:
            raise ValueError(f"TR_PROVIDER desconocido: {self.provider}")

        log.info("Translator init: provider=%s model=%s", self.provider, self.model)

    def translate_to_en(self, text: str) -> Optional[str]:
        if not text or self.provider == "none":
            return None

        t0 = time.time()
        ok = False
        out = None
        err = None

        try:
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_INSTR},
                              {"role": "user",   "content": text}],
                    temperature=0,
                )
                out = (resp.choices[0].message.content or "").strip()
                ok = bool(out)

            elif self.provider == "cohere":
                try:
                    resp = self.client.chat(
                        model=self.model,
                        messages=[{"role": "system", "content": SYSTEM_INSTR},
                                  {"role": "user",   "content": text}],
                        temperature=0,
                    )
                    if hasattr(resp, "message") and getattr(resp.message, "content", None):
                        parts = resp.message.content
                        if isinstance(parts, list) and parts and hasattr(parts[0], "text"):
                            out = parts[0].text.strip()
                        else:
                            out = getattr(resp, "text", None)
                    else:
                        out = getattr(resp, "text", None)
                except Exception:
                    # fallback a generate
                    resp = self.client.generate(
                        model=self.model,
                        prompt=f"{SYSTEM_INSTR}\n\n{text}",
                        temperature=0,
                    )
                    if hasattr(resp, "generations") and resp.generations:
                        out = resp.generations[0].text.strip()
                ok = bool(out)

            elif self.provider == "gemini":
                model = self.client.GenerativeModel(self.model)
                resp = model.generate_content([SYSTEM_INSTR, text])
                out = (getattr(resp, "text", None) or "").strip()
                ok = bool(out)

        except Exception as e:
            err = e

        dt = (time.time() - t0) * 1000.0
        if ok:
            log.debug(
                "TR ok provider=%s model=%s ms=%.0f in_chars=%d out_chars=%d sample_in='%s'",
                self.provider, self.model, dt, len(text), len(out or ""), _sample(text)
            )
            return out
        else:
            log.warning(
                "TR fail provider=%s model=%s ms=%.0f in_chars=%d sample_in='%s' err=%s",
                self.provider, self.model, dt, len(text), _sample(text), err
            )
            return None