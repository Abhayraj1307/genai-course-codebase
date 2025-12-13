# flask-agent/llm_interface.py
from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
load_dotenv(_THIS_DIR / ".env")
load_dotenv(_THIS_DIR.parent / ".env")


def _looks_like_image_request(text: str) -> bool:
    t = text.strip().lower()

    # Always trigger on explicit command
    if t.startswith(("/img ", "/image ", "img:", "image:")):
        return True

    # Verb + noun later (covers "generate a cat image", "create logo", etc.)
    if re.search(r"\b(generate|create|make|draw|render|produce)\b.*\b(image|picture|photo|logo|poster)\b", t):
        return True

    # Common phrasing
    if "image of" in t or "picture of" in t or "photo of" in t:
        return True

    return False


def _extract_image_prompt(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^(\/img|\/image)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^(img:|image:)\s*", "", t, flags=re.IGNORECASE)
    return t.strip() or "A detailed, high-quality image."


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatBackend:
    """
    Flask backend without LangChain.

    Requirement mapping:
    1) No secondary LLM call for streaming -> single streaming OpenAI chat call.
    2) Image generation "if model allows" -> attempts gpt-image-1 and gracefully
       reports if not allowed (e.g., org not verified / no access).
    """

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Create flask-agent/.env and set OPENAI_API_KEY=..."
            )

        # Hard timeout so requests don't hang forever
        self.client = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)

        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self.image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()

        # Persistent history expected by existing app.py
        self.state_path = _THIS_DIR / "conversation_state_dev.json"
        self.history: List[Dict[str, str]] = self._load_state()

        # Image output folder (Flask serves /static/*)
        self.static_dir = _THIS_DIR / "static"
        self.generated_dir = self.static_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self.system_prompt = (
            "You are a helpful assistant. Be concise, correct, and practical."
        )

    # -------- required by app.py --------
    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def reset_conversation(self) -> None:
        self.history = []
        self._save_state()

    def _load_state(self) -> List[Dict[str, str]]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    # -------- main entry --------
    def predict(
        self,
        user_input: str,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        yield from self.stream(user_input=user_input, history=history)

    def stream(
        self,
        user_input: str,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        effective_history = list(history) if history is not None else self.history

        # IMAGE MODE
        if _looks_like_image_request(user_input):
            assistant_text = ""
            for chunk in self._handle_image_request(user_input):
                assistant_text += chunk
                yield chunk

            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_text})
            self._save_state()
            return

        # TEXT MODE (single streaming call)
        messages = self._build_messages(user_input, effective_history)
        assistant_accum = ""

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
        )

        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                assistant_accum += token
                yield token

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_accum})
        self._save_state()

    # -------- helpers --------
    def _build_messages(
        self,
        user_input: str,
        history: Iterable[Dict[str, str]],
    ) -> List[ChatMessage]:
        msgs: List[ChatMessage] = [ChatMessage(role="system", content=self.system_prompt)]

        for h in history:
            role = (h.get("role") or "").strip()
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append(ChatMessage(role=role, content=content))

        msgs.append(ChatMessage(role="user", content=user_input))
        return msgs

    def _handle_image_request(self, user_input: str) -> Generator[str, None, None]:
        prompt = _extract_image_prompt(user_input)

        # immediate UI feedback
        print("IMAGE MODE PROMPT:", prompt, flush=True)
        yield "\n[Generating image...]\n"

        # Try image API. If not allowed, DO NOT crash; stream a clear message.
        try:
            start = time.time()
            img = self.client.images.generate(
                model=self.image_model,
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
            print(f"IMAGE DONE in {time.time() - start:.1f}s", flush=True)

            b64 = img.data[0].b64_json
            image_bytes = base64.b64decode(b64)

            filename = f"{int(time.time())}_{uuid.uuid4().hex}.png"
            out_path = self.generated_dir / filename
            out_path.write_bytes(image_bytes)

            url_path = f"/static/generated/{filename}"
            yield f"\n✅ Image generated: {url_path}\n\n"
            yield f"![generated image]({url_path})\n"
            return

        except Exception as e:
            # Common: 403 org not verified or model access blocked
            msg = str(e)
            yield "\n❌ Image generation is NOT available for this API key/org right now.\n"
            yield "Fix: OpenAI Platform → Settings → Organization → General → Verify Organization.\n"
            yield f"\nDetails: {msg}\n"
            return
