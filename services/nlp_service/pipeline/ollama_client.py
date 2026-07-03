"""
Ollama Client — OPTIMIZED for speed.
Single LLM call for intent + response. Supports streaming.
Uses Llama 3.2 3B for faster inference on CPU.

Automatic cloud fallback: if the local Ollama server is unreachable
(not installed, not running, model not pulled, etc.), every call
transparently falls back to the Groq API (free tier, also serves
Llama models) so the chatbot keeps working without any code changes
at the call sites.
"""
import os
import ollama
import json
import re

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False

SYSTEM_PROMPT = """You are an intelligent multilingual chatbot for Indian languages.
You can converse fluently in: Hindi (hi), Bengali (bn), Marathi (mr), Tamil (ta), Telugu (te), Kannada (kn), and English (en).

RULES:
1. Always respond in the SAME language the user wrote in.
2. Be polite, helpful, and concise.
3. If you don't know something, say so honestly.
4. Provide detailed, helpful, and informative responses.
5. For translation requests, translate accurately."""

# Combined prompt: classify intent AND generate response in ONE call
COMBINED_PROMPT = """User message: "{message}"
Detected language: {language}

First identify the intent from: greeting, farewell, thanks, ask_name, ask_help, language_change, general_knowledge, weather, time_date, translation_request, joke, news, math_calculation, dictionary, sentiment, complaint, feedback, small_talk, ask_creator, fallback

Respond with EXACTLY this format (2 lines only):
INTENT: <intent_name>
RESPONSE: <your natural response in {language}>"""


class OllamaClient:
    """Ollama wrapper with automatic Groq cloud fallback — single call for classify+respond."""

    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model = model_name
        self.ollama_available = False  # Track whether Ollama is reachable
        self._verify_model()

        # --- Cloud fallback setup (used only if Ollama is unreachable) ---
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.groq_client = None
        self.last_backend = "groq" if not self.ollama_available else "ollama"

        if self.groq_api_key:
            if GROQ_SDK_AVAILABLE:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print(f"✅ Groq fallback ready (model: {self.groq_model})")
            else:
                print("⚠️ GROQ_API_KEY is set but the 'groq' package isn't installed. Run: pip install groq")
        else:
            print("ℹ️ No GROQ_API_KEY set — cloud fallback disabled (Ollama-only mode).")

        # Warn if NEITHER backend is available
        if not self.ollama_available and not self.groq_client:
            print("❌ WARNING: No LLM backend available! Neither Ollama nor Groq is configured.")

    def _verify_model(self):
        """Check if the Ollama model is available. Never raises — sets self.ollama_available instead."""
        try:
            ollama.show(self.model)
            self.ollama_available = True
            print(f"✅ Ollama model loaded: {self.model}")
        except Exception as e:
            print(f"⚠️ Model '{self.model}' not available: {e}")
            # Fallback to default tag
            try:
                self.model = "llama3.2"
                ollama.show(self.model)
                self.ollama_available = True
                print(f"✅ Fallback model: {self.model}")
            except Exception:
                self.ollama_available = False
                print("❌ No local Ollama model available (will rely on Groq fallback if configured).")

    # ------------------------------------------------------------------
    # Unified chat helpers — every method below goes through these two,
    # so Ollama vs. Groq is decided in exactly one place.
    # ------------------------------------------------------------------

    def _chat(self, messages: list, options: dict) -> dict:
        """Try Ollama first; auto-fallback to Groq if Ollama is unreachable."""
        # Skip Ollama entirely if it's known to be offline
        if self.ollama_available:
            try:
                response = ollama.chat(model=self.model, messages=messages, options=options)
                self.last_backend = "ollama"
                return response
            except Exception as e:
                self.ollama_available = False  # Mark as offline for future calls
                if not self.groq_client:
                    raise
                print(f"⚠️ Ollama unreachable ({e}). Falling back to Groq API...")

        # Groq fallback
        if not self.groq_client:
            raise ConnectionError("No LLM backend available: Ollama is offline and Groq is not configured.")
        self.last_backend = "groq"
        return self._groq_chat(messages, options)

    def _stream_chat(self, messages: list, options: dict):
        """Try Ollama streaming first; auto-fallback to Groq streaming if unreachable."""
        # Skip Ollama entirely if it's known to be offline
        if self.ollama_available:
            try:
                stream = ollama.chat(model=self.model, messages=messages, stream=True, options=options)
                first_chunk = next(stream)  # force early failure if Ollama is down
                self.last_backend = "ollama"

                def _combined():
                    yield first_chunk
                    yield from stream
                return _combined()
            except Exception as e:
                self.ollama_available = False  # Mark as offline for future calls
                if not self.groq_client:
                    raise
                print(f"⚠️ Ollama unreachable ({e}). Falling back to Groq API (streaming)...")

        # Groq fallback
        if not self.groq_client:
            raise ConnectionError("No LLM backend available: Ollama is offline and Groq is not configured.")
        self.last_backend = "groq"
        return self._groq_stream(messages, options)

    def _groq_chat(self, messages: list, options: dict) -> dict:
        """Call Groq's OpenAI-compatible chat API and normalize the response
        to look exactly like an ollama.chat() response, so call sites don't change."""
        resp = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=options.get("temperature", 0.6),
            max_tokens=options.get("num_predict", 1024),
            top_p=options.get("top_p", 0.9),
        )
        return {"message": {"content": resp.choices[0].message.content}}

    def _groq_stream(self, messages: list, options: dict):
        """Stream from Groq, yielding chunks shaped like ollama's stream chunks."""
        stream = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=options.get("temperature", 0.6),
            max_tokens=options.get("num_predict", 1024),
            top_p=options.get("top_p", 0.9),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield {"message": {"content": delta}}

    def classify_and_respond(self, message: str, language: str,
                              history: list = None) -> dict:
        """
        SINGLE call: classify intent + generate response together.
        Returns: {"intent": str, "confidence": float, "response": str}
        """
        lang_names = {
            "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
            "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "en": "English"
        }
        lang_name = lang_names.get(language, "English")

        prompt = COMBINED_PROMPT.format(
            message=message,
            language=lang_name
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add last 3 turns for context (minimal to keep it fast)
        if history:
            for turn in history[-3:]:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("text", "")
                })

        messages.append({"role": "user", "content": prompt})

        try:
            response = self._chat(
                messages,
                options={
                    "temperature": 0.5,
                    "num_predict": 1024,    # Allow for very detailed responses
                    "num_ctx": 2048,        # Large context window to prevent cutoffs
                    "repeat_penalty": 1.1,
                    "top_k": 10,
                    "top_p": 0.7,
                }
            )
            raw = response["message"]["content"].strip()
            return self._parse_combined_response(raw)

        except Exception as e:
            print(f"LLM error (Ollama + Groq both unavailable): {e}")
            return {
                "intent": "fallback",
                "confidence": 0.0,
                "response": "Sorry, I encountered an error. Please try again."
            }

    def _parse_combined_response(self, raw: str) -> dict:
        """Parse the combined INTENT + RESPONSE output."""
        intent = "fallback"
        response_text = raw
        confidence = 0.7

        lines = raw.strip().split('\n')

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Extract intent
            if line_stripped.upper().startswith('INTENT:'):
                intent_val = line_stripped.split(':', 1)[1].strip().lower()
                # Clean up any extra text
                intent_val = intent_val.split()[0] if intent_val else "fallback"
                # Remove any punctuation
                intent_val = re.sub(r'[^a-z_]', '', intent_val)
                if intent_val:
                    intent = intent_val
                    confidence = 0.85

            # Extract response
            elif line_stripped.upper().startswith('RESPONSE:'):
                response_text = line_stripped.split(':', 1)[1].strip()
                # Include remaining lines too
                remaining = '\n'.join(l.strip() for l in lines[i+1:] if l.strip()
                                      and not l.strip().upper().startswith('INTENT:'))
                if remaining:
                    response_text += '\n' + remaining
                break

        # If we couldn't parse, use the whole raw text as response
        if response_text == raw and 'INTENT:' in raw:
            # Remove the INTENT line from response
            response_text = '\n'.join(
                l for l in lines
                if not l.strip().upper().startswith('INTENT:')
                and not l.strip().upper().startswith('RESPONSE:')
            ).strip()

        if not response_text:
            response_text = raw

        return {
            "intent": intent,
            "confidence": confidence,
            "response": response_text
        }

    def generate_response(self, message: str, intent: str, language: str,
                          history: list = None) -> str:
        """Generate a standalone response (for dynamic intents)."""
        lang_names = {
            "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
            "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "en": "English"
        }
        lang_name = lang_names.get(language, "English")

        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nRespond in {lang_name}. Provide a detailed and informative answer."},
            {"role": "user", "content": message}
        ]

        try:
            response = self._chat(
                messages,
                options={
                    "temperature": 0.6,
                    "num_predict": 1024,
                    "num_ctx": 2048,
                    "top_k": 20,
                }
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, message: str, language: str,
                        history: list = None):
        """
        Stream response tokens for real-time display.
        Includes conversation history for multi-turn context.
        Yields chunks of text.
        """
        lang_names = {
            "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
            "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "en": "English"
        }
        lang_name = lang_names.get(language, "English")

        messages = [
            {"role": "system", "content": f"Reply in {lang_name}. Provide a detailed and helpful response."}
        ]

        # Add conversation history for multi-turn context
        if history:
            for turn in history[-6:]:  # Last 3 exchanges
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("text", "")
                })

        messages.append({"role": "user", "content": message})

        try:
            stream = self._stream_chat(
                messages,
                options={
                    "temperature": 0.6,
                    "num_predict": 1024,
                    "num_ctx": 2048,
                    "top_k": 10,
                }
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield token
        except Exception as e:
            yield f"Error: {e}"

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the active LLM backend."""
        lang_names = {
            "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
            "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "en": "English"
        }
        src = lang_names.get(source_lang, source_lang)
        tgt = lang_names.get(target_lang, target_lang)

        try:
            response = self._chat(
                [
                    {"role": "system", "content": f"Translate from {src} to {tgt}. Return ONLY the translation."},
                    {"role": "user", "content": text}
                ],
                options={"temperature": 0.1, "num_predict": 100, "num_ctx": 512}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Translation error: {e}"

    def summarize_context(self, history: list) -> str:
        """Summarize conversation history into a concise context string."""
        if not history:
            return ""

        text_to_summarize = ""
        for turn in history:
            role = turn.get("role", "unknown")
            text = turn.get("text", "")
            text_to_summarize += f"{role.capitalize()}: {text}\n"

        messages = [
            {"role": "system", "content": "You are a highly efficient memory summarizer. Summarize the following conversation in 2-3 concise sentences. Focus on the main topics discussed and any important facts the user shared. Do not include pleasantries."},
            {"role": "user", "content": text_to_summarize}
        ]

        try:
            response = self._chat(
                messages,
                options={"temperature": 0.3, "num_predict": 150, "num_ctx": 2048}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"Summarization error: {e}")
            return "Previous conversation context retained."

    def health_check(self) -> bool:
        """Check if either backend (Ollama or Groq fallback) is available."""
        # Re-check Ollama availability (it may have come back online)
        if not self.ollama_available:
            try:
                ollama.show(self.model)
                self.ollama_available = True
            except Exception:
                pass
        return self.ollama_available or (self.groq_client is not None)

    def active_backend(self) -> str:
        """Which backend is currently reachable: 'ollama', 'groq', or 'none'."""
        # Re-check Ollama availability (it may have come back online)
        if not self.ollama_available:
            try:
                ollama.show(self.model)
                self.ollama_available = True
            except Exception:
                pass
        if self.ollama_available:
            return "ollama"
        return "groq" if self.groq_client is not None else "none"
