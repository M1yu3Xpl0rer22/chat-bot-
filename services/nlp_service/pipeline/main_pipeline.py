"""
Main NLP Pipeline — Orchestrates the complete message processing flow.
Language Detection (FastText) → Intent Classification (Ollama) → Response Generation (KB + Ollama)
"""
import os
import json
import random
from datetime import datetime

from services.nlp_service.pipeline.ollama_client import OllamaClient
from services.nlp_service.translator import TranslationService

try:
    from services.nlp_service.rag_service import RAGService, RAG_AVAILABLE
except ImportError:
    RAG_AVAILABLE = False

# Try to load FastText language detector, fall back to Ollama-based detection
try:
    from services.nlp_service.pipeline.language_detector import LanguageDetector
    _fasttext_available = True
except Exception:
    _fasttext_available = False
    print("FastText not available. Language detection will use Ollama.")


class KnowledgeBase:
    """Loads and manages the intent-response knowledge base."""

    def __init__(self, kb_path: str = None):
        if kb_path is None:
            kb_path = os.path.join(
                os.path.dirname(__file__),
                '../../../data/knowledge_base/entries.json'
            )
        self.kb_path = os.path.abspath(kb_path)
        self.intents = {}
        self._load()

    def _load(self):
        """Load knowledge base from JSON file."""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for entry in data.get("intents", []):
                intent_name = entry["intent"]
                self.intents[intent_name] = entry
            print(f"✅ Knowledge Base loaded: {len(self.intents)} intents")
        except FileNotFoundError:
            print(f"⚠️ KB file not found at {self.kb_path}")
        except Exception as e:
            print(f"⚠️ KB load error: {e}")

    def get_response(self, intent: str, language: str) -> str | None:
        """Get a curated KB response for a given intent and language."""
        entry = self.intents.get(intent)
        if not entry:
            return None

        responses = entry.get("responses", {})
        lang_responses = responses.get(language, responses.get("en", []))

        if isinstance(lang_responses, list) and lang_responses:
            return random.choice(lang_responses)
        elif isinstance(lang_responses, str):
            return lang_responses
        return None

    def is_dynamic(self, intent: str) -> bool:
        """Check if an intent requires dynamic (Ollama) response generation."""
        entry = self.intents.get(intent, {})
        return entry.get("dynamic", False)

    def get_all_intents(self) -> list:
        """Get list of all intent names."""
        return list(self.intents.keys())

    def get_intent_details(self, intent: str) -> dict | None:
        """Get full details for an intent."""
        return self.intents.get(intent)

    def add_intent(self, intent_data: dict):
        """Add or update an intent in the KB."""
        intent_name = intent_data.get("intent")
        if intent_name:
            self.intents[intent_name] = intent_data
            self._save()

    def delete_intent(self, intent_name: str) -> bool:
        """Delete an intent from the KB."""
        if intent_name in self.intents:
            del self.intents[intent_name]
            self._save()
            return True
        return False

    def _save(self):
        """Save KB back to JSON file."""
        data = {"domain": "General Multilingual Assistant", "intents": list(self.intents.values())}
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class NLPPipeline:
    """Complete NLP processing pipeline for the multilingual chatbot."""

    SUPPORTED_LANGS = {"hi", "bn", "mr", "ta", "te", "kn", "en"}

    def __init__(self, model_name: str = "llama3.2:3b"):
        print("🚀 Initializing NLP Pipeline...")

        # Initialize components
        self.ollama = OllamaClient(model_name)
        self.kb = KnowledgeBase()
        self.translator = TranslationService(self.ollama)

        # RAG Service
        if RAG_AVAILABLE:
            self.rag = RAGService(model_name)
        else:
            self.rag = None

        # FastText for language detection (optional)
        self.lang_detector = None
        if _fasttext_available:
            try:
                self.lang_detector = LanguageDetector()
                print("✅ FastText language detector loaded")
            except Exception as e:
                print(f"⚠️ FastText detector failed: {e}. Using Ollama for detection.")

        # Session store (in-memory for now, Redis later)
        self.sessions = {}

        print("✅ NLP Pipeline ready!")

    def detect_language(self, text: str) -> tuple[str, float]:
        """Detect the language of input text."""
        if self.lang_detector:
            lang, conf = self.lang_detector.detect_language(text)
            # Map fasttext codes to our supported set
            if lang in self.SUPPORTED_LANGS:
                return lang, conf
            # If detected lang is not in our supported set, default to English
            return "en", conf
        else:
            # Fallback: simple script-based detection
            return self._script_based_detection(text)

    def _script_based_detection(self, text: str) -> tuple[str, float]:
        """Detect language based on Unicode script ranges."""
        for char in text:
            cp = ord(char)
            # Devanagari (Hindi/Marathi)
            if 0x0900 <= cp <= 0x097F:
                # Try to distinguish Hindi vs Marathi (basic heuristic)
                return "hi", 0.8
            # Bengali
            elif 0x0980 <= cp <= 0x09FF:
                return "bn", 0.9
            # Tamil
            elif 0x0B80 <= cp <= 0x0BFF:
                return "ta", 0.9
            # Telugu
            elif 0x0C00 <= cp <= 0x0C7F:
                return "te", 0.9
            # Kannada
            elif 0x0C80 <= cp <= 0x0CFF:
                return "kn", 0.9
        # Default to English if no Indic script found
        return "en", 0.7

    def process(self, message: str, session_id: str = None) -> dict:
        """
        OPTIMIZED: Process user message with minimum Ollama calls.
        Static intents → instant KB response (0ms Ollama).
        Dynamic intents → single combined Ollama call.
        """
        if not message or not message.strip():
            return {
                "response": "Please type a message.",
                "intent": "fallback",
                "confidence": 0.0,
                "language": "en",
                "session_id": session_id
            }

        # Step 1: Detect language (FastText — instant, no Ollama)
        language, lang_confidence = self.detect_language(message)

        # Step 2: Try quick pattern matching for common intents BEFORE Ollama
        quick_result = self._quick_intent_match(message, language)
        if quick_result:
            intent = quick_result["intent"]
            response = quick_result["response"]
            confidence = 0.95
        else:
            # Step 2.5: RAG Context
            if session_id and self.rag and self.rag.has_document(session_id):
                context = self.rag.search(message, session_id)
                if context:
                    message = f"Document Context:\n{context}\n\nUser Question:\n{message}"

            # Step 3: Use SINGLE Ollama call (classify + respond together)
            history = self._get_session_history(session_id) if session_id else None
            result = self.ollama.classify_and_respond(message, language, history)
            intent = result.get("intent", "fallback")
            confidence = result.get("confidence", 0.5)
            response = result.get("response", "")

            # For recognized static intents, prefer KB response (more natural)
            if not self.kb.is_dynamic(intent):
                kb_response = self.kb.get_response(intent, language)
                if kb_response:
                    response = kb_response

            # Handle special intents
            if intent == "translation_request":
                response = self._handle_translation(message, language)
            elif intent == "time_date":
                response = self._handle_time_date(language)

        # Step 4: Update session history
        if session_id:
            self._update_session(session_id, message, response, intent, language)

        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "language": language,
            "language_confidence": lang_confidence,
            "entities": {},
            "session_id": session_id
        }

    def _quick_intent_match(self, message: str, language: str) -> dict | None:
        """
        FAST pattern matching for common intents — no Ollama call needed.
        Returns instantly for greetings, farewells, thanks, etc.
        """
        msg_lower = message.strip().lower().strip(" !.,?।")

        # Greeting patterns
        greetings = {"hello", "hi", "hey", "good morning", "good evening", "good afternoon",
                     "नमस्ते", "हैलो", "हाय", "प्रणाम", "শুভ", "নমস্কার", "হ্যালো",
                     "नमस्कार", "हॅलो", "வணக்கம்", "ஹலோ", "నమస్కారం", "హలో",
                     "ನಮಸ್ಕಾರ", "ಹಲೋ"}
        if msg_lower in greetings:
            return {"intent": "greeting", "response": self.kb.get_response("greeting", language)}

        # Farewell patterns
        farewells = {"bye", "goodbye", "see you", "take care",
                     "अलविदा", "बाय", "फिर मिलेंगे", "বিদায়", "বাই",
                     "निरोप", "பிரியாவிடை", "వీడ్కోలు", "ವಿದಾಯ"}
        if msg_lower in farewells:
            return {"intent": "farewell", "response": self.kb.get_response("farewell", language)}

        # Thanks patterns
        thanks = {"thank you", "thanks", "thanks a lot", "appreciate it",
                  "धन्यवाद", "शुक्रिया", "थैंक्स", "ধন্যবাদ", "থ্যাঙ্কস",
                  "आभार", "நன்றி", "ధన్యవాదాలు", "ಧನ್ಯವಾದ"}
        if msg_lower in thanks:
            return {"intent": "thanks", "response": self.kb.get_response("thanks", language)}

        # "Who are you" / "What's your name" patterns
        name_phrases = {"what is your name", "who are you", "your name",
                        "तुम्हारा नाम क्या है", "तुम कौन हो", "तुझे नाव काय",
                        "তোমার নাম কি", "তুমি কে"}
        if msg_lower in name_phrases:
            return {"intent": "ask_name", "response": self.kb.get_response("ask_name", language)}

        # "Who made you" patterns
        creator_phrases = {"who made you", "who created you", "who built you",
                           "तुम्हें किसने बनाया", "তোমাকে কে বানিয়েছে", "तुला कोणी बनवले"}
        if msg_lower in creator_phrases:
            return {"intent": "ask_creator", "response": self.kb.get_response("ask_creator", language)}

        return None  # No quick match → use Ollama

    def _handle_translation(self, message: str, source_lang: str) -> str:
        """Handle translation request."""
        target_lang = self.translator.detect_target_language(message)
        if target_lang == source_lang:
            target_lang = "en" if source_lang != "en" else "hi"

        # Try to extract the text to translate from the message
        result = self.translator.translate(message, source_lang, target_lang)
        lang_names = TranslationService.get_supported_languages()
        target_name = lang_names.get(target_lang, target_lang)
        return f"🔄 {target_name}: {result}"

    def _handle_time_date(self, language: str) -> str:
        """Handle time/date request."""
        now = datetime.now()
        date_str = now.strftime("%d %B %Y")
        time_str = now.strftime("%I:%M %p")
        day_str = now.strftime("%A")

        responses = {
            "en": f"📅 Today is {day_str}, {date_str}\n🕐 Current time: {time_str}",
            "hi": f"📅 आज {day_str}, {date_str} है\n🕐 वर्तमान समय: {time_str}",
            "bn": f"📅 আজ {day_str}, {date_str}\n🕐 বর্তমান সময়: {time_str}",
            "mr": f"📅 आज {day_str}, {date_str} आहे\n🕐 सध्याची वेळ: {time_str}",
            "ta": f"📅 இன்று {day_str}, {date_str}\n🕐 தற்போதைய நேரம்: {time_str}",
            "te": f"📅 ఈరోజు {day_str}, {date_str}\n🕐 ప్రస్తుత సమయం: {time_str}",
            "kn": f"📅 ಇಂದು {day_str}, {date_str}\n🕐 ಪ್ರಸ್ತುತ ಸಮಯ: {time_str}",
        }
        return responses.get(language, responses["en"])

    # --- Session Management (in-memory, can swap to Redis) ---

    def _get_session_history(self, session_id: str) -> list | None:
        """Get conversation history for a session."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id].get("history", [])
        return None

    def _update_session(self, session_id: str, user_msg: str, bot_msg: str,
                        intent: str, language: str):
        """Update session with new conversation turn."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "language": language,
                "created_at": datetime.now().isoformat()
            }

        session = self.sessions[session_id]
        session["history"].append({"role": "user", "text": user_msg})
        session["history"].append({"role": "assistant", "text": bot_msg})
        session["language"] = language
        session["last_intent"] = intent

        # Smart Context Compression: Summarize if > 8 turns (4 exchanges)
        if len(session["history"]) > 8:
            # Get the oldest 6 turns to summarize
            turns_to_summarize = session["history"][:6]
            recent_turns = session["history"][6:]
            
            # Summarize the oldest turns using Ollama
            summary = self.ollama.summarize_context(turns_to_summarize)
            
            # Rebuild history with the summary + recent turns
            session["history"] = [
                {"role": "system", "text": f"Summary of previous context: {summary}"}
            ] + recent_turns

    def get_session_info(self, session_id: str) -> dict | None:
        """Get session information."""
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str):
        """Clear a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_all_sessions(self) -> dict:
        """Get all active sessions (for admin)."""
        return {
            sid: {
                "language": s.get("language"),
                "turns": len(s.get("history", [])),
                "created_at": s.get("created_at"),
                "last_intent": s.get("last_intent")
            }
            for sid, s in self.sessions.items()
        }
