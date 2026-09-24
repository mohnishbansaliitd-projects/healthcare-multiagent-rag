"""
Clinical RAG agent: retrieves relevant protocols via embedding similarity (Ollama bge-m3)
and generates the answer with an LLM (Ollama granite3.1-dense:8b) grounded on that context,
plus a physician disclaimer. Falls back to keyword matching if Ollama isn't reachable.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover - exercised only when requests is missing
    requests = None

import numpy as np


class ClinicalRAGAgent:
    """KB chunks are embedded once at construction and cached to disk; queries are embedded
    the same way and matched by cosine similarity."""

    EMBED_MODEL = "bge-m3"
    GEN_MODEL = "granite3.1-dense:8b"
    OLLAMA_HOST = "http://localhost:11434"
    EMBED_TIMEOUT_S = 30
    GENERATE_TIMEOUT_S = 90
    # Cosine-similarity floor below which we treat the KB as "not covering"
    # the query -- the embedding-similarity analogue of the old keyword
    # matcher's `score >= 2` gate. Calibrated against this 3-disease KB:
    # on-topic queries score ~0.46-0.64, off-topic queries score ~0.35-0.40.
    SIMILARITY_THRESHOLD = 0.45
    TOP_K = 2

    CACHE_NPY_NAME = "kb_embeddings.npy"
    CACHE_META_NAME = "kb_embeddings_meta.json"

    SYSTEM_INSTRUCTIONS = (
        "You are a clinical information assistant embedded in a healthcare messaging system. "
        "Answer the patient's question using ONLY the verified clinical context provided below. "
        "Do not use outside medical knowledge and do not invent facts, dosages, or numbers that are "
        "not present in the context. If the retrieved context does not fully or precisely answer the "
        "question, say so plainly and hedge rather than guessing. Keep the answer concise (a few "
        "sentences or a short bulleted list), clinically appropriate, and easy for a patient to read. "
        "Never state or imply a diagnosis. Always make clear this is general educational information, "
        "not a substitute for professional medical advice, and that the patient should consult their "
        "physician or seek emergency care for anything urgent."
    )

    NO_MATCH_RESPONSE = (
        "I do not have specific verified clinical guidelines for that condition in my protocol database. "
        "Please consult your healthcare provider directly for personalized advice."
    )
    NO_MATCH_DISCLAIMER = (
        "Disclaimer: This AI assistant provides general educational information only and does not "
        "replace medical advice."
    )
    ANSWER_DISCLAIMER = (
        "Disclaimer: This AI assistant provides general educational information only. Always follow "
        "your physician's specific instructions."
    )

    def __init__(
        self,
        kb_path: Optional[str] = None,
        ollama_host: Optional[str] = None,
        embed_model: Optional[str] = None,
        gen_model: Optional[str] = None,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ):
        if kb_path is None:
            kb_path = os.path.join(os.path.dirname(__file__), "..", "data", "clinical_knowledge_base.json")
        self.kb_path = kb_path
        with open(kb_path, "r") as f:
            self.knowledge_base = json.load(f)

        self.ollama_host = ollama_host or self.OLLAMA_HOST
        self.embed_model = embed_model or self.EMBED_MODEL
        self.gen_model = gen_model or self.GEN_MODEL
        self.top_k = top_k or self.TOP_K
        self.use_cache = use_cache

        self.chunks: List[Dict[str, Any]] = [
            {"disease": d, "text": self._build_chunk_text(d)} for d in self.knowledge_base["diseases"]
        ]

        self.ollama_available = requests is not None
        self.kb_embeddings: Optional[np.ndarray] = None
        self._fallback_notice_printed = False

        if not self.ollama_available:
            self._print_fallback_notice("the 'requests' package is not installed")
            return

        cached = self._load_cache() if self.use_cache else None
        if cached is not None:
            self.kb_embeddings = cached
            return

        try:
            embeddings = [self._embed_text(c["text"]) for c in self.chunks]
            self.kb_embeddings = np.vstack(embeddings)
            if self.use_cache:
                self._save_cache(self.kb_embeddings)
        except Exception as e:  # noqa: BLE001 - any network/parse failure triggers fallback
            self.ollama_available = False
            self.kb_embeddings = None
            self._print_fallback_notice(f"could not embed knowledge base at startup ({e})")

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chunk_text(entry: Dict[str, Any]) -> str:
        """Builds a text chunk from a disease entry, generically covering whatever fields exist."""
        lines = [f"Disease: {entry.get('name', 'Unknown')}"]
        for key, value in entry.items():
            if key in ("id", "name"):
                continue
            label = key.replace("_", " ").capitalize()
            if isinstance(value, list):
                lines.append(f"{label}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def _cache_paths(self) -> Tuple[str, str]:
        data_dir = os.path.dirname(os.path.abspath(self.kb_path))
        return (
            os.path.join(data_dir, self.CACHE_NPY_NAME),
            os.path.join(data_dir, self.CACHE_META_NAME),
        )

    def _kb_hash(self) -> str:
        h = hashlib.md5()
        h.update(self.embed_model.encode("utf-8"))
        for c in self.chunks:
            h.update(c["text"].encode("utf-8"))
        return h.hexdigest()

    def _load_cache(self) -> Optional[np.ndarray]:
        npy_path, meta_path = self._cache_paths()
        if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
            return None
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("hash") != self._kb_hash():
                return None
            arr = np.load(npy_path)
            if arr.shape[0] != len(self.chunks):
                return None
            return arr
        except Exception:
            return None

    def _save_cache(self, embeddings: np.ndarray) -> None:
        try:
            npy_path, meta_path = self._cache_paths()
            np.save(npy_path, embeddings)
            with open(meta_path, "w") as f:
                json.dump(
                    {"hash": self._kb_hash(), "model": self.embed_model, "n_chunks": len(self.chunks)},
                    f,
                )
        except Exception as e:  # noqa: BLE001 - caching is best-effort only
            print(f"[ClinicalRAGAgent] Warning: could not persist embedding cache ({e}).")

    # ------------------------------------------------------------------
    # Ollama calls
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> np.ndarray:
        resp = requests.post(
            f"{self.ollama_host}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=self.EMBED_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["embedding"], dtype=np.float32)

    def _generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_host}/api/generate",
            json={"model": self.gen_model, "prompt": prompt, "stream": False},
            timeout=self.GENERATE_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()

    def _print_fallback_notice(self, reason: str) -> None:
        if not self._fallback_notice_printed:
            print(
                f"[ClinicalRAGAgent] FALLBACK TRIGGERED: Ollama unreachable, "
                f"using legacy keyword/token-overlap matching instead of embedding-based RAG. "
                f"Reason: {reason}"
            )
            self._fallback_notice_printed = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_top_k(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Embeds the query and returns the top-k (chunk, cosine_similarity) pairs from the KB."""
        if self.kb_embeddings is None:
            raise RuntimeError("KB embeddings are unavailable")
        top_k = top_k or self.top_k
        q_emb = self._embed_text(query)
        norms = np.linalg.norm(self.kb_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        sims = (self.kb_embeddings @ q_emb) / norms
        order = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in order]

    def retrieve_relevant_protocol(self, query: str) -> Optional[Dict[str, Any]]:
        """Legacy fallback retriever: best matching disease entry via keyword/token overlap."""
        q_lower = query.lower()
        best_match = None
        highest_score = 0

        for d in self.knowledge_base["diseases"]:
            score = 0
            d_name_lower = d["name"].lower()

            if d_name_lower in q_lower:
                score += 10
            else:
                # token overlap so "type 2 diabetes" still matches "type 2 diabetes mellitus"
                name_words = [w for w in d_name_lower.replace("(", "").replace(")", "").split() if len(w) > 2]
                matched_name_words = [w for w in name_words if w in q_lower]
                if matched_name_words:
                    score += len(matched_name_words) * 2

            for s in d.get("symptoms", []):
                if s.lower() in q_lower or any(sw in q_lower for sw in s.lower().split() if len(sw) > 3):
                    score += 1

            if score > highest_score and score >= 2:
                highest_score = score
                best_match = d

        return best_match

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, context_block: str) -> str:
        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            f"Verified clinical context:\n{context_block}\n\n"
            f"Patient question: {query}\n\n"
            f"Answer:"
        )

    def _answer_query_fallback(self, user_query: str) -> Dict[str, Any]:
        """Old keyword/token-overlap answer path, used when Ollama is unreachable."""
        self._print_fallback_notice("query-time fallback")
        protocol = self.retrieve_relevant_protocol(user_query)

        if protocol is None:
            return {
                "matched_disease": None,
                "response_text": self.NO_MATCH_RESPONSE,
                "disclaimer": self.NO_MATCH_DISCLAIMER,
                "retrieval_method": "keyword_overlap_fallback",
            }

        text = (
            f"Regarding {protocol['name']}:\n"
            f"• Common Symptoms: {', '.join(protocol['symptoms'])}\n"
            f"• Recommended First-Line Management: {protocol['first_line_management']}\n"
            f"• Emergency Warning Signs: {', '.join(protocol['emergency_flags'])}\n"
            f"• Follow-Up Guideline: {protocol['follow_up_protocol']}"
        )

        return {
            "matched_disease": protocol["name"],
            "response_text": text,
            "disclaimer": self.ANSWER_DISCLAIMER,
            "retrieval_method": "keyword_overlap_fallback",
        }

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """Generates a grounded clinical response via embedding retrieval + LLM generation."""
        if not self.ollama_available or self.kb_embeddings is None:
            return self._answer_query_fallback(user_query)

        try:
            results = self.retrieve_top_k(user_query, top_k=self.top_k)
        except Exception as e:  # noqa: BLE001 - connection refused / timeout / bad response
            self.ollama_available = False
            self._print_fallback_notice(f"query-time embedding call failed ({e})")
            return self._answer_query_fallback(user_query)

        top_similarity = results[0][1] if results else 0.0

        if not results or top_similarity < self.SIMILARITY_THRESHOLD:
            return {
                "matched_disease": None,
                "response_text": self.NO_MATCH_RESPONSE,
                "disclaimer": self.NO_MATCH_DISCLAIMER,
                "retrieval_method": "embedding_similarity",
                "top_similarity": top_similarity,
            }

        context_block = "\n\n".join(
            f"[Context {i + 1} | similarity={score:.2f}]\n{chunk['text']}"
            for i, (chunk, score) in enumerate(results)
        )
        top_disease = results[0][0]["disease"]["name"]
        prompt = self._build_prompt(user_query, context_block)

        try:
            llm_response = self._generate(prompt)
        except Exception as e:  # noqa: BLE001 - connection refused / timeout / bad response
            self.ollama_available = False
            self._print_fallback_notice(f"query-time generation call failed ({e})")
            return self._answer_query_fallback(user_query)

        if not llm_response:
            llm_response = (
                "I was unable to generate a specific response for this query from the verified "
                "context. Please consult your healthcare provider directly for personalized advice."
            )

        return {
            "matched_disease": top_disease,
            "response_text": llm_response,
            "disclaimer": self.ANSWER_DISCLAIMER,
            "retrieval_method": "embedding_similarity",
            "top_similarity": top_similarity,
        }
