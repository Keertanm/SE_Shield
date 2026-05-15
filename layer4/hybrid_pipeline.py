"""
layer4/hybrid_pipeline.py
===========================
Full hybrid SE detection pipeline.

Wires all layers in the final confirmed architecture:

  Raw text
    → Layer 1 : adversarial normalisation
    → Layer 2 : SVM (binary gate) + LR (risk_score)
    → Layer 4a: Risk Counter (conversation context, SVM override)
    → Layer 3 : NLI sub-type classification
    → Layer 4b: Semantic Window (pattern detection, entity risk)
    → Output  : dashboard-ready conversation assessment

Usage
-----
  from layer4.hybrid_pipeline import HybridPipeline

  pipe = HybridPipeline()

  # Process one message at a time (streaming, real-time)
  result = pipe.process(
      text            = "Urgent: verify your account now.",
      conversation_id = "conv_42",
      message_id      = "conv_42_msg_003",
      timestamp       = "2026-05-08T09:15:00Z",
  )
  print(result["alert_level"])   # "HIGH"
  print(result["attack_pattern"]) # "authority_then_credential"

  # Process a full conversation at once (batch / test mode)
  results = pipe.process_conversation(conversation)

Phase 2 changes
---------------
Only change: pass real conversation_id from your data source.
  email → thread_id or hash(sender+recipient)
  chat  → session_id or room_id
  SMS   → hash(from_number + to_number)

All layer internals unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent        # layer4/
_PROJECT_ROOT = _HERE.parent                           # hybrid_se/
_LAYER3_DIR   = _PROJECT_ROOT / "layer3_slm"

# Force layer3_slm to sys.path[0] regardless of existing state.
# If layer3_slm is already in sys.path (inserted by test runner or caller),
# it may be at a position AFTER hybrid_se/src, causing `from src.layer3_pipeline`
# to resolve against hybrid_se/src/ instead of layer3_slm/src/.
# Remove and re-insert at 0 to guarantee correct resolution order.
_l3_str = str(_LAYER3_DIR)
if _l3_str in sys.path:
    sys.path.remove(_l3_str)
sys.path.insert(0, _l3_str)

# Project root goes in second position (needed for layer4 imports).
_pr_str = str(_PROJECT_ROOT)
if _pr_str not in sys.path:
    sys.path.insert(1, _pr_str)

from layer4.layer4a_risk_counter    import RiskCounter
from layer4.layer4b_semantic_window import SemanticWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal Layer 1 normaliser (reuse from integrate_layers if available)
# ---------------------------------------------------------------------------

class _Layer1:
    _SUBS = str.maketrans({
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
        "7": "t", "@": "a", "$": "s",
        "\u0430": "a", "\u0435": "e", "\u043e": "o",
    })

    def normalise(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"%[0-9A-Fa-f]{2}",
                      lambda m: bytes.fromhex(m.group(0)[1:]).decode("latin-1", errors="replace"),
                      text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.translate(self._SUBS)


# ---------------------------------------------------------------------------
# Hybrid Pipeline
# ---------------------------------------------------------------------------

class HybridPipeline:
    """
    Full hybrid SE detection pipeline (L1 → L2 → L4a → L3 → L4b).

    Parameters
    ----------
    tfidf_path / svm_path / lr_path : Layer 2 model paths.
    layer3_model_name               : HuggingFace model ID.
    window_size                     : Sliding window depth (default 10).
    suspicious_threshold            : L4a threshold for SVM override (default 3.0).
    device                          : "cpu" | "cuda" | None (auto).
    """

    def __init__(
        self,
        tfidf_path:         str | Path | None = None,
        svm_path:           str | Path | None = None,
        lr_path:            str | Path | None = None,
        layer3_model_name:  str = "cross-encoder/nli-deberta-v3-small",
        window_size:        int = 10,
        suspicious_threshold: float = 3.0,
        device:             Optional[str] = None,
    ) -> None:
        import joblib
        # Re-assert layer3_slm at sys.path[0] immediately before local imports.
        # Defends against external code (e.g. server identity import block)
        # inserting paths that push layer3_slm away from position 0 between
        # module load time and __init__ call time.
        _l3 = str(_LAYER3_DIR)
        if _l3 in sys.path:
            sys.path.remove(_l3)
        sys.path.insert(0, _l3)

        from config_layer3 import (
            ATTACK_LABELS, HYPOTHESIS_TEMPLATES,
            LAYER2_THRESHOLD, MAX_LENGTH,
        )
        from src.layer3_pipeline import Layer3Pipeline

        _models = _PROJECT_ROOT / "models"
        tfidf_path = tfidf_path or _models / "tfidf_vectorizer.pkl"
        svm_path   = svm_path   or _models / "stage1a_svm_final.pkl"
        lr_path    = lr_path    or _models / "stage1b_lr_final.pkl"

        logger.info("Loading Layer 2 models …")
        self._tfidf = joblib.load(tfidf_path)
        self._svm   = joblib.load(svm_path)
        self._lr    = joblib.load(lr_path)
        logger.info("Layer 2 ready.")

        self._l1 = _Layer1()

        self._l3 = Layer3Pipeline(
            model_name           = layer3_model_name,
            labels               = ATTACK_LABELS,
            hypothesis_templates = HYPOTHESIS_TEMPLATES,
            layer2_threshold     = LAYER2_THRESHOLD,
            max_length           = MAX_LENGTH,
            device               = device,
        )

        self._l4a = RiskCounter(
            window_size          = window_size,
            suspicious_threshold = suspicious_threshold,
        )
        self._l4b = SemanticWindow(window_size=window_size)

        logger.info("HybridPipeline ready.")

    # ------------------------------------------------------------------ #
    # Primary interface                                                     #
    # ------------------------------------------------------------------ #

    def process(
        self,
        text:            str,
        conversation_id: str,
        message_id:      str | None = None,
        timestamp:       str | None = None,
    ) -> dict:
        """
        Process one message through the full pipeline.

        Returns the Layer 4b dashboard output contract, extended with
        the per-message Layer 3 result under key "last_message".

        Parameters
        ----------
        text            : Raw message text.
        conversation_id : Unique conversation ID.
                          Phase 1: synthetic ("conv_42")
                          Phase 2: thread_id / session_id / hash(from+to)
        message_id      : Optional unique message ID.
        timestamp       : Optional ISO-8601 timestamp.
        """
        # ── L1: normalise ────────────────────────────────────────────────
        clean = self._l1.normalise(text)

        # ── L2: SVM + LR ─────────────────────────────────────────────────
        vec        = self._tfidf.transform([clean])
        svm_pred   = int(self._svm.predict(vec)[0])
        svm_label  = "suspicious" if svm_pred == 1 else "benign"
        risk_score = int(self._lr.predict_proba(vec)[0][1] * 100)

        # ── L4a: Risk Counter ────────────────────────────────────────────
        ctx = self._l4a.update(
            conversation_id = conversation_id,
            svm_label       = svm_label,
            risk_score      = risk_score,
        )

        # Apply SVM override: if conversation is suspicious but this
        # message looked benign, force Layer 3 to run NLI anyway.
        effective_svm = "suspicious" if ctx["override_svm"] else svm_label

        # Apply dynamic confidence threshold from L4a context.
        self._l3.min_subtype_confidence = ctx["recommended_min_conf"]

        # ── L3: NLI classification ───────────────────────────────────────
        l3_result = self._l3.run(
            text              = clean,
            layer2_risk_score = risk_score,
            layer2_label      = effective_svm,
            message_id        = message_id,
            timestamp         = timestamp,
        )

        # ── L4b: Semantic Window ─────────────────────────────────────────
        assessment = self._l4b.update(conversation_id, l3_result)

        # Combine: return conversation assessment + per-message detail
        return {
            **assessment,
            "last_message": l3_result,
            "l4a_context":  ctx,
        }

    def process_conversation(self, messages: list[dict]) -> list[dict]:
        """
        Process a full conversation sequentially.

        Each message dict:
          text            : str  (required)
          conversation_id : str  (required)
          message_id      : str  (optional)
          timestamp       : str  (optional)

        Returns list of per-message pipeline outputs (cumulative assessment
        grows with each message).

        Example
        -------
        conv = [
            {"text": "Hi I'm from IT",            "conversation_id": "conv_42",
             "message_id": "m1", "timestamp": "2026-05-08T09:00:00Z"},
            {"text": "Need your login to fix it",  "conversation_id": "conv_42",
             "message_id": "m2", "timestamp": "2026-05-08T09:05:00Z"},
        ]
        results = pipe.process_conversation(conv)
        print(results[-1]["alert_level"])   # "HIGH"
        """
        results = []
        for msg in messages:
            result = self.process(
                text            = msg["text"],
                conversation_id = msg["conversation_id"],
                message_id      = msg.get("message_id"),
                timestamp       = msg.get("timestamp"),
            )
            results.append(result)
        return results

    def reset_conversation(self, conversation_id: str) -> None:
        """Reset both L4a and L4b state for a resolved conversation."""
        self._l4a.reset(conversation_id)
        self._l4b.reset(conversation_id)

    def active_conversations(self) -> dict:
        """Return summary of all active conversation windows."""
        return {
            "layer4a": self._l4a.list_active(),
            "layer4b": self._l4b.list_active(),
        }