"""LangChain callback handler for promptfirewall.

Scans LLM inputs for PII and prompt injection before they reach the model.
Optionally redacts PII in-place and blocks detected injection attempts.

Usage::

    from langchain_promptfirewall import PromptFirewallHandler

    handler = PromptFirewallHandler()
    llm = ChatOpenAI(callbacks=[handler])
    llm.invoke("Hello, my SSN is 123-45-6789")
    # Raises PromptInjectionError or redacts PII before sending
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks import BaseCallbackHandler

import promptfirewall

logger = logging.getLogger("promptfirewall.langchain")


class PromptInjectionError(ValueError):
    """Raised when a prompt injection attempt is detected and blocking is enabled."""

    def __init__(
        self,
        message: str,
        injection_score: float,
        injection_labels: List[str],
    ) -> None:
        super().__init__(message)
        self.injection_score = injection_score
        self.injection_labels = injection_labels


class PiiDetectedError(ValueError):
    """Raised when PII is detected and blocking mode is enabled."""

    def __init__(
        self,
        message: str,
        pii_findings: List[Dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.pii_findings = pii_findings


@dataclass
class ScanEvent:
    """Record of a single prompt scan."""

    prompt: str
    is_safe: bool
    injection_score: float
    injection_labels: List[str]
    pii_findings: List[Dict[str, Any]]
    redacted_text: Optional[str]
    action_taken: str  # "passed", "redacted", "blocked_injection", "blocked_pii"
    latency_us: int


class PromptFirewallHandler(BaseCallbackHandler):
    """LangChain callback handler that scans prompts with promptfirewall.

    Scans every prompt for PII and prompt injection before it reaches the LLM.
    Runs entirely locally with zero network calls.  Typical scan latency is
    ~12 microseconds.

    Args:
        detect_pii: Enable PII detection. Default ``True``.
        detect_injection: Enable prompt-injection detection. Default ``True``.
        on_injection: Action when injection is detected: ``"block"`` raises
            :class:`PromptInjectionError`, ``"warn"`` logs a warning and
            continues. Default ``"block"``.
        redact_pii: If ``True``, redact PII in prompts before they reach the
            LLM by mutating the prompts list in-place. Default ``True``.
        block_pii: If ``True``, raise :class:`PiiDetectedError` instead of
            redacting. Takes precedence over *redact_pii*. Default ``False``.
        injection_threshold: Score threshold (0.0--1.0) above which a prompt
            is considered an injection attempt. Default ``0.7``.
        pii_types: PII entity types to detect. ``None`` means all supported
            types.  Valid values: ``"email"``, ``"phone"``, ``"ssn"``,
            ``"credit_card"``, ``"api_key"``, ``"ip_address"``.
        redact_with: Replacement token for redacted PII, e.g.
            ``"[REDACTED]"``. Default ``"[REDACTED]"``.
        on_scan: Optional callback invoked with each :class:`ScanEvent`.
        logger: Logger instance to use.  Defaults to
            ``logging.getLogger("promptfirewall.langchain")``.

    Example::

        from langchain_openai import ChatOpenAI
        from langchain_promptfirewall import PromptFirewallHandler

        handler = PromptFirewallHandler(
            on_injection="block",
            injection_threshold=0.5,
        )
        llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
        llm.invoke("Summarize this document")
    """

    # LangChain uses this to decide callback ordering
    raise_error = True

    def __init__(
        self,
        *,
        detect_pii: bool = True,
        detect_injection: bool = True,
        on_injection: str = "block",
        redact_pii: bool = True,
        block_pii: bool = False,
        injection_threshold: float = 0.7,
        pii_types: Optional[List[str]] = None,
        redact_with: str = "[REDACTED]",
        on_scan: Optional[Callable[[ScanEvent], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        if on_injection not in ("block", "warn"):
            raise ValueError(
                f"on_injection must be 'block' or 'warn', got {on_injection!r}"
            )
        self.detect_pii = detect_pii
        self.detect_injection = detect_injection
        self.on_injection = on_injection
        self.redact_pii = redact_pii
        self.block_pii = block_pii
        self.injection_threshold = injection_threshold
        self.pii_types = pii_types
        self.redact_with = redact_with
        self.on_scan = on_scan
        self._logger = logger or globals()["logger"]
        self._scan_history: List[ScanEvent] = []

    @property
    def scan_history(self) -> List[ScanEvent]:
        """Return a copy of the scan event history."""
        return list(self._scan_history)

    # ------------------------------------------------------------------
    # LangChain callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Scan each prompt string before it reaches the LLM.

        This callback is invoked by LangChain for non-chat models.  It
        mutates *prompts* in-place when redacting PII.
        """
        for i, prompt in enumerate(prompts):
            redacted = self._scan_text(prompt)
            if redacted is not None:
                prompts[i] = redacted

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Scan chat messages before they reach the LLM.

        Extracts text content from ``BaseMessage`` objects and scans each
        one.  Mutates ``message.content`` in-place when redacting.
        """
        for message_group in messages:
            for message in message_group:
                content = getattr(message, "content", None)
                if not content or not isinstance(content, str):
                    continue
                redacted = self._scan_text(content)
                if redacted is not None:
                    message.content = redacted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_text(self, text: str) -> Optional[str]:
        """Scan *text* and return the redacted version, or ``None`` if unchanged.

        Raises :class:`PromptInjectionError` or :class:`PiiDetectedError`
        when blocking is configured and the respective threat is detected.
        """
        result = promptfirewall.scan(
            text,
            detect_pii=self.detect_pii,
            detect_injection=self.detect_injection,
            pii_types=self.pii_types,
            injection_threshold=self.injection_threshold,
            redact=self.redact_pii or self.block_pii,
            redact_with=self.redact_with,
        )

        pii_dicts = [f.to_dict() for f in result.pii_findings]
        action = "passed"
        redacted: Optional[str] = None

        # -- Injection check (higher severity) -------------------------
        if (
            self.detect_injection
            and result.injection_score >= self.injection_threshold
        ):
            action = "blocked_injection"
            self._record(text, result, pii_dicts, action)

            if self.on_injection == "block":
                raise PromptInjectionError(
                    f"Prompt injection detected "
                    f"(score={result.injection_score:.2f}, "
                    f"labels={result.injection_labels})",
                    injection_score=result.injection_score,
                    injection_labels=list(result.injection_labels),
                )
            self._logger.warning(
                "Prompt injection detected (score=%.2f, labels=%s) "
                "but blocking is disabled",
                result.injection_score,
                result.injection_labels,
            )

        # -- PII check -------------------------------------------------
        if self.detect_pii and result.pii_findings:
            if self.block_pii:
                action = "blocked_pii"
                self._record(text, result, pii_dicts, action)
                types_found = [f["entity_type"] for f in pii_dicts]
                raise PiiDetectedError(
                    f"PII detected in prompt: {types_found}",
                    pii_findings=pii_dicts,
                )

            if self.redact_pii and result.redacted_text is not None:
                action = "redacted"
                redacted = result.redacted_text
                self._logger.info(
                    "Redacted %d PII finding(s) in prompt",
                    len(result.pii_findings),
                )

        self._record(text, result, pii_dicts, action)
        return redacted

    def _record(
        self,
        prompt: str,
        result: Any,
        pii_dicts: List[Dict[str, Any]],
        action: str,
    ) -> None:
        """Append a :class:`ScanEvent` and invoke the *on_scan* callback."""
        event = ScanEvent(
            prompt=prompt,
            is_safe=result.is_safe,
            injection_score=result.injection_score,
            injection_labels=list(result.injection_labels),
            pii_findings=pii_dicts,
            redacted_text=result.redacted_text,
            action_taken=action,
            latency_us=result.latency_us,
        )
        self._scan_history.append(event)
        if self.on_scan:
            self.on_scan(event)
