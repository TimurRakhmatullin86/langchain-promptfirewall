"""LangChain integration for promptfirewall.

Sub-millisecond PII detection and prompt injection firewall for LangChain.

Usage::

    from langchain_promptfirewall import PromptFirewallHandler

    handler = PromptFirewallHandler()
    llm = ChatOpenAI(callbacks=[handler])
"""

from langchain_promptfirewall._handler import (
    PiiDetectedError,
    PromptFirewallHandler,
    PromptInjectionError,
    ScanEvent,
)

__all__ = [
    "PiiDetectedError",
    "PromptFirewallHandler",
    "PromptInjectionError",
    "ScanEvent",
]

__version__ = "0.1.0"
