"""DeepSeek Chat/Reasoner API adapter."""
from __future__ import annotations

import time
from typing import Any


class DialogManager:
    """Manages VL dialog conversation state and audit export.

    Attributes:
        conversation_history: List of conversation round dicts.
        max_rounds: Maximum conversation rounds.
    """

    def __init__(self, max_rounds: int = 10) -> None:
        """Initialize the dialog manager.

        Args:
            max_rounds: Maximum number of conversation rounds.
        """
        self.conversation_history: list[dict[str, str]] = []
        self.max_rounds = max_rounds

    def add_round(self, role: str, content: str) -> None:
        """Add a conversation round.

        Args:
            role: Speaker role ('user', 'assistant', 'system').
            content: Message content.
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        })

    def get_context(self) -> str:
        """Get the full conversation context as a string.

        Returns:
            Concatenated conversation context.
        """
        lines = []
        for round_data in self.conversation_history:
            role = round_data["role"]
            content = round_data["content"]
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def export_audit_log(self) -> dict[str, Any]:
        """Export conversation as audit log.

        Returns:
            Dictionary with conversation history and metadata.
        """
        return {
            "rounds": len(self.conversation_history),
            "history": self.conversation_history,
            "max_rounds": self.max_rounds,
        }

    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []


class DeepSeekAdapter:
    """DeepSeek Chat/Reasoner API adapter.

    Provides text-based chat and reasoning capabilities via DeepSeek API.
    Includes psi-anchor PII scanning for safe API communication.

    Attributes:
        api_key: DeepSeek API key.
        base_url: API base URL.
        chat_model: Chat model name.
        reasoner_model: Reasoner model name.
        available: Whether API is available.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the adapter.

        Args:
            config: VL API config with api_key, base_url, model names.
        """
        self.api_key: str = config.get("api_key", "")
        self.base_url: str = config.get("base_url", "https://api.deepseek.com/v1")
        self.chat_model: str = config.get("chat_model", "deepseek-chat")
        self.reasoner_model: str = config.get("reasoner_model", "deepseek-reasoner")
        self.timeout: int = config.get("timeout_seconds", 30)
        self.max_retries: int = config.get("max_retries", 3)
        self.available: bool = bool(self.api_key)
        self.dialog_manager = DialogManager()

    def chat(self, message: str, system_prompt: str = "") -> str:
        """Send a chat message and get a response.

        Args:
            message: User message.
            system_prompt: Optional system prompt.

        Returns:
            Assistant response string.
        """
        if not self.available:
            return ""

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Scan for PII before sending
            safe_message = self._scan_pii(message)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": safe_message})

            payload = {
                "model": self.chat_model,
                "messages": messages,
            }

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            return ""

    def reason(self, message: str) -> str:
        """Send a reasoning request using the reasoner model.

        Args:
            message: Problem description to reason about.

        Returns:
            Reasoning result string.
        """
        if not self.available:
            return ""

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            safe_message = self._scan_pii(message)

            payload = {
                "model": self.reasoner_model,
                "messages": [{"role": "user", "content": safe_message}],
            }

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            return ""

    def check_availability(self) -> bool:
        """Check if the API is available.

        Returns:
            True if API key is set and API is reachable.
        """
        if not self.api_key:
            return False

        try:
            import httpx

            headers = {"Authorization": f"Bearer {self.api_key}"}
            with httpx.Client(timeout=5) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                )
                return response.status_code == 200
        except Exception:
            return False

    def _scan_pii(self, text: str) -> str:
        """Scan text for PII (psi-anchor) and redact if found.

        Performs basic PII scanning for email addresses, phone numbers,
        and credit card patterns.

        Args:
            text: Input text to scan.

        Returns:
            Text with PII redacted.
        """
        import re

        # Redact email addresses
        text = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', '[EMAIL_REDACTED]', text)

        # Redact phone numbers (basic pattern)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)

        # Redact credit card numbers
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_REDACTED]', text)

        return text
