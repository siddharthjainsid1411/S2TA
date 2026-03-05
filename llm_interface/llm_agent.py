# llm_interface/llm_agent.py
#
# Gemini-backed LLM agent with retry loop.
# Builds the system prompt, calls the Gemini API, parses JSON from the
# response, runs the two-pass validator, and retries on validation failure
# (up to MAX_RETRIES attempts).

import json
import logging
import os
import re

from llm_interface.prompt_builder import build_system_prompt
from llm_interface.validator import validate_and_clamp

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _extract_json(text: str) -> dict:
    """Extract the first complete JSON object from an LLM response string."""
    text = text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Find the first top-level JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

    # Walk to find matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                return json.loads(json_str)

    raise ValueError("Could not find complete JSON object in LLM response.")


def _build_gemini_contents(messages: list) -> list:
    """
    Convert internal message list to the google-genai contents format.
    System messages are skipped here — they are passed via system_instruction.
    """
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        # Skip the system message — handled via system_instruction config
        if msg["role"] == "system":
            continue
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    return contents


class LLMAgent:
    """
    Gemini-backed task specification generator.

    Parameters
    ----------
    model   : Gemini model name.
    api_key : Gemini API key. Falls back to GEMINI_API_KEY environment variable.
    """

    def __init__(
        self,
        model:   str = "gemini-3.1-flash-lite-preview",
        api_key: str = None,
    ):
        self.model   = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._system_prompt = None

    def _get_client(self):
        """Lazily import and create the Gemini client."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )
        client = genai.Client(api_key=self.api_key)
        # Apply httpx timeout to avoid indefinite hangs on slow networks.
        if hasattr(client, '_http_client') and hasattr(client._http_client, '_client'):
            from httpx import Timeout
            client._http_client._client._timeout = Timeout(30.0)
        return client

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(include_examples=True)
        return self._system_prompt

    def _call_llm(self, client, messages: list) -> str:
        """Make one Gemini API call and return the response text."""
        from google.genai import types

        system_prompt = self._get_system_prompt()
        contents      = _build_gemini_contents(messages)

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        return response.text

    def generate(self, task_description: str) -> dict:
        """
        Generate and validate a task spec from a natural language description.

        Parameters
        ----------
        task_description : str
            Natural language description of the manipulation task.

        Returns
        -------
        dict : validated and clamped task spec, ready for json_parser.

        Raises
        ------
        RuntimeError if validation still fails after MAX_RETRIES attempts.
        """
        client = self._get_client()

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user",   "content": task_description},
        ]

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info("Gemini call attempt %d / %d ...", attempt, MAX_RETRIES)

            raw_text = self._call_llm(client, messages)
            logger.debug("Gemini raw response:\n%s", raw_text)

            # Parse JSON
            try:
                spec_dict = _extract_json(raw_text)
            except (ValueError, json.JSONDecodeError) as e:
                error_msg = f"Could not parse JSON from your response: {e}"
                logger.warning("Attempt %d: JSON parse failed: %s", attempt, e)
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response was not valid JSON.\n"
                        f"Error: {error_msg}\n"
                        f"Please output ONLY a valid JSON object, nothing else."
                    ),
                })
                continue

            # Validate + clamp
            fixed_dict, errors, warnings = validate_and_clamp(spec_dict)

            if warnings:
                for w in warnings:
                    logger.warning("Auto-fixed: %s", w)

            if not errors:
                logger.info(
                    "Spec validated successfully on attempt %d "
                    "(%d auto-fixes applied).",
                    attempt, len(warnings)
                )
                return fixed_dict

            # Retry with error feedback
            error_list = "\n".join(f"  - {e}" for e in errors)
            logger.warning(
                "Attempt %d: validation failed with %d errors:\n%s",
                attempt, len(errors), error_list
            )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    f"Your task spec has the following validation errors. "
                    f"Please fix ALL of them and output the corrected JSON:\n\n"
                    f"{error_list}"
                ),
            })

        raise RuntimeError(
            f"LLM failed to produce a valid task spec after {MAX_RETRIES} attempts. "
            f"Last errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
