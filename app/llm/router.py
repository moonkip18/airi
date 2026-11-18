from typing import List, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
import json
import structlog

from openai import AsyncOpenAI

from app.llm.base import LLMProvider
from app.config import settings

T = TypeVar("T", bound=BaseModel)

logger = structlog.get_logger()


class RouterProvider(LLMProvider):
    """LLM provider for the configured OpenAI-compatible gateway — an OpenAI-compatible
    unified gateway. Chat Completions protocol; Bearer auth."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key or settings.llm_api_key,
            base_url=settings.llm_base_url,
            # Explicit ceiling per HTTP attempt: the SDK default is a 600s
            # read timeout, so one hung gateway call would hold the
            # per-conversation lock for up to 10 minutes and the whole chat
            # looks frozen. Bounded here instead.
            timeout=settings.llm_timeout_seconds,
            # The SDK only retries transient errors (429/5xx/network), but
            # its default of 2 retries silently multiplies worst-case
            # latency; this app does its own (bounded) retrying on top.
            max_retries=settings.llm_max_retries,
        )

        self.model = model or settings.llm_model

        # Thinking models (GLM etc.) spend tokens AND wall-clock time on
        # internal reasoning before answering. For her tasks (casual chat
        # decisions, JSON extraction) it's mostly wasted latency, so we ask
        # the provider to disable it. B.AI's GLM refuses to fully disable
        # thinking but accepts an effort level ("low"), so the negotiation
        # is a chain: disabled -> effort=low -> plain requests. The failed
        # state is permanent for the process lifetime.
        self._thinking_state = "unknown" if settings.llm_disable_thinking else "off"

    def _create_kwargs(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        thinking: Optional[str] = None,
    ) -> Dict:
        kwargs: Dict = {
            "model": self.model,
            "messages": messages,
            # Upstream GLM only accepts 2-decimal temperatures; float noise
            # from callers gets normalized here (defense in depth).
            "temperature": round(float(temperature), 2),
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if thinking == "disabled":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        elif thinking == "low":
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "effort": "low"}}
        return kwargs

    async def _create(self, messages, temperature: float, max_tokens: int):
        """Single completion request with thinking-disable negotiation."""
        if self._thinking_state == "unknown":
            try:
                response = await self.client.chat.completions.create(
                    **self._create_kwargs(
                        messages, temperature, max_tokens, thinking="disabled",
                    )
                )
                self._thinking_state = "disabled"
                logger.info("llm_thinking_disabled")
                return response
            except Exception as first_error:
                logger.warning(
                    "llm_thinking_disable_unsupported",
                    error=str(first_error)[:150],
                )
                # Some gateways always think but let you tune the effort —
                # that still saves most of the wasted latency.
                try:
                    response = await self.client.chat.completions.create(
                        **self._create_kwargs(
                            messages, temperature, max_tokens, thinking="low",
                        )
                    )
                    self._thinking_state = "low"
                    logger.info("llm_thinking_low_effort")
                    return response
                except Exception as second_error:
                    # Provider understands neither parameter — plain requests.
                    self._thinking_state = "off"
                    logger.warning(
                        "llm_thinking_param_unsupported",
                        error=str(second_error)[:150],
                    )
                    return await self.client.chat.completions.create(
                        **self._create_kwargs(messages, temperature, max_tokens)
                    )
        return await self.client.chat.completions.create(
            **self._create_kwargs(messages, temperature, max_tokens)
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> str | BaseModel:

        if response_format:
            return await self.generate_structured(
                messages,
                response_format,
                temperature,
                max_tokens,
            )

        try:

            # glm-5.3-flash on B.AI always thinks: with a small max_tokens
            # budget the thinking phase can consume everything and the
            # visible content comes back EMPTY. Escalate and retry instead
            # of returning an empty string.
            current_max_tokens = max_tokens
            for attempt in range(1, 3):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=round(float(temperature), 2),
                    max_tokens=current_max_tokens,
                )

                content = response.choices[0].message.content or ""
                if content.strip() or attempt == 2:
                    return content

                logger.warning(
                    "llm_empty_content_retry",
                    attempt=attempt,
                    max_tokens=current_max_tokens,
                )
                # Jump hard: on this model the thinking phase alone can eat
                # several hundred tokens; small escalations stay starved.
                current_max_tokens = min(4000, max(1000, current_max_tokens * 8))

            return ""

        except Exception as e:

            logger.error(
                "llm_generation_failed",
                error=str(e),
                model=self.model,
            )

            raise

    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> T:

        schema = response_model.model_json_schema()

        schema_str = json.dumps(
            schema,
            indent=2,
        )

        system_prompt = (
            "You are a structured output generator. "
            "Respond ONLY with valid JSON matching the provided schema. "
            "Do not include markdown, explanations, or additional text. "
            "Keep string values SHORT (reasoning_summary under 12 words; "
            "response is a casual chat message, not an essay).\n\n"
            f"Schema:\n{schema_str}"
        )

        structured_messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *messages,
        ]

        last_error: Optional[Exception] = None
        # Thinking models (e.g. GLM via OpenAI-compatible endpoints) spend
        # tokens on internal reasoning BEFORE writing the visible answer.
        # If the budget runs out mid-thought the visible content comes back
        # EMPTY (or truncated JSON). So on each retry we escalate the budget.
        current_max_tokens = max_tokens

        import time as _time

        # Retry a couple of times: a single malformed JSON or a validation
        # slip must not silently drop the whole interaction (e.g. a reply
        # she clearly decided to send). The invalid output is fed back so
        # the model can correct itself.
        for attempt in range(1, 4):
            call_started = _time.monotonic()
            try:

                response = await self._create(
                    structured_messages, temperature, current_max_tokens,
                )

                call_seconds = round(_time.monotonic() - call_started, 1)
                usage = getattr(response, "usage", None)
                logger.info(
                    "llm_call_done",
                    seconds=call_seconds,
                    attempt=attempt,
                    max_tokens=current_max_tokens,
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )

                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                finish_reason = str(getattr(choice, "finish_reason", "") or "")

                if not content:
                    # Empty answer = the thinking budget was consumed before
                    # any output. Escalate and retry; do NOT treat "{}" as
                    # the model's decision.
                    logger.warning(
                        "llm_empty_content_retry",
                        attempt=attempt,
                        finish_reason=finish_reason,
                        max_tokens=current_max_tokens,
                    )
                    last_error = ValueError("empty model output")
                    current_max_tokens = min(4000, current_max_tokens * 3)
                    continue

                if finish_reason == "length":
                    # Truncated mid-output (thinking ate the budget). Don't
                    # waste a round feeding back "bad JSON" — just escalate.
                    logger.warning(
                        "llm_truncated_retry",
                        attempt=attempt,
                        max_tokens=current_max_tokens,
                        content_preview=content[:80],
                    )
                    last_error = json.JSONDecodeError(
                        "output truncated by max_tokens", content or "", 0,
                    )
                    current_max_tokens = min(4000, current_max_tokens * 3)
                    continue

                data = json.loads(content)

                return response_model(**data)

            except (json.JSONDecodeError, ValidationError) as e:

                last_error = e

                logger.warning(
                    "llm_structured_retry",
                    attempt=attempt,
                    error=str(e)[:200],
                )

                # Truncated thinking can also produce cut-off JSON —
                # escalate the budget alongside the corrective feedback.
                current_max_tokens = min(4000, current_max_tokens * 3)

                structured_messages = structured_messages + [
                    {
                        "role": "assistant",
                        "content": content,
                    },
                    {
                        "role": "user",
                        "content": (
                            "That was not valid JSON matching the schema. "
                            "Respond again with ONLY valid JSON that "
                            "matches the schema exactly."
                        ),
                    },
                ]

            except Exception as e:

                logger.error(
                    "llm_structured_generation_failed",
                    error=str(e),
                    model=self.model,
                )

                raise

        logger.error(
            "llm_structured_json_decode_failed",
            error=str(last_error),
        )

        raise last_error

    async def close(self):

        await self.client.close()


llm_provider = RouterProvider()
