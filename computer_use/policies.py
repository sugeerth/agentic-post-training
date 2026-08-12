"""VLM policies — the thing that looks at a screenshot and decides what to do.

A policy owns the conversation; the rollout owns the environment. The split
matters because the conversation is where all the context-management decisions
live (how many frames to keep, what to cache, how to answer a malformed
action), and none of those belong in the environment loop.

The lifecycle is two calls:

    decision = await policy.begin(task, first_frame)
    while decision.action is not None:
        frame = await env.execute(decision.action)
        decision = await policy.observe(frame)

`Decision.action is None` means the model stopped calling the tool and answered
in text — that is how a computer-use agent signals it is finished.

Why a hand-written loop rather than the SDK tool runner: the computer tool is
Anthropic-defined and schema-less, so there is no Python function for the
runner to call. The execution side is an environment, not a callable, and the
loop has to interleave with it. The runner remains the right choice for
ordinary custom tools.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from computer_use.actions import (
    DEFAULT_TOOL_BETA,
    DEFAULT_TOOL_VERSION,
    computer_tool,
    parse_tool_use,
    system_prompt,
)
from computer_use.types import Action, Screenshot

#: Claude's most capable widely available model is the default here because
#: GUI grounding is one of the tasks where model capability shows up most
#: directly in the success rate. Override per-run if you are cost-bound.
DEFAULT_MODEL = "claude-opus-5"

#: Opting into server-side refusal fallbacks: on a policy decline the API
#: re-runs the request on Anthropic's recommended fallback model inside the
#: same call, so a long rollout does not die on one classifier hit.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True, slots=True)
class Decision:
    """What the policy decided to do with the frame it was shown."""

    action: Action | None
    rationale: str = ""
    tool_use_id: str | None = None
    stop_reason: str = ""
    usage: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.action is None


@runtime_checkable
class VLMPolicy(Protocol):
    """A policy that maps (task, screenshots) to actions."""

    async def begin(self, task: str, screenshot: Screenshot) -> Decision: ...

    async def observe(
        self, screenshot: Screenshot, *, error: str | None = None
    ) -> Decision: ...

    def reset(self) -> None: ...


# --------------------------------------------------------------------------- #
# Claude computer-use policy
# --------------------------------------------------------------------------- #


@dataclass
class PolicyConfig:
    """Knobs for `ClaudeComputerUsePolicy`.

    `keep_frames` is the one worth tuning first. Screenshots dominate a
    computer-use context window, and stale frames are actively unhelpful — the
    agent should be reasoning about what is on screen now. Keeping the last few
    frames and replacing older ones with a text placeholder holds context flat
    across a long episode instead of growing it linearly.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 16000
    effort: str = "high"
    tool_version: str = DEFAULT_TOOL_VERSION
    tool_beta: str = DEFAULT_TOOL_BETA
    keep_frames: int = 3
    cache_system_prompt: bool = True
    enable_refusal_fallback: bool = True
    display_number: int | None = None
    extra_context: str = ""


class RefusalError(RuntimeError):
    """The model declined the task. Carries the policy category when given."""

    def __init__(self, message: str, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


class ClaudeComputerUsePolicy:
    """Drives a Claude model through the computer-use tool.

    Conversation state lives on the instance: `begin` seeds it, `observe`
    extends it. Call `reset()` between episodes, or construct one policy per
    episode — the rollout runner does the latter so concurrent episodes cannot
    share history.
    """

    def __init__(
        self,
        env_width: int,
        env_height: int,
        config: PolicyConfig | None = None,
        *,
        client: Any = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self.width = env_width
        self.height = env_height
        self._client = client
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._messages: list[dict[str, Any]] = []
        self._task = ""
        self._pending_tool_use_id: str | None = None

    # ---- client ----------------------------------------------------------- #

    def _get_client(self) -> Any:
        """Lazily construct an `AsyncAnthropic`.

        Constructed with no explicit key when none was passed: the SDK resolves
        `ANTHROPIC_API_KEY`, then `ANTHROPIC_AUTH_TOKEN`, then an `ant auth
        login` profile — so a machine authenticated with a profile works
        without any environment variable set.
        """
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ClaudeComputerUsePolicy needs the Anthropic SDK. Install with "
                '`pip install "agentic-post-training[computer-use]"`.'
            ) from exc
        self._client = AsyncAnthropic(api_key=self._api_key) if self._api_key else AsyncAnthropic()
        return self._client

    # ---- VLMPolicy -------------------------------------------------------- #

    def reset(self) -> None:
        self._messages = []
        self._task = ""
        self._pending_tool_use_id = None

    async def begin(self, task: str, screenshot: Screenshot) -> Decision:
        self.reset()
        self._task = task
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Task: {task}"},
                screenshot.to_content_block(),
                {"type": "text", "text": "This is the current screen. Begin."},
            ],
        })
        return await self._turn()

    async def observe(
        self, screenshot: Screenshot, *, error: str | None = None
    ) -> Decision:
        """Feed back the frame produced by the last action.

        An `error` is reported through the same channel with `is_error: true`
        rather than being raised, so the model can see what went wrong and
        correct itself — which is the whole point of returning a tool result.
        """
        if self._pending_tool_use_id is None:
            raise RuntimeError("observe() called with no action outstanding")

        content: list[dict[str, Any]] = []
        if error:
            content.append({"type": "text", "text": f"Action failed: {error}"})
        content.append(screenshot.to_content_block())

        self._messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": self._pending_tool_use_id,
                "content": content,
                **({"is_error": True} if error else {}),
            }],
        })
        self._pending_tool_use_id = None
        return await self._turn()

    # ---- one API round-trip ---------------------------------------------- #

    async def _turn(self) -> Decision:
        cfg = self.config
        client = self._get_client()
        self._prune_frames()

        betas = [cfg.tool_beta]
        request: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "system": self._system_blocks(),
            "messages": self._messages,
            "tools": [computer_tool(
                self.width, self.height,
                version=cfg.tool_version,
                display_number=cfg.display_number,
            )],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": cfg.effort},
        }
        if cfg.enable_refusal_fallback:
            betas.append(FALLBACK_BETA)
            request["fallbacks"] = "default"
        request["betas"] = betas

        response = await client.beta.messages.create(**request)

        # Check the stop reason before touching content: a refusal returns a
        # successful 200 whose content may be empty.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise RefusalError(
                f"model declined the task: {getattr(details, 'explanation', '') or 'no explanation'}",
                category=getattr(details, "category", None),
            )

        self._messages.append({"role": "assistant", "content": response.content})

        rationale = " ".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        tool_use = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"), None
        )
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        }

        if tool_use is None:
            return Decision(None, rationale, None, response.stop_reason or "end_turn", usage)

        self._pending_tool_use_id = tool_use.id
        return Decision(
            action=parse_tool_use(tool_use),
            rationale=rationale,
            tool_use_id=tool_use.id,
            stop_reason=response.stop_reason or "tool_use",
            usage=usage,
        )

    # ---- context management ---------------------------------------------- #

    def _system_blocks(self) -> list[dict[str, Any]]:
        block: dict[str, Any] = {
            "type": "text",
            "text": system_prompt(self.config.extra_context),
        }
        if self.config.cache_system_prompt:
            # Tools render before system, so a breakpoint here caches both.
            # Everything volatile (the frames) lives after it in `messages`.
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _prune_frames(self) -> None:
        """Keep only the most recent `keep_frames` screenshots in history.

        Older images are swapped for a one-line placeholder in place. Text,
        tool_use blocks, and the tool_result envelopes all stay, so the model
        keeps the full record of what it did — it just stops re-reading frames
        that no longer describe the screen. Without this, context grows by a
        full image every turn and a 40-step episode spends most of its budget
        on screens that are long gone.

        This rewrites part of the message history, so only the system + tools
        prefix stays cacheable across turns. That is the right trade here:
        the frames are the volatile part, and they sit after the breakpoint.
        """
        budget = max(1, self.config.keep_frames)
        slots: list[tuple[list[Any], int]] = []

        for message in self._messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for i, block in enumerate(content):
                if not isinstance(block, dict):
                    continue  # SDK block objects (assistant turns) hold no images
                if block.get("type") == "image":
                    slots.append((content, i))
                elif block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        slots.extend(
                            (inner, j)
                            for j, sub in enumerate(inner)
                            if isinstance(sub, dict) and sub.get("type") == "image"
                        )

        for container, index in slots[: max(0, len(slots) - budget)]:
            container[index] = {"type": "text", "text": "[earlier screenshot omitted]"}


# --------------------------------------------------------------------------- #
# Deterministic policies — tests, demos, and baselines
# --------------------------------------------------------------------------- #


class ScriptedPolicy:
    """Replays a fixed action list. No API calls, no network, no keys.

    Two uses beyond testing: it is the baseline a learned policy has to beat,
    and it is how you record a gold trajectory to pair against a model's
    attempt when building preference data.

    `actions` may also be a callable taking the step index and returning an
    `Action` or `None` (meaning "done"), which is enough to express a reactive
    scripted agent without pulling in a policy framework.
    """

    def __init__(
        self,
        actions: Sequence[Action] | Callable[[int], Action | None],
        *,
        final_response: str = "Done.",
        rationales: Iterable[str] = (),
    ) -> None:
        self._actions = actions
        self._rationales = list(rationales)
        self._final = final_response
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    async def begin(self, task: str, screenshot: Screenshot) -> Decision:
        self.reset()
        return self._next()

    async def observe(
        self, screenshot: Screenshot, *, error: str | None = None
    ) -> Decision:
        return self._next()

    def _next(self) -> Decision:
        idx = self._index
        if callable(self._actions):
            action = self._actions(idx)
        else:
            action = self._actions[idx] if idx < len(self._actions) else None

        rationale = self._rationales[idx] if idx < len(self._rationales) else ""
        self._index += 1
        if action is None:
            return Decision(None, rationale or self._final, None, "end_turn")
        return Decision(action, rationale, f"scripted_{idx}", "tool_use")
