"""Tests for the computer-use (GUI) subsystem.

Everything here runs offline: no API key, no browser, no GPU. That is a
property of the design rather than the tests — `MockComputer` renders real
frames and exposes ground-truth state, and `ScriptedPolicy` stands in for the
model, so the loop, the reward, and the data conversions are all exercised for
real. The one test that touches the Anthropic request shape uses a stub client
and asserts on the payload.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import AgentStatus
from agents.communication import MessageBus, MessageType
from computer_use import (
    Action,
    ActionError,
    ActionKind,
    ClaudeComputerUsePolicy,
    ComputerUseAgent,
    MockComputer,
    PolicyConfig,
    RefusalError,
    RewardConfig,
    RolloutConfig,
    ScriptedPolicy,
    StateVerifier,
    Trajectory,
    TrajectoryStatus,
    assign_step_credit,
    computer_tool,
    run_episode,
    run_group,
    score_trajectory,
    validate,
)
from computer_use.dataset import (
    from_dict,
    load_jsonl,
    render_transcript,
    save_jsonl,
    to_dict,
    to_preference_pairs,
    to_rollout_batch,
    to_step_preference_pairs,
    to_training_examples,
)
from computer_use.rewards import Verdict
from computer_use.types import Step

TASK = "Set the email and save"
VERIFIER = StateVerifier({"email": "ada@example.com", "notify": True, "saved": True})

EMAIL_FIELD = (290, 218)
NOTIFY_BOX = (76, 382)
SAVE_AFTER_SCROLL = (305, 624)


def click(x, y):
    return Action(ActionKind.LEFT_CLICK, coordinate=(x, y))


GOOD_SCRIPT = [
    Action(ActionKind.SCREENSHOT),
    click(*EMAIL_FIELD),
    Action(ActionKind.TYPE, text="ada@example.com"),
    click(*NOTIFY_BOX),
    Action(ActionKind.SCROLL, coordinate=(640, 400), scroll_direction="down", scroll_amount=3),
    click(*SAVE_AFTER_SCROLL),
]

SLOPPY_SCRIPT = [
    Action(ActionKind.SCREENSHOT),
    Action(ActionKind.SCREENSHOT),
    click(900, 500),
    click(*EMAIL_FIELD),
    Action(ActionKind.TYPE, text="ada@example.com"),
    click(*NOTIFY_BOX),
    Action(ActionKind.SCROLL, coordinate=(640, 400), scroll_direction="down", scroll_amount=3),
    click(*SAVE_AFTER_SCROLL),
]

BAD_SCRIPT = [
    Action(ActionKind.SCREENSHOT),
    Action(ActionKind.TYPE, text="ada@example.com"),
    click(*NOTIFY_BOX),
]


def episode(script, **kwargs):
    return asyncio.run(run_episode(
        TASK,
        MockComputer.settings_form(),
        ScriptedPolicy(script),
        verifier=VERIFIER,
        reward_config=RewardConfig(optimal_steps=6),
        **kwargs,
    ))


class TestActions(unittest.TestCase):
    def test_tool_input_round_trip(self):
        original = Action(
            ActionKind.SCROLL, coordinate=(10, 20),
            scroll_direction="down", scroll_amount=3,
        )
        restored = Action.from_tool_input(original.to_tool_input())
        self.assertEqual(original, restored)

    def test_parses_coordinate_lists(self):
        action = Action.from_tool_input({"action": "left_click", "coordinate": [5, 6]})
        self.assertEqual(action.coordinate, (5, 6))
        self.assertIs(action.kind, ActionKind.LEFT_CLICK)

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ValueError):
            Action.from_tool_input({"action": "teleport"})

    def test_omits_unset_fields(self):
        payload = Action(ActionKind.SCREENSHOT).to_tool_input()
        self.assertEqual(payload, {"action": "screenshot"})

    def test_validation_catches_missing_arguments(self):
        with self.assertRaises(ActionError):
            validate(Action(ActionKind.LEFT_CLICK))
        with self.assertRaises(ActionError):
            validate(Action(ActionKind.TYPE))
        with self.assertRaises(ActionError):
            validate(Action(ActionKind.SCROLL, coordinate=(1, 1), scroll_direction="sideways"))

    def test_validation_bounds_checks_coordinates(self):
        with self.assertRaises(ActionError):
            validate(click(2000, 10), width=1280, height=720)
        validate(click(100, 100), width=1280, height=720)  # must not raise

    def test_scaled_rescales_both_points(self):
        action = Action(
            ActionKind.LEFT_CLICK_DRAG, coordinate=(100, 200), start_coordinate=(10, 20)
        ).scaled(2.0, 0.5)
        self.assertEqual(action.coordinate, (200, 100))
        self.assertEqual(action.start_coordinate, (20, 10))

    def test_computer_tool_declares_no_input_schema(self):
        # The computer tool is Anthropic-defined; supplying a schema would make
        # it an ordinary custom tool without the trained behavior.
        tool = computer_tool(1280, 720)
        self.assertNotIn("input_schema", tool)
        self.assertEqual(tool["name"], "computer")
        self.assertEqual(tool["display_width_px"], 1280)
        self.assertNotIn("display_number", tool)
        self.assertEqual(computer_tool(800, 600, display_number=1)["display_number"], 1)


class TestMockComputer(unittest.TestCase):
    def test_renders_a_real_png(self):
        env = MockComputer.settings_form()
        frame = asyncio.run(env.screenshot())
        self.assertTrue(frame.data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual((frame.width, frame.height), (1280, 720))
        self.assertGreater(len(frame.data), 500)

    def test_click_focuses_and_type_enters_text(self):
        env = MockComputer.settings_form()
        asyncio.run(env.execute(click(*EMAIL_FIELD)))
        asyncio.run(env.execute(Action(ActionKind.TYPE, text="hi")))
        self.assertEqual(env.state()["email"], "hi")

    def test_typing_without_focus_changes_nothing(self):
        env = MockComputer.settings_form()
        asyncio.run(env.execute(Action(ActionKind.TYPE, text="hi")))
        self.assertEqual(env.state()["email"], "")

    def test_checkbox_toggles(self):
        env = MockComputer.settings_form()
        asyncio.run(env.execute(click(*NOTIFY_BOX)))
        self.assertTrue(env.state()["notify"])
        asyncio.run(env.execute(click(*NOTIFY_BOX)))
        self.assertFalse(env.state()["notify"])

    def test_save_is_unreachable_until_scrolled(self):
        env = MockComputer.settings_form()
        # The button's page coordinate is off-screen, so clicking it is an error.
        with self.assertRaises(ActionError):
            asyncio.run(env.execute(click(305, 804)))
        asyncio.run(env.execute(Action(
            ActionKind.SCROLL, coordinate=(640, 400),
            scroll_direction="down", scroll_amount=3,
        )))
        asyncio.run(env.execute(click(*SAVE_AFTER_SCROLL)))
        self.assertTrue(env.state()["saved"])

    def test_scroll_clamps_to_content(self):
        env = MockComputer.settings_form()
        asyncio.run(env.execute(Action(
            ActionKind.SCROLL, coordinate=(640, 400),
            scroll_direction="down", scroll_amount=99,
        )))
        self.assertEqual(env.state()["scroll_y"], 900 - 720)

    def test_reset_restores_initial_state(self):
        env = MockComputer.settings_form()
        asyncio.run(env.execute(click(*NOTIFY_BOX)))
        asyncio.run(env.reset())
        self.assertFalse(env.state()["notify"])
        self.assertEqual(env.action_log, [])


class TestRollout(unittest.TestCase):
    def test_successful_episode_is_verified(self):
        trajectory = episode(GOOD_SCRIPT)
        self.assertIs(trajectory.status, TrajectoryStatus.SUCCESS)
        self.assertEqual(trajectory.num_steps, 6)
        self.assertEqual(trajectory.metadata["verdict"]["reason"], "all criteria met")

    def test_verifier_overrides_a_confident_wrong_claim(self):
        # BAD_SCRIPT ends by "finishing" without ever saving. The policy says
        # it is done; the verifier says otherwise, and the verifier wins.
        trajectory = episode(BAD_SCRIPT)
        self.assertIs(trajectory.status, TrajectoryStatus.FAILURE)
        self.assertFalse(trajectory.metadata["verdict"]["details"]["saved"])

    def test_step_budget_is_enforced(self):
        forever = ScriptedPolicy(lambda i: Action(ActionKind.SCREENSHOT))
        trajectory = asyncio.run(run_episode(
            TASK, MockComputer.settings_form(), forever,
            verifier=VERIFIER, config=RolloutConfig(max_steps=3),
        ))
        self.assertIs(trajectory.status, TrajectoryStatus.MAX_STEPS)
        self.assertEqual(trajectory.num_steps, 3)

    def test_invalid_action_is_recorded_not_raised(self):
        trajectory = episode([click(305, 804), *GOOD_SCRIPT])
        self.assertIsNotNone(trajectory.steps[0].error)
        self.assertTrue(trajectory.steps[0].failed)
        self.assertGreater(trajectory.num_steps, 1)  # the episode continued

    def test_frames_can_be_dropped(self):
        trajectory = episode(GOOD_SCRIPT, config=RolloutConfig(store_frames=False))
        self.assertTrue(all(s.observation is None for s in trajectory.steps))

    def test_frames_are_stored_by_default(self):
        trajectory = episode(GOOD_SCRIPT)
        self.assertTrue(all(s.observation is not None for s in trajectory.steps))

    def test_grounding_ignores_scrolls(self):
        # A scroll over empty background is correct behavior and must not be
        # scored as a missed click.
        trajectory = episode(GOOD_SCRIPT)
        scroll_step = next(s for s in trajectory.steps if s.action.kind is ActionKind.SCROLL)
        self.assertNotIn("hit", scroll_step.metadata)
        self.assertEqual(trajectory.metadata["stats"]["grounding_rate"], 1.0)

    def test_run_group_produces_independent_attempts(self):
        group = asyncio.run(run_group(
            TASK,
            MockComputer.settings_form,
            lambda env: ScriptedPolicy(GOOD_SCRIPT),
            group_size=4,
            verifier=VERIFIER,
            reward_config=RewardConfig(optimal_steps=6),
        ))
        self.assertEqual(len(group), 4)
        self.assertTrue(all(t.succeeded for t in group))


class TestRewards(unittest.TestCase):
    def setUp(self):
        self.good = episode(GOOD_SCRIPT)
        self.sloppy = episode(SLOPPY_SCRIPT)
        self.bad = episode(BAD_SCRIPT)

    def test_quality_ordering(self):
        self.assertGreater(self.good.reward, self.sloppy.reward)
        self.assertGreater(self.sloppy.reward, self.bad.reward)

    def test_success_dominates_shaping(self):
        # No amount of efficient, well-aimed failing outranks a clumsy success.
        self.assertGreater(self.sloppy.reward, self.bad.reward + 0.3)

    def test_rewards_do_not_saturate_at_the_clip(self):
        # A flawless run reaches exactly 1.0; anything less must land below it,
        # or shaping carries no information between successes.
        self.assertAlmostEqual(self.good.reward, 1.0, places=6)
        self.assertLess(self.sloppy.reward, 1.0)

    def test_breakdown_explains_the_score(self):
        breakdown = self.good.metadata["reward_breakdown"]
        self.assertAlmostEqual(sum(breakdown.values()), self.good.reward, places=6)
        self.assertIn("efficiency", breakdown)

    def test_redundancy_is_penalized(self):
        self.assertGreater(self.sloppy.metadata["stats"]["redundancy_rate"], 0)
        self.assertEqual(self.good.metadata["stats"]["redundancy_rate"], 0)

    def test_partial_credit_reflects_criteria_met(self):
        verdict = Verdict(False, "", {"a": True, "b": False, "c": False})
        self.assertAlmostEqual(verdict.partial_credit, 1 / 3)
        self.assertEqual(Verdict(True, "").partial_credit, 1.0)

    def test_state_verifier_supports_predicates(self):
        verifier = StateVerifier({"retries": lambda v: v.isdigit()})
        traj = Trajectory(task="t", steps=(), status=TrajectoryStatus.SUCCESS)
        self.assertTrue(verifier(traj, {"retries": "3"}).success)
        self.assertFalse(verifier(traj, {"retries": "many"}).success)
        # A predicate that raises is a failed criterion, not a crash.
        self.assertFalse(verifier(traj, {"retries": None}).success)

    def test_step_credit_is_discounted_toward_the_end(self):
        steps = tuple(
            Step(index=i, action=Action(ActionKind.SCREENSHOT)) for i in range(4)
        )
        credits = assign_step_credit(steps, 1.0, RewardConfig(discount=0.9))
        self.assertEqual(len(credits), 4)
        self.assertAlmostEqual(credits[-1], 1.0)
        self.assertLess(credits[0], credits[-1])

    def test_step_credit_penalizes_the_failing_step(self):
        steps = (
            Step(index=0, action=Action(ActionKind.SCREENSHOT)),
            Step(index=1, action=Action(ActionKind.SCREENSHOT), error="boom"),
        )
        credits = assign_step_credit(steps, 1.0, RewardConfig(discount=1.0))
        self.assertLess(credits[1], credits[0])

    def test_score_trajectory_corrects_an_optimistic_status(self):
        traj = Trajectory(task="t", steps=(), status=TrajectoryStatus.SUCCESS)
        scored = score_trajectory(traj, Verdict(False, "nope"))
        self.assertIs(scored.status, TrajectoryStatus.FAILURE)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.trajectories = [episode(GOOD_SCRIPT), episode(SLOPPY_SCRIPT), episode(BAD_SCRIPT)]

    def test_preference_pairs_prefer_the_better_trajectory(self):
        pairs = to_preference_pairs(self.trajectories)
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertGreater(pair.metadata["chosen_reward"], pair.metadata["rejected_reward"])
            self.assertGreaterEqual(pair.metadata["margin"], 0.05)

    def test_min_margin_filters_out_noise(self):
        self.assertEqual(to_preference_pairs(self.trajectories, min_margin=2.0), [])

    def test_single_trajectory_yields_no_pairs(self):
        self.assertEqual(to_preference_pairs(self.trajectories[:1]), [])

    def test_step_pairs_isolate_the_diverging_decision(self):
        pairs = to_step_preference_pairs(self.trajectories, include_frames=False)
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertNotEqual(pair.chosen, pair.rejected)
            self.assertIn("step_index", pair.metadata)
            # chosen/rejected are single actions, not whole transcripts
            self.assertEqual(len(pair.chosen.splitlines()), 1)

    def test_step_pairs_carry_the_observation_when_asked(self):
        pairs = to_step_preference_pairs(self.trajectories, include_frames=True)
        self.assertTrue(pairs)
        self.assertIn("observation", pairs[0].metadata)
        self.assertIn("data", pairs[0].metadata["observation"])

    def test_rollout_batch_matches_grpo_expectations(self):
        batch = to_rollout_batch({TASK: self.trajectories})
        self.assertEqual(len(batch.prompts), 1)
        self.assertEqual(len(batch.responses[0]), 3)
        self.assertEqual(len(batch.rewards[0]), 3)
        self.assertEqual(batch.metadata["groups"][0]["group_size"], 3)

    def test_rollout_batch_feeds_grpo_directly(self):
        from techniques.grpo import GRPOTechnique

        technique = GRPOTechnique()
        technique.prepare(model=None, tokenizer=None, cfg=None)
        metrics = technique.step(to_rollout_batch({TASK: self.trajectories}))
        self.assertEqual(metrics.step, 1)
        self.assertIsInstance(metrics.loss, float)

    def test_training_examples_keep_only_verified_successes(self):
        examples = to_training_examples(self.trajectories)
        self.assertEqual(len(examples), 1)  # best_per_task
        self.assertIn("ada@example.com", examples[0].response)

    def test_transcript_includes_actions_and_rationale(self):
        trajectory = asyncio.run(run_episode(
            TASK, MockComputer.settings_form(),
            ScriptedPolicy(GOOD_SCRIPT, rationales=["looking first"]),
            verifier=VERIFIER,
        ))
        text = render_transcript(trajectory)
        self.assertIn("# looking first", text)
        self.assertIn('"action": "left_click"', text)
        self.assertNotIn("# looking first", render_transcript(trajectory, include_rationale=False))

    def test_dict_round_trip_without_frames(self):
        original = self.trajectories[0]
        restored = from_dict(to_dict(original))
        self.assertEqual(restored.task, original.task)
        self.assertEqual(restored.status, original.status)
        self.assertAlmostEqual(restored.reward, original.reward)
        self.assertEqual(
            [s.action for s in restored.steps], [s.action for s in original.steps]
        )
        self.assertIsNone(restored.steps[0].observation)

    def test_dict_round_trip_preserves_frames(self):
        original = self.trajectories[0]
        restored = from_dict(to_dict(original, include_frames=True))
        self.assertEqual(restored.steps[0].observation.data, original.steps[0].observation.data)

    def test_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "traj.jsonl"
            count = save_jsonl(self.trajectories, path)
            self.assertEqual(count, 3)
            reloaded = list(load_jsonl(path))
            self.assertEqual(len(reloaded), 3)
            self.assertEqual(reloaded[0].task, self.trajectories[0].task)
            # Line-delimited, one object per line.
            self.assertEqual(len(path.read_text().strip().splitlines()), 3)
            json.loads(path.read_text().splitlines()[0])


# --------------------------------------------------------------------------- #
# Anthropic request shape — stubbed client, no network
# --------------------------------------------------------------------------- #


def _block(**kwargs):
    return SimpleNamespace(**kwargs)


def _response(content, stop_reason="tool_use", stop_details=None):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
    )


class StubClient:
    """Captures request payloads and replays queued responses."""

    def __init__(self, responses):
        self.requests = []
        self._responses = list(responses)
        outer = self

        class _Messages:
            async def create(self, **kwargs):
                outer.requests.append(kwargs)
                return outer._responses.pop(0)

        self.beta = SimpleNamespace(messages=_Messages())


class TestClaudePolicy(unittest.TestCase):
    def _policy(self, responses, config=None):
        return ClaudeComputerUsePolicy(
            1280, 720, config or PolicyConfig(), client=StubClient(responses)
        )

    def test_request_declares_the_computer_tool_and_betas(self):
        policy = self._policy([_response([
            _block(type="text", text="Clicking save."),
            _block(type="tool_use", id="tu_1", name="computer",
                   input={"action": "left_click", "coordinate": [10, 20]}),
        ])])
        frame = asyncio.run(MockComputer.settings_form().screenshot())
        decision = asyncio.run(policy.begin(TASK, frame))

        request = policy._client.requests[0]
        self.assertEqual(request["model"], "claude-opus-5")
        self.assertEqual(request["tools"][0]["name"], "computer")
        self.assertEqual(request["tools"][0]["display_width_px"], 1280)
        self.assertIn("computer-use-2025-11-24", request["betas"])
        self.assertEqual(request["output_config"], {"effort": "high"})
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        # Refusal fallback is opt-in-by-default so one classifier hit does not
        # kill a long rollout.
        self.assertEqual(request["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", request["betas"])
        # The system prompt is cached; the volatile frames sit after it.
        self.assertEqual(request["system"][0]["cache_control"], {"type": "ephemeral"})

        self.assertEqual(decision.action.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(decision.action.coordinate, (10, 20))
        self.assertEqual(decision.rationale, "Clicking save.")
        self.assertEqual(decision.usage["input_tokens"], 10)

    def test_fallback_can_be_disabled(self):
        policy = self._policy(
            [_response([_block(type="text", text="done")], stop_reason="end_turn")],
            PolicyConfig(enable_refusal_fallback=False),
        )
        frame = asyncio.run(MockComputer.settings_form().screenshot())
        asyncio.run(policy.begin(TASK, frame))
        request = policy._client.requests[0]
        self.assertNotIn("fallbacks", request)
        self.assertNotIn("server-side-fallback-2026-07-01", request["betas"])

    def test_text_only_response_ends_the_episode(self):
        policy = self._policy([
            _response([_block(type="text", text="All done.")], stop_reason="end_turn")
        ])
        frame = asyncio.run(MockComputer.settings_form().screenshot())
        decision = asyncio.run(policy.begin(TASK, frame))
        self.assertTrue(decision.is_terminal)
        self.assertEqual(decision.rationale, "All done.")

    def test_refusal_is_raised_before_content_is_read(self):
        policy = self._policy([_response(
            [], stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber", explanation="nope"),
        )])
        frame = asyncio.run(MockComputer.settings_form().screenshot())
        with self.assertRaises(RefusalError) as ctx:
            asyncio.run(policy.begin(TASK, frame))
        self.assertEqual(ctx.exception.category, "cyber")

    def test_refusal_surfaces_as_a_trajectory_status(self):
        policy = self._policy([_response([], stop_reason="refusal")])
        trajectory = asyncio.run(run_episode(
            TASK, MockComputer.settings_form(), policy, verifier=VERIFIER
        ))
        self.assertIs(trajectory.status, TrajectoryStatus.REFUSED)

    def test_tool_result_carries_the_new_frame(self):
        policy = self._policy([
            _response([_block(type="tool_use", id="tu_1", name="computer",
                              input={"action": "screenshot"})]),
            _response([_block(type="text", text="done")], stop_reason="end_turn"),
        ])
        env = MockComputer.settings_form()
        frame = asyncio.run(env.screenshot())
        asyncio.run(policy.begin(TASK, frame))
        asyncio.run(policy.observe(frame))

        result_block = _last_tool_result(policy._client.requests[1])
        self.assertEqual(result_block["type"], "tool_result")
        self.assertEqual(result_block["tool_use_id"], "tu_1")
        self.assertEqual(result_block["content"][0]["type"], "image")

    def test_errors_are_returned_to_the_model_not_raised(self):
        policy = self._policy([
            _response([_block(type="tool_use", id="tu_1", name="computer",
                              input={"action": "screenshot"})]),
            _response([_block(type="text", text="ok")], stop_reason="end_turn"),
        ])
        frame = asyncio.run(MockComputer.settings_form().screenshot())
        asyncio.run(policy.begin(TASK, frame))
        asyncio.run(policy.observe(frame, error="coordinate out of bounds"))

        result_block = _last_tool_result(policy._client.requests[1])
        self.assertTrue(result_block["is_error"])
        self.assertIn("coordinate out of bounds", result_block["content"][0]["text"])

    def test_observe_without_a_pending_action_is_an_error(self):
        policy = self._policy([])
        frame = asyncio.run(MockComputer.settings_form().screenshot())
        with self.assertRaises(RuntimeError):
            asyncio.run(policy.observe(frame))

    def test_stale_frames_are_pruned_from_history(self):
        turns = 6
        responses = [
            _response([_block(type="tool_use", id=f"tu_{i}", name="computer",
                              input={"action": "screenshot"})])
            for i in range(turns)
        ]
        policy = self._policy(responses, PolicyConfig(keep_frames=2))
        frame = asyncio.run(MockComputer.settings_form().screenshot())
        asyncio.run(policy.begin(TASK, frame))
        for _ in range(turns - 1):
            asyncio.run(policy.observe(frame))

        images = _count_images(policy._client.requests[-1]["messages"])
        self.assertEqual(images, 2)
        # Pruning replaces frames in place; the record of what happened stays.
        placeholders = json.dumps(
            policy._client.requests[-1]["messages"], default=str
        ).count("earlier screenshot omitted")
        self.assertEqual(placeholders, turns - 2)


class TestComputerUseAgent(unittest.TestCase):
    """The agent path: bus messages in, training data out."""

    def _run(self, bus=None):
        agent = ComputerUseAgent()
        if bus is not None:
            bus.register_agent(agent)
        result = asyncio.run(agent.execute(
            tasks=[(TASK, VERIFIER)],
            env_factory=MockComputer.settings_form,
            policy_factory=lambda env: ScriptedPolicy(GOOD_SCRIPT),
            group_size=2,
            reward_config=RewardConfig(optimal_steps=6),
        ))
        return agent, result

    def test_produces_every_training_shape(self):
        agent, result = self._run()
        self.assertIs(agent.status, AgentStatus.COMPLETED)
        self.assertEqual(result["report"]["episodes"], 2)
        self.assertEqual(result["report"]["success_rate"], 1.0)
        self.assertEqual(len(result["rollout_batch"].prompts), 1)
        self.assertEqual(len(result["training_examples"]), 1)
        self.assertEqual(len(agent.trajectories), 2)
        for key in ("preference_pairs", "step_preference_pairs", "trajectories"):
            self.assertIn(key, result)

    def test_narrates_progress_on_the_bus(self):
        bus = MessageBus(verbose=False)
        _, _ = self._run(bus)
        senders = {m.sender for m in bus.history}
        self.assertIn("Operator", senders)
        kinds = {m.msg_type for m in bus.history}
        self.assertIn(MessageType.DATA_SHARE, kinds)

    def test_requires_a_policy_factory(self):
        agent = ComputerUseAgent()
        with self.assertRaises(ValueError):
            asyncio.run(agent.run(tasks=[(TASK, VERIFIER)]))

    def test_requires_tasks(self):
        agent = ComputerUseAgent()
        with self.assertRaises(ValueError):
            asyncio.run(agent.run(policy_factory=lambda env: ScriptedPolicy(GOOD_SCRIPT)))


def _last_tool_result(request):
    """The most recent tool_result block in a captured request.

    The stub holds a live reference to the policy's message list, so indexing
    from the end picks up turns appended after the call. Search instead.
    """
    for message in reversed(request["messages"]):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return block
    raise AssertionError("no tool_result block in request")


def _count_images(messages):
    total = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                total += 1
            elif block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    total += sum(
                        1 for b in inner if isinstance(b, dict) and b.get("type") == "image"
                    )
    return total


if __name__ == "__main__":
    unittest.main()
