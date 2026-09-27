"""Tests for the GUI benchmark: the task suite, the metrics, and the evaluator.

The load-bearing test here is `test_every_task_is_solvable_by_its_gold_script`.
A benchmark whose reference solutions have rotted is worse than no benchmark —
it reports failures that belong to the harness and blames the agent. That test
(and the CI step that mirrors it) is what makes a score trustworthy.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use import (
    Action,
    ActionKind,
    MockComputer,
    NoisyPolicy,
    ScriptedPolicy,
    run_episode,
)
from computer_use.benchmark import (
    GUIBenchEvaluator,
    format_report,
    gold_factory,
    noisy_factory,
    run_benchmark,
)
from computer_use.cli import main as gui_main
from computer_use.metrics import (
    pass_at_k,
    summarize,
    summarize_task,
    wilson_interval,
)
from computer_use.tasks import DIFFICULTIES, SUITE, get_task, suite
from core.registry import get_evaluator, list_evaluators


class TestSuite(unittest.TestCase):
    def test_every_task_is_solvable_by_its_gold_script(self):
        """The suite's own reference solutions must still work.

        Without this, a geometry change in the mock silently turns the whole
        benchmark into a measurement of the harness rather than the agent.
        """
        async def check():
            for task in SUITE:
                trajectory = await run_episode(
                    task.instruction,
                    task.env_factory(),
                    ScriptedPolicy(list(task.gold)),
                    verifier=task.verifier,
                    reward_config=task.reward_config(),
                )
                self.assertTrue(
                    trajectory.succeeded,
                    f"{task.name} failed its own gold script: "
                    f"{trajectory.metadata['verdict']['reason']}",
                )
                # Gold is the reference, so it should also be efficient and
                # perfectly grounded — a sloppy reference sets a sloppy bar.
                self.assertAlmostEqual(trajectory.reward, 1.0, places=6, msg=task.name)

        asyncio.run(check())

    def test_gold_matches_the_declared_step_budget(self):
        for task in SUITE:
            effective = sum(1 for a in task.gold if not a.is_read_only)
            self.assertLessEqual(
                effective, task.optimal_steps,
                f"{task.name}: gold takes {effective} effective steps but "
                f"optimal_steps is {task.optimal_steps}",
            )

    def test_task_names_are_unique(self):
        names = [t.name for t in SUITE]
        self.assertEqual(len(names), len(set(names)))

    def test_every_difficulty_tier_is_populated(self):
        tiers = {t.difficulty for t in SUITE}
        self.assertEqual(tiers, set(DIFFICULTIES))

    def test_filtering_by_difficulty(self):
        easy = suite(difficulty="easy")
        self.assertTrue(easy)
        self.assertTrue(all(t.difficulty == "easy" for t in easy))
        self.assertEqual(len(suite(difficulty=["easy", "hard"])),
                         len(easy) + len(suite(difficulty="hard")))

    def test_filtering_by_tag_is_a_disjunction(self):
        nav = suite(tags=["navigation"])
        self.assertTrue(all("navigation" in t.tags for t in nav))
        combined = suite(tags=["navigation", "toggle"])
        self.assertGreaterEqual(len(combined), len(nav))

    def test_filters_compose(self):
        selected = suite(difficulty="hard", tags=["navigation"])
        for task in selected:
            self.assertEqual(task.difficulty, "hard")
            self.assertIn("navigation", task.tags)

    def test_unknown_filters_are_rejected(self):
        with self.assertRaises(ValueError):
            suite(difficulty="trivial")
        with self.assertRaises(ValueError):
            suite(names=["nope.nope"])
        with self.assertRaises(KeyError):
            get_task("nope.nope")

    def test_get_task_round_trips(self):
        self.assertIs(get_task("files.rename"), get_task("files.rename"))
        self.assertEqual(get_task("files.rename").difficulty, "hard")

    def test_verifiers_reject_the_untouched_environment(self):
        # A task whose verifier passes on a fresh environment measures nothing.
        for task in SUITE:
            env = task.env_factory()
            empty = run_episode(task.instruction, env, ScriptedPolicy([]),
                                verifier=task.verifier)
            trajectory = asyncio.run(empty)
            self.assertFalse(trajectory.succeeded, f"{task.name} passes trivially")


class TestNewScenarios(unittest.TestCase):
    def click(self, x, y):
        return Action(ActionKind.LEFT_CLICK, coordinate=(x, y))

    def test_widgets_on_other_screens_are_unreachable(self):
        env = MockComputer.checkout_flow()
        # PLACE ORDER lives on the payment screen; its coordinates hit nothing
        # while the cart is showing.
        asyncio.run(env.execute(self.click(180, 304)))
        self.assertFalse(env.state()["ordered"])
        self.assertEqual(env.state()["screen"], "main")

    def test_navigation_switches_screens(self):
        env = MockComputer.checkout_flow()
        asyncio.run(env.execute(self.click(160, 354)))
        self.assertEqual(env.state()["screen"], "shipping")

    def test_radio_group_is_single_choice(self):
        env = MockComputer.checkout_flow()
        asyncio.run(env.execute(self.click(160, 354)))
        self.assertEqual(env.state()["speed"], "standard")  # default
        asyncio.run(env.execute(self.click(76, 236)))       # express
        self.assertEqual(env.state()["speed"], "express")
        self.assertFalse(env.state()["standard"])

    def test_navigation_resets_the_viewport(self):
        env = MockComputer.file_manager()
        asyncio.run(env.execute(Action(
            ActionKind.SCROLL, coordinate=(640, 400),
            scroll_direction="down", scroll_amount=2,
        )))
        asyncio.run(env.execute(self.click(305, 342)))  # DELETE -> confirm
        self.assertEqual(env.state()["scroll_y"], 0)

    def test_destructive_action_needs_confirmation(self):
        env = MockComputer.file_manager()
        asyncio.run(env.execute(self.click(74, 219)))   # select notes.txt
        asyncio.run(env.execute(self.click(305, 342)))  # DELETE
        self.assertFalse(env.state()["deleted"], "delete fired without confirming")
        asyncio.run(env.execute(self.click(305, 252)))  # confirm
        self.assertTrue(env.state()["deleted"])


class TestMetrics(unittest.TestCase):
    def test_pass_at_1_equals_the_success_rate(self):
        for n, c in ((8, 4), (10, 3), (5, 5), (6, 0)):
            self.assertAlmostEqual(pass_at_k(n, c, 1), c / n, places=9)

    def test_pass_at_k_against_hand_computed_values(self):
        # 1 - C(7,2)/C(8,2) = 1 - 21/28
        self.assertAlmostEqual(pass_at_k(8, 1, 2), 0.25, places=9)
        # n - c < k, so at least one success is guaranteed in any k-subset
        self.assertEqual(pass_at_k(8, 5, 4), 1.0)
        self.assertEqual(pass_at_k(4, 0, 4), 0.0)

    def test_pass_at_k_is_monotonic_in_k(self):
        values = [pass_at_k(16, 5, k) for k in range(1, 12)]
        self.assertEqual(values, sorted(values))

    def test_pass_at_k_rejects_impossible_arguments(self):
        with self.assertRaises(ValueError):
            pass_at_k(4, 2, 8)   # k > n
        with self.assertRaises(ValueError):
            pass_at_k(4, 9, 2)   # more successes than attempts

    def test_wilson_interval_brackets_the_estimate(self):
        low, high = wilson_interval(6, 10)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)

    def test_wilson_interval_stays_in_range_at_the_extremes(self):
        # The normal approximation returns >1.0 here; Wilson does not.
        low, high = wilson_interval(10, 10)
        self.assertLessEqual(high, 1.0)
        self.assertGreater(low, 0.5)
        low, high = wilson_interval(0, 10)
        self.assertGreaterEqual(low, 0.0)
        self.assertLess(high, 0.5)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_wilson_interval_narrows_with_more_samples(self):
        narrow = wilson_interval(80, 100)
        wide = wilson_interval(8, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_summary_reports_only_supported_k_values(self):
        trajectories = _trajectories(successes=2, failures=2)
        result = summarize_task("t", "easy", trajectories, k_values=(1, 2, 4, 8))
        # 4 attempts cannot support pass@8.
        self.assertEqual(sorted(result.pass_at), [1, 2, 4])

    def test_reliability_gap_measures_retry_headroom(self):
        result = summarize_task("t", "easy", _trajectories(2, 2), k_values=(1, 4))
        self.assertGreater(result.reliability_gap, 0)
        perfect = summarize_task("t", "easy", _trajectories(4, 0), k_values=(1, 4))
        self.assertEqual(perfect.reliability_gap, 0.0)

    def test_suite_pass_at_k_averages_tasks_rather_than_pooling(self):
        # A task never solved must drag the suite number down, not be masked by
        # a task solved every time.
        solved = summarize_task("a", "easy", _trajectories(4, 0), k_values=(1,))
        unsolved = summarize_task("b", "hard", _trajectories(0, 4), k_values=(1,))
        report = summarize([solved, unsolved])
        self.assertAlmostEqual(report.pass_at[1], 0.5)

    def test_report_rolls_up_and_serializes(self):
        report = summarize([
            summarize_task("a", "easy", _trajectories(3, 1), k_values=(1, 2)),
            summarize_task("b", "hard", _trajectories(1, 3), k_values=(1, 2)),
        ])
        self.assertEqual(report.episodes, 8)
        self.assertEqual(report.successes, 4)
        self.assertAlmostEqual(report.success_rate, 0.5)
        self.assertEqual(report.by_difficulty()["easy"]["success_rate"], 0.75)
        payload = report.to_dict()
        json.dumps(payload)  # must be serializable
        self.assertEqual(len(payload["tasks"]), 2)

    def test_empty_report_is_safe(self):
        report = summarize([])
        self.assertEqual(report.episodes, 0)
        self.assertEqual(report.success_rate, 0.0)
        self.assertEqual(report.mean_reward, 0.0)


class TestNoisyPolicy(unittest.TestCase):
    def test_noise_is_reproducible_for_a_seed(self):
        task = get_task("settings.email")

        async def run(seed):
            return await run_episode(
                task.instruction, task.env_factory(),
                NoisyPolicy(list(task.gold), miss_rate=0.5, seed=seed),
                verifier=task.verifier, reward_config=task.reward_config(),
            )

        a, b = asyncio.run(run(7)), asyncio.run(run(7))
        self.assertEqual(a.action_sequence(), b.action_sequence())

    def test_different_seeds_produce_different_runs(self):
        task = get_task("settings.email")

        async def run(seed):
            return await run_episode(
                task.instruction, task.env_factory(),
                NoisyPolicy(list(task.gold), miss_rate=0.9, seed=seed),
                verifier=task.verifier, reward_config=task.reward_config(),
            )

        sequences = {tuple(asyncio.run(run(s)).action_sequence()) for s in range(6)}
        self.assertGreater(len(sequences), 1)

    def test_zero_miss_rate_reproduces_gold(self):
        task = get_task("checkout.express")
        trajectory = asyncio.run(run_episode(
            task.instruction, task.env_factory(),
            NoisyPolicy(list(task.gold), miss_rate=0.0),
            verifier=task.verifier, reward_config=task.reward_config(),
        ))
        self.assertTrue(trajectory.succeeded)

    def test_noisy_factory_varies_the_seed_per_attempt(self):
        # Identical seeds across a group would give zero variance, which makes
        # pass@k meaningless and produces no preference pairs.
        task = get_task("settings.email")
        factory = noisy_factory(task, miss_rate=0.9, seed=0)
        env = task.env_factory()
        first, second = factory(env), factory(env)
        self.assertNotEqual(first._seed, second._seed)


class TestBenchmark(unittest.TestCase):
    def test_gold_run_sweeps_the_suite(self):
        report, trajectories = asyncio.run(run_benchmark(attempts=2))
        self.assertEqual(report.episodes, len(SUITE) * 2)
        self.assertEqual(report.success_rate, 1.0)
        self.assertEqual(len(trajectories), len(SUITE) * 2)
        self.assertEqual(report.metadata["policy"], "gold")

    def test_noisy_run_lands_between_the_extremes(self):
        tasks = suite(names=["settings.email", "files.rename"])
        report, _ = asyncio.run(run_benchmark(
            tasks, noisy_factory(tasks[0], miss_rate=0.6, seed=3), attempts=6,
        ))
        self.assertGreater(report.episodes, 0)
        self.assertLessEqual(report.success_rate, 1.0)
        self.assertGreaterEqual(report.success_rate, 0.0)

    def test_explicit_factory_is_used(self):
        tasks = suite(names=["files.select"])
        report, _ = asyncio.run(run_benchmark(
            tasks, gold_factory(tasks[0]), attempts=3,
        ))
        self.assertEqual(report.success_rate, 1.0)
        self.assertNotIn("policy", report.metadata)

    def test_format_report_renders_without_ansi(self):
        report, _ = asyncio.run(run_benchmark(suite(difficulty="easy"), attempts=2))
        text = format_report(report, color=False)
        self.assertNotIn("\033", text)
        self.assertIn("files.select", text)
        self.assertIn("pass@k", text)


class TestRegisteredEvaluator(unittest.TestCase):
    def test_gui_bench_is_in_the_registry(self):
        self.assertIn("gui_bench", list_evaluators())
        self.assertIs(get_evaluator("gui_bench"), GUIBenchEvaluator)

    def test_evaluate_returns_a_scored_result_with_a_confidence_interval(self):
        evaluator = GUIBenchEvaluator(tasks=suite(difficulty="easy"), attempts=2)
        result = evaluator.evaluate()
        self.assertEqual(result.metric_name, "gui_bench")
        self.assertEqual(result.value, 100.0)
        self.assertEqual(result.n, len(suite(difficulty="easy")) * 2)
        # The framework's other evaluators leave these None.
        self.assertIsNotNone(result.ci_low)
        self.assertIsNotNone(result.ci_high)
        self.assertLessEqual(result.ci_low, result.value)
        self.assertGreaterEqual(result.ci_high, result.value)

    def test_evaluate_keeps_the_episodes_for_training(self):
        evaluator = GUIBenchEvaluator(tasks=suite(names=["files.select"]), attempts=3)
        evaluator.evaluate()
        self.assertEqual(len(evaluator.last_trajectories), 3)
        self.assertIsNotNone(evaluator.last_report)

    def test_evaluate_rejects_an_unusable_model_argument(self):
        with self.assertRaises(TypeError):
            GUIBenchEvaluator(attempts=1).evaluate(model=42)

    def test_async_callers_use_evaluate_async(self):
        # Every agent in this framework is async, so this is the path that
        # actually gets used.
        async def from_an_agent():
            evaluator = GUIBenchEvaluator(
                tasks=suite(names=["files.select"]), attempts=2
            )
            return await evaluator.evaluate_async()

        result = asyncio.run(from_an_agent())
        self.assertEqual(result.value, 100.0)
        self.assertEqual(result.n, 2)

    def test_sync_evaluate_refuses_inside_an_event_loop(self):
        # Rather than raising an opaque asyncio error, or blocking the caller's
        # loop for the length of a benchmark run.
        async def from_an_agent():
            with self.assertRaises(RuntimeError) as ctx:
                GUIBenchEvaluator(attempts=1).evaluate()
            return str(ctx.exception)

        message = asyncio.run(from_an_agent())
        self.assertIn("evaluate_async", message)


class TestCLI(unittest.TestCase):
    def test_tasks_json_lists_the_suite(self):
        code = gui_main(["tasks", "--json"])
        self.assertEqual(code, 0)

    def test_bench_runs_and_exits_clean(self):
        code = gui_main([
            "bench", "--policy", "gold", "--attempts", "2",
            "--difficulty", "easy", "--json",
        ])
        self.assertEqual(code, 0)

    def test_unknown_task_name_exits_with_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            gui_main(["bench", "--task", "not.a.task"])
        self.assertEqual(ctx.exception.code, 2)

    def test_worlds_benchmarks_generated_applications(self):
        code = gui_main([
            "bench", "--worlds", "2", "--per-environment", "1",
            "--policy", "gold", "--attempts", "1", "--json",
        ])
        self.assertEqual(code, 0)

    def test_held_out_worlds_are_disjoint_from_training_worlds(self):
        """The flag has to actually change which apps you run on."""
        from computer_use.cli import _select

        def seeds(held_out: bool) -> set[int]:
            args = argparse.Namespace(
                worlds=3, held_out=held_out, per_environment=1, max_depth=4,
                min_depth=2, seed=0, difficulty=None, tags=None, task=None,
                synthetic=False, hard=False,
            )
            return {t.metadata["world_seed"] for t in _select(args)}

        self.assertFalse(seeds(held_out=False) & seeds(held_out=True))

    def test_collect_writes_a_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "traj.jsonl"
            code = gui_main([
                "collect", "--policy", "noisy", "--attempts", "4",
                "--task", "settings.email", "--miss-rate", "0.6",
                "--out", str(out), "--json",
            ])
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())
            self.assertEqual(len(out.read_text().strip().splitlines()), 4)


def _trajectories(successes: int, failures: int):
    """Minimal scored trajectories, for exercising the metrics directly."""
    from computer_use.types import Trajectory, TrajectoryStatus

    made = []
    for _ in range(successes):
        made.append(Trajectory(
            task="t", steps=(), status=TrajectoryStatus.SUCCESS, reward=1.0,
            metadata={"stats": {"grounding_rate": 1.0}},
        ))
    for _ in range(failures):
        made.append(Trajectory(
            task="t", steps=(), status=TrajectoryStatus.FAILURE, reward=0.1,
            metadata={"stats": {"grounding_rate": 0.5}},
        ))
    return made


if __name__ == "__main__":
    unittest.main()
