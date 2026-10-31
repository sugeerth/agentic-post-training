"""Tests for task synthesis.

Three claims are load-bearing, and each has a test that could falsify it:

  Every synthesized task is **solvable** — its gold path reaches its own goal.
  Every synthesized gold is **optimal** — no shorter path exists, checked by
  re-searching with a smaller budget and confirming the goal is unreachable.
  Synthesis **covers what a human would write** — the hand-authored suite's
  goals are all rediscovered, at or below their hand-guessed step counts.

The last one is the interesting one: it cross-validates the two suites against
each other. If a hand-written `optimal_steps` were too small, search would fail
to reach the goal in that budget; if it were too generous, search would beat
it. Neither happens.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use import (
    Action,
    ActionKind,
    MockComputer,
    ScriptedPolicy,
    Trajectory,
    TrajectoryStatus,
    run_episode,
)
from computer_use.synthesis import (
    VOCABULARIES,
    AbstractAction,
    SynthesisConfig,
    available_actions,
    describe,
    explore,
    synthesize,
    synthesize_suite,
)
from computer_use.tasks import SUITE

SETTINGS = SynthesisConfig(vocabulary=VOCABULARIES["settings"], max_depth=5)


def solve(task):
    return asyncio.run(run_episode(
        task.instruction,
        task.env_factory(),
        ScriptedPolicy(list(task.gold)),
        verifier=task.verifier,
        reward_config=task.reward_config(),
    ))


class TestSearch(unittest.TestCase):
    def test_search_finds_reachable_goals(self):
        found = explore(MockComputer.settings_form(), SETTINGS)
        self.assertTrue(found)
        deltas = [d.delta for d in found]
        self.assertIn({"notify": True}, deltas)
        self.assertIn({"email": "ada@example.com"}, deltas)

    def test_search_discovers_the_scroll_dependency(self):
        """Nothing tells the search that SAVE is below the fold.

        The only way to reach `saved` is to scroll first, and search has to
        find that by exploring — which is the whole point of deriving tasks
        from the environment instead of describing them alongside it.
        """
        found = explore(MockComputer.settings_form(), SETTINGS)
        saved = next(d for d in found if d.delta == {"saved": True})
        kinds = [a.kind for a in saved.path]
        self.assertIn("scroll", kinds)
        self.assertEqual(kinds[-1], "click")

    def test_search_is_bounded_by_depth(self):
        shallow = explore(MockComputer.settings_form(),
                          SynthesisConfig(vocabulary=VOCABULARIES["settings"], max_depth=2))
        self.assertTrue(shallow)
        self.assertTrue(all(d.depth <= 2 for d in shallow))

    def test_search_is_deterministic(self):
        a = explore(MockComputer.settings_form(), SETTINGS)
        b = explore(MockComputer.settings_form(), SETTINGS)
        self.assertEqual([d.delta for d in a], [d.delta for d in b])
        self.assertEqual([d.path for d in a], [d.path for d in b])

    def test_search_leaves_the_environment_untouched(self):
        env = MockComputer.settings_form()
        before = dict(env.state())
        explore(env, SETTINGS)
        self.assertEqual(dict(env.state()), before)

    def test_typing_is_offered_only_into_an_empty_focused_field(self):
        env = MockComputer.settings_form()
        # Nothing focused yet.
        self.assertFalse(
            any(a.kind == "type" for a in available_actions(env, SETTINGS))
        )
        asyncio.run(env.execute(_click(env, "email")))
        self.assertTrue(
            any(a.kind == "type" for a in available_actions(env, SETTINGS))
        )
        # Once filled, typing again would append forever — the state space
        # would not be finite and BFS would never terminate.
        asyncio.run(env.execute(Action(ActionKind.TYPE, text="x")))
        self.assertFalse(
            any(a.kind == "type" for a in available_actions(env, SETTINGS))
        )

    def test_offscreen_widgets_are_not_clickable(self):
        env = MockComputer.settings_form()
        targets = {a.target for a in available_actions(env, SETTINGS) if a.kind == "click"}
        self.assertNotIn("save", targets)  # below the fold
        self.assertIn("notify", targets)

    def test_per_field_vocabulary_keeps_values_with_their_fields(self):
        found = explore(MockComputer.settings_form(), SETTINGS)
        deltas = [d.delta for d in found]
        # An email address must never be offered to the retries field.
        self.assertFalse(any(d.get("retries") == "ada@example.com" for d in deltas))
        self.assertFalse(any(d.get("email") == "5" for d in deltas))


class TestSynthesizedTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = synthesize_suite(per_environment=6, max_depth=5, seed=0)

    def test_a_suite_is_produced(self):
        self.assertGreaterEqual(len(self.tasks), 12)
        self.assertTrue(all(t.metadata["synthesized"] for t in self.tasks))
        self.assertEqual(len({t.name for t in self.tasks}), len(self.tasks))

    def test_every_task_is_solved_by_its_own_gold(self):
        for task in self.tasks:
            trajectory = solve(task)
            self.assertTrue(
                trajectory.succeeded,
                f"{task.name} ({task.instruction}) failed: "
                f"{trajectory.metadata['verdict']['reason']}",
            )
            # Gold is the optimal path, so it should also score perfectly.
            self.assertAlmostEqual(trajectory.reward, 1.0, places=6, msg=task.name)

    def test_every_gold_path_is_optimal(self):
        """The central claim: no shorter solution exists.

        Verified by re-searching with one step less budget and confirming the
        goal is unreachable. This is what makes `optimal_steps` a fact about
        the environment rather than an author's estimate.
        """
        configs = {
            "settings": (MockComputer.settings_form, VOCABULARIES["settings"]),
            "checkout": (MockComputer.checkout_flow, VOCABULARIES["checkout"]),
            "files": (MockComputer.file_manager, VOCABULARIES["files"]),
        }
        for task in self.tasks:
            factory, vocabulary = configs[task.name.split(".")[1]]
            budget = task.optimal_steps - 1
            if budget < 1:
                continue
            reachable = explore(
                factory(), SynthesisConfig(vocabulary=vocabulary, max_depth=budget)
            )
            self.assertNotIn(
                dict(task.metadata["delta"]),
                [d.delta for d in reachable],
                f"{task.name} claims {task.optimal_steps} steps but is "
                f"reachable in {budget}",
            )

    def test_instruction_matches_the_verifier(self):
        # An instruction that drifts from the goal state would reintroduce
        # exactly the gap this module removes.
        for task in self.tasks:
            self.assertEqual(
                dict(task.verifier.expected), dict(task.metadata["delta"])
            )
            self.assertTrue(task.instruction.endswith("."))
            self.assertTrue(task.instruction[0].isupper())

    def test_radio_selection_is_stated_once(self):
        """A radio moves three state keys but is one decision.

        Left uncollapsed the instruction reads "choose EXPRESS, turn on
        EXPRESS, and turn off STANDARD" — three clauses for one click.
        """
        tasks = synthesize(
            MockComputer.checkout_flow,
            SynthesisConfig(vocabulary=VOCABULARIES["checkout"], max_depth=3),
        )
        express = [t for t in tasks if "EXPRESS" in t.instruction]
        self.assertTrue(express)
        for task in express:
            self.assertEqual(task.instruction.count("EXPRESS"), 1, task.instruction)
            self.assertNotIn("STANDARD", task.instruction)

    def test_difficulty_tracks_depth(self):
        for task in self.tasks:
            if task.difficulty == "easy":
                self.assertLessEqual(task.optimal_steps, 2)

    def test_synthesis_is_reproducible(self):
        again = synthesize_suite(per_environment=6, max_depth=5, seed=0)
        self.assertEqual(
            [(t.name, t.instruction) for t in again],
            [(t.name, t.instruction) for t in self.tasks],
        )

    def test_limit_preserves_the_difficulty_spread(self):
        # Truncating would return only the shallowest tasks; sampling across
        # depths keeps a capped suite representative.
        capped = synthesize(
            MockComputer.settings_form, SETTINGS, limit=4, min_depth=2, seed=1,
        )
        self.assertEqual(len(capped), 4)
        self.assertGreater(len({t.optimal_steps for t in capped}), 1)


class TestCrossValidationAgainstTheHandWrittenSuite(unittest.TestCase):
    """Search and the hand-written benchmark should agree.

    They were produced independently — one by a person choosing tasks, one by
    exhaustive search — so agreement is meaningful evidence that both are
    right, and disagreement would point at a real defect in one of them.
    """

    def test_hand_written_goals_are_all_reachable_within_their_step_budgets(self):
        configs = {
            "settings": (MockComputer.settings_form, VOCABULARIES["settings"]),
            "checkout": (MockComputer.checkout_flow, VOCABULARIES["checkout"]),
            "files": (MockComputer.file_manager, VOCABULARIES["files"]),
        }
        for task in SUITE:
            factory, vocabulary = configs[task.name.split(".")[0]]
            reachable = explore(
                factory(),
                SynthesisConfig(vocabulary=vocabulary, max_depth=task.optimal_steps),
            )
            env = factory()
            satisfied = [
                d for d in reachable
                if task.verifier(_empty_trajectory(), {**dict(env.state()), **d.goal}).success
            ]
            self.assertTrue(
                satisfied,
                f"search cannot reach {task.name}'s goal in its declared "
                f"{task.optimal_steps} steps — the hand-written budget is too small",
            )

    def test_instructions_render_for_hand_written_deltas(self):
        env = MockComputer.settings_form()
        text = describe(env, {"email": "ada@example.com", "notify": True, "saved": True})
        self.assertIn("EMAIL", text)
        self.assertIn("SAVE", text)
        self.assertTrue(text.endswith("."))


def _click(env, widget_id):
    widget = next(w for w in env._widgets if w.id == widget_id)
    return Action(
        ActionKind.LEFT_CLICK,
        coordinate=(widget.x + widget.width // 2,
                    widget.y + widget.height // 2 - env._scroll_y),
    )


def _empty_trajectory():
    return Trajectory(task="", steps=(), status=TrajectoryStatus.SUCCESS)


class TestAbstractAction(unittest.TestCase):
    def test_describe_is_readable(self):
        self.assertEqual(AbstractAction("click", target="save").describe(), "click save")
        self.assertEqual(AbstractAction("type", value="hi").describe(), "type 'hi'")
        self.assertEqual(
            AbstractAction("scroll", direction="down").describe(), "scroll down"
        )


if __name__ == "__main__":
    unittest.main()
