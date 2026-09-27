"""Tests for the policy that learns from this pipeline's own output.

The claim under test is narrow and falsifiable: a policy trained only on data
this framework generated, reading only pixels, solves tasks in applications it
has never seen — and it does so *because of the training*, not because the
feature set already encodes the answer.

So the load-bearing tests come in pairs. `test_training_beats_the_same_features
_untrained` is the whole result; without it, a high held-out score would be
equally consistent with the features doing all the work and the data being
worthless. And `test_the_policy_decides_from_the_frame_alone` is what makes the
score mean anything at all — a policy that peeks at ground truth would score
well while proving nothing about operating a GUI.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use import (
    ActionKind,
    RolloutConfig,
    Screenshot,
    ScriptedPolicy,
    run_episode,
)
from computer_use.learn import (
    Goal,
    LearnedPolicy,
    accuracy,
    closed_loop,
    examples_from,
    features,
    read_instruction,
    train,
    untrained,
)
from computer_use.perception import parse_screen
from computer_use.worlds import curriculum

CONFIG = RolloutConfig(store_frames=True, max_steps=24)


def demonstrate(tasks):
    """Roll out each task's searched solution, which is the training data."""
    return [
        asyncio.run(run_episode(
            task.instruction, task.env_factory(), ScriptedPolicy(list(task.gold)),
            verifier=task.verifier, reward_config=task.reward_config(), config=CONFIG,
        ))
        for task in tasks
    ]


class TestInstructions(unittest.TestCase):
    def test_reads_each_kind_of_goal(self):
        goals = read_instruction('Set CITY to "BERLIN" and turn on DARK MODE.')
        self.assertEqual([g.kind for g in goals], ["set", "toggle"])
        self.assertEqual(goals[0].value, "BERLIN")
        self.assertEqual(goals[1].phrase, "DARK MODE")

    def test_goals_come_back_in_the_order_they_are_written(self):
        goals = read_instruction('Choose CSV and press SAVE.')
        self.assertEqual([g.kind for g in goals], ["choose", "press"])

    def test_a_verb_inside_a_word_is_not_a_goal(self):
        # "press" lives inside "EXPRESS". Without a word boundary this invents
        # a goal to press a button called "- 1 DAY", and the episode fails on a
        # screen where everything the task asked for was right there.
        goals = read_instruction("Choose EXPRESS - 1 DAY and press PLACE ORDER.")
        self.assertEqual([g.kind for g in goals], ["choose", "press"])
        self.assertEqual(goals[0].phrase, "EXPRESS - 1 DAY")
        self.assertEqual(goals[1].phrase, "PLACE ORDER")


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        task = curriculum(range(1), per_world=1)[0]
        env = task.env_factory()
        cls.screen = parse_screen(asyncio.run(env.reset()).data)
        cls.element = cls.screen.elements[0]

    def test_no_feature_hands_over_the_answer(self):
        """The closest thing to a giveaway is token overlap.

        And overlap is just as high for the caption naming a field as for the
        field itself, so the model still has to learn which one to click.
        """
        goal = Goal("set", self.element.label or "ANYTHING", "x")
        names = set(features(goal, self.element, self.screen))
        self.assertFalse({n for n in names if "correct" in n or "target" in n})

    def test_vocabulary_is_confined_to_navigation(self):
        # Word identity is allowed for choosing a way off a screen and nowhere
        # else: for grounding it would let the model memorize labels from the
        # training apps instead of learning to read the one in front of it.
        grounding = features(Goal("set", "PHONE"), self.element, self.screen)
        navigating = features(Goal("navigate", "PHONE"), self.element, self.screen)
        self.assertFalse([n for n in grounding if n.startswith("word:")])
        if self.element.tokens:
            self.assertTrue([n for n in navigating if n.startswith("word:")])


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = curriculum(range(8), per_world=3)
        cls.examples = examples_from(demonstrate(cls.tasks))
        cls.model = train(cls.examples)

    def test_demonstrations_become_decisions(self):
        self.assertTrue(self.examples)
        self.assertTrue(all(e.usable for e in self.examples))
        # A decision with one candidate teaches nothing; these are real choices.
        self.assertGreater(
            sum(len(e.candidates) for e in self.examples) / len(self.examples), 2.0,
        )

    def test_training_fits_the_demonstrations(self):
        self.assertGreater(accuracy(self.model, self.examples), 0.8)

    def test_an_untrained_model_does_not(self):
        self.assertGreater(
            accuracy(self.model, self.examples),
            accuracy(untrained(self.model, seed=1), self.examples),
        )

    def test_it_learns_that_captions_are_not_controls(self):
        # The words "MAX RETRIES" above an input are not the input. Clicking
        # the caption is the single most natural mistake here, and the weights
        # should say so.
        self.assertLess(self.model.weights.get("kind:text", 0.0), 0.0)


class TestPolicy(unittest.TestCase):
    def test_the_policy_decides_from_the_frame_alone(self):
        """The whole benchmark rests on this.

        `state()` is ground truth — what the verifier checks against. A policy
        that reads it is grading its own homework, and every held-out number
        this module produces would be meaningless. The guarantee here is
        structural rather than a matter of discipline: the policy is handed
        nothing but a `Screenshot`, so it is given the frame and the PNG bytes
        in it and produces a real action from those alone.
        """
        task = curriculum(range(1), per_world=1)[0]
        env = task.env_factory()
        frame = asyncio.run(env.reset())
        model = train(examples_from(demonstrate([task])))

        detached = Screenshot(data=frame.data, width=frame.width, height=frame.height)
        decision = asyncio.run(LearnedPolicy(model=model).begin(task.instruction, detached))

        self.assertIsNotNone(decision.action, "decided nothing from the pixels")
        self.assertIn(decision.action.kind, (ActionKind.LEFT_CLICK, ActionKind.SCROLL))
        if decision.action.kind is ActionKind.LEFT_CLICK:
            self.assertIsNotNone(
                env._hit_test(*decision.action.coordinate),
                "clicked somewhere no control exists",
            )

    def test_it_goes_looking_instead_of_clicking_the_wrong_thing(self):
        scroller = next(
            t for t in curriculum(range(12), per_world=3)
            if "SCROLL" not in t.instruction and any(
                a.kind.value == "scroll" for a in t.gold
            )
        )
        model = train(examples_from(demonstrate(curriculum(range(8), per_world=3))))
        trajectory = asyncio.run(run_episode(
            scroller.instruction, scroller.env_factory(), LearnedPolicy(model=model),
            verifier=scroller.verifier, config=CONFIG,
        ))
        self.assertTrue(
            any(step.action.kind.value == "scroll" for step in trajectory.steps),
            "never scrolled, so it must have acted on something it could see",
        )


class TestClosedLoop(unittest.TestCase):
    def test_overlapping_worlds_are_refused(self):
        with self.assertRaises(ValueError):
            closed_loop(train_worlds=range(4), test_worlds=range(2, 6))

    def test_training_beats_the_same_features_untrained(self):
        """The result: generated apps, searched tasks, learned policy, new apps.

        Kept small enough to run in the suite. The full-size version is
        `agentic-gui learn`, which trains on thirty applications.
        """
        result = closed_loop(
            train_worlds=range(10), test_worlds=range(1000, 1004),
            per_world=2, controls=2,
        )
        self.assertGreater(result.test_tasks, 0)
        self.assertGreater(
            result.trained_rate, result.untrained_rate,
            f"training changed nothing: {result.trained_rate:.0%} trained vs "
            f"{result.untrained_rate:.0%} with random weights",
        )
        self.assertGreater(result.trained_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
