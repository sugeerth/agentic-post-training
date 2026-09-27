"""Tests for learning from practice rather than from demonstrations.

The interesting risk here is not that the loop fails — it is that it *appears*
to work. A self-improvement loop can raise every number it can see while
getting no better at anything, by filling its training pool with the tasks it
already solves. So the assertions split in two:

  What the loop can see (practice pass rate, training accuracy) is never
  asserted as evidence of improvement on its own.

  What it cannot see (held-out applications, scored by a verifier the policy
  never reads) is where improvement has to show up.

`test_practice_improves_held_out_accuracy` is the result.
`test_a_stalled_loop_is_reported_rather_than_hidden` is the guard that makes
the result trustworthy, since a loop that cannot detect its own stall will
report one anyway.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use import RolloutConfig, ScriptedPolicy, run_episode
from computer_use.evolve import EvolveReport, Round, evolve, fork_supervision, harvest
from computer_use.learn import LearnedPolicy, examples_from, train
from computer_use.worlds import curriculum

CONFIG = RolloutConfig(store_frames=True, max_steps=24)


def episode(task, policy):
    return asyncio.run(run_episode(
        task.instruction, task.env_factory(), policy,
        verifier=task.verifier, reward_config=task.reward_config(), config=CONFIG,
    ))


class TestForkPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = curriculum(range(1), per_world=3)[-1]
        cls.good = episode(cls.task, ScriptedPolicy(list(cls.task.gold)))

    def test_a_success_and_a_failure_yield_a_correction(self):
        # Break the gold script at its first click: the run now fails, and the
        # step where it stopped agreeing with the successful run is exactly the
        # decision worth teaching.
        broken = []
        swapped = False
        for action in self.task.gold:
            if action.kind.value == "left_click" and not swapped:
                swapped = True
                broken.append(action.__class__(action.kind, coordinate=(1200, 700)))
                continue
            broken.append(action)
        bad = episode(self.task, ScriptedPolicy(broken))

        self.assertTrue(self.good.succeeded)
        if bad.succeeded:
            self.skipTest("the perturbation did not change the outcome")
        corrections = fork_supervision(self.good, bad)
        self.assertTrue(corrections, "no correction recovered from the divergence")
        self.assertTrue(all(c.usable for c in corrections))

    def test_two_successes_teach_no_correction(self):
        """A correction needs something to correct.

        Between two runs that both worked, the difference is style. Mining it
        for supervision invents a mistake that was never made.
        """
        self.assertEqual(fork_supervision(self.good, self.good), [])

    def test_failures_alone_produce_nothing(self):
        empty = episode(self.task, ScriptedPolicy([]))
        self.assertFalse(empty.succeeded)
        decisions, from_success, from_forks = harvest([empty, empty])
        self.assertEqual((decisions, from_success, from_forks), ([], 0, 0))


class TestGroups(unittest.TestCase):
    def test_a_deterministic_policy_makes_a_pointless_group(self):
        """Why sampling exists at all.

        Rolled out with temperature 0, every attempt is the same attempt, so a
        group of eight carries exactly as much information as one — and a loop
        built on it can never find a disagreement to learn from.
        """
        task = curriculum(range(1), per_world=2)[-1]
        model = train(examples_from([episode(task, ScriptedPolicy(list(task.gold)))]))
        cold = [
            [s.action.to_tool_input() for s in episode(
                task, LearnedPolicy(model=model, temperature=0.0, seed=n)).steps]
            for n in range(3)
        ]
        self.assertEqual(cold[0], cold[1])
        self.assertEqual(cold[1], cold[2])

    def test_sampling_makes_attempts_differ(self):
        task = curriculum(range(4), per_world=3)[-1]
        model = train(examples_from([episode(task, ScriptedPolicy(list(task.gold)))]))
        warm = {
            tuple(s.action.to_tool_input()["action"] + str(s.action.coordinate)
                  for s in episode(
                      task, LearnedPolicy(model=model, temperature=2.0, seed=n)).steps)
            for n in range(8)
        }
        self.assertGreater(len(warm), 1, "sampling produced identical attempts")


class TestSplits(unittest.TestCase):
    def test_practising_on_the_test_apps_is_refused(self):
        with self.assertRaises(ValueError):
            evolve(seed_worlds=range(2), practice_worlds=range(2, 6),
                   test_worlds=range(4, 8), rounds=1)

    def test_demonstrating_a_practice_app_is_refused(self):
        # The whole claim is that practice apps are never demonstrated.
        with self.assertRaises(ValueError):
            evolve(seed_worlds=range(4), practice_worlds=range(2, 8),
                   test_worlds=range(1000, 1004), rounds=1)


class TestReporting(unittest.TestCase):
    def _report(self, practice, held):
        report = EvolveReport(held_out_total=10)
        for index, (p, h) in enumerate(zip(practice, held, strict=True)):
            report.rounds.append(Round(
                index=index, attempts=100, solved=int(p * 100), from_success=1,
                from_forks=0, pool=1, train_accuracy=1.0, held_out=h,
                held_out_total=10,
            ))
        return report

    def test_a_stalled_loop_is_reported_rather_than_hidden(self):
        # Practice climbing while held-out accuracy sits still is the documented
        # failure of verifier-in-the-loop training: the pool fills with what the
        # policy already solves and the visible number rises on its own.
        stalled = self._report([0.4, 0.7, 0.9], [5, 5, 5])
        self.assertTrue(stalled.stalled)
        self.assertIn("feeding on what it already solves", format_notes(stalled))

    def test_real_improvement_is_not_flagged_as_a_stall(self):
        improving = self._report([0.4, 0.6, 0.8], [5, 7, 9])
        self.assertFalse(improving.stalled)


def format_notes(report):
    from computer_use.evolve import format_report

    return format_report(report)


class TestLoop(unittest.TestCase):
    def test_practice_improves_held_out_accuracy(self):
        """The result: better on new apps, with no new demonstrations.

        Three demonstrated applications, the rest practised blind. Anything the
        policy learns beyond round zero came from its own attempts, filtered by
        a verifier it cannot read.
        """
        report = evolve(
            seed_worlds=range(2), practice_worlds=range(2, 10),
            test_worlds=range(1000, 1004), rounds=2, group_size=3,
            per_world=2, temperature=1.0,
        )
        self.assertEqual(len(report.rounds), 3)
        self.assertGreater(report.rounds[1].attempts, 0)
        self.assertGreaterEqual(
            report.best, report.start,
            "practice made the policy worse on held-out applications",
        )
        self.assertGreater(report.rounds[-1].pool, report.rounds[0].pool)


if __name__ == "__main__":
    unittest.main()
