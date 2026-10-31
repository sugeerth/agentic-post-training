"""Tests for procedural world generation.

The generator's output is consumed by a search that assumes a well-formed GUI,
so the invariants here are the ones that would otherwise corrupt everything
downstream silently:

  Widgets never overlap — overlap makes hit-testing ambiguous, which makes
  every derived gold path unreliable without failing loudly.
  Labels are always renderable — an unrenderable label draws as filled blocks,
  which a vision model reads as noise rather than as a hard task.
  Every shipped world can host a task — a world nobody can act in contributes
  a zero to a benchmark and looks like agent failure.
  Splits hold out whole worlds — the entire point of the module.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use import ScriptedPolicy, run_episode
from computer_use._render import text_width
from computer_use.worlds import (
    DISTRACTOR_SUFFIXES,
    curriculum,
    generate,
    generate_many,
    is_viable,
    renderable,
    split,
)

SEEDS = range(12)


class TestGeneration(unittest.TestCase):
    def test_generation_is_deterministic_in_the_seed(self):
        a, b = generate(7), generate(7)
        self.assertEqual(a.spec, b.spec)
        self.assertEqual(a.widgets, b.widgets)
        self.assertEqual(dict(a.vocabulary), dict(b.vocabulary))

    def test_different_seeds_give_different_worlds(self):
        shapes = {generate(s).spec.summary() for s in range(20)}
        self.assertGreater(len(shapes), 3)

    def test_widgets_never_overlap(self):
        """Overlap would make hit-testing ambiguous and corrupt gold paths."""
        for seed in SEEDS:
            world = generate(seed)
            by_screen: dict[str, list] = {}
            for widget in world.widgets:
                by_screen.setdefault(widget.screen, []).append(widget)
            for screen, widgets in by_screen.items():
                for i, a in enumerate(widgets):
                    for b in widgets[i + 1:]:
                        self.assertFalse(
                            _overlaps(a, b),
                            f"world{seed:03d} {screen}: {a.id} overlaps {b.id}",
                        )

    def test_every_label_is_renderable(self):
        for seed in SEEDS:
            for widget in generate(seed).widgets:
                for text in (widget.label, widget.placeholder):
                    if text:
                        self.assertTrue(
                            renderable(text),
                            f"world{seed:03d}: {text!r} has no glyph",
                        )

    def test_widgets_fit_inside_the_content_area(self):
        for seed in SEEDS:
            world = generate(seed)
            for widget in world.widgets:
                self.assertGreaterEqual(widget.x, 0)
                self.assertLessEqual(widget.x + widget.width, 1280)
                self.assertLessEqual(
                    widget.y + widget.height, world.content_height,
                    f"world{seed:03d}: {widget.id} falls outside the page",
                )

    def test_ids_are_unique(self):
        for seed in SEEDS:
            ids = [w.id for w in generate(seed).widgets]
            self.assertEqual(len(ids), len(set(ids)), f"world{seed:03d}")

    def test_every_world_renders_a_real_png(self):
        for seed in (0, 3, 7):
            frame = asyncio.run(generate(seed).build().screenshot())
            self.assertTrue(frame.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_multi_screen_worlds_are_navigable(self):
        multi = [generate(s) for s in range(40)]
        multi = [w for w in multi if w.spec.screens > 1]
        self.assertTrue(multi, "no multi-screen world in 40 seeds")
        for world in multi:
            shows = {w.shows for w in world.widgets if w.shows}
            self.assertTrue(shows, f"{world.name} has extra screens but no way in")

    def test_scrolling_worlds_hide_their_action_below_the_fold(self):
        scrollers = [generate(s) for s in range(40)]
        scrollers = [w for w in scrollers if w.spec.scroll]
        self.assertTrue(scrollers, "no scrolling world in 40 seeds")
        for world in scrollers:
            self.assertGreater(world.content_height, 720)
            action = next(w for w in world.widgets if w.sets)
            self.assertGreater(action.y, 720, f"{world.name} action is not hidden")

    def test_every_shipped_world_can_host_a_task(self):
        worlds = generate_many(range(20))
        self.assertTrue(worlds)
        for world in worlds:
            self.assertTrue(is_viable(world), f"{world.name} has no reachable goal")

    def test_vocabulary_is_derived_from_the_world(self):
        # Values live with the fields that own them, so synthesis cannot
        # generate type-inappropriate goals.
        for seed in SEEDS:
            world = generate(seed)
            field_ids = {w.id for w in world.widgets if w.kind == "field"}
            self.assertEqual(set(world.vocabulary), field_ids, world.name)

    def test_world_factory_produces_independent_environments(self):
        world = generate(1)
        factory = world.factory()
        first, second = factory(), factory()
        asyncio.run(first.execute(_first_click(first)))
        self.assertNotEqual(dict(first.state()), dict(second.state()))


class TestCurriculum(unittest.TestCase):
    # Deliberately wide. At eight worlds this suite was green while one world
    # in five was quietly scoring its own gold below 1.0 — the multi-screen
    # layouts that expose it only start appearing past seed 16.
    @classmethod
    def setUpClass(cls):
        cls.tasks = curriculum(range(30), per_world=2)

    def test_tasks_are_produced_across_worlds(self):
        self.assertGreaterEqual(len(self.tasks), 12)
        self.assertGreater(len({t.metadata["world_seed"] for t in self.tasks}), 3)

    def test_every_generated_task_is_solved_by_its_generated_gold(self):
        """End to end with no human in the loop at any stage.

        The app was generated, the task was searched out of it, the solution is
        the search path, and the check is the goal state. If any link were
        wrong this would fail.
        """
        for task in self.tasks:
            trajectory = asyncio.run(run_episode(
                task.instruction,
                task.env_factory(),
                ScriptedPolicy(list(task.gold)),
                verifier=task.verifier,
                reward_config=task.reward_config(),
            ))
            self.assertTrue(
                trajectory.succeeded,
                f"{task.name} ({task.instruction}): "
                f"{trajectory.metadata['verdict']['reason']}",
            )
            self.assertAlmostEqual(trajectory.reward, 1.0, places=6, msg=task.name)

    def test_tasks_carry_their_world_of_origin(self):
        for task in self.tasks:
            self.assertIn("world_seed", task.metadata)
            self.assertTrue(task.name.startswith("world"))

    def test_difficulty_is_monotone_in_depth(self):
        # A deeper task must never be labeled easier than a shallower one.
        order = {"easy": 0, "medium": 1, "hard": 2}
        by_depth: dict[int, set[str]] = {}
        for task in self.tasks:
            by_depth.setdefault(task.optimal_steps, set()).add(task.difficulty)
        floors = {d: min(order[x] for x in v) for d, v in by_depth.items()}
        for shallow, deep in zip(sorted(floors), sorted(floors)[1:], strict=False):
            self.assertLessEqual(
                floors[shallow], floors[deep],
                f"depth {deep} is ranked easier than depth {shallow}",
            )


class TestHardMode(unittest.TestCase):
    """Distractors exist to stop word-matching from being a whole strategy.

    An app with both "EMAIL" and "EMAIL BACKUP" cannot be operated by finding
    the goal's words on the screen, because every one of them appears on two
    controls. That is what makes a held-out score mean something once the easy
    distribution saturates.
    """

    SEEDS = range(20)

    def test_hard_worlds_carry_near_duplicate_captions(self):
        twinned = 0
        for seed in self.SEEDS:
            labels = [w.label for w in generate(seed, hard=True).widgets if w.label]
            for label in labels:
                suffixed = any(
                    other == f"{label} {suffix}"
                    for other in labels for suffix in DISTRACTOR_SUFFIXES
                )
                if suffixed:
                    twinned += 1
                    break
        self.assertGreater(twinned, len(self.SEEDS) // 2)

    def test_easy_worlds_do_not(self):
        # Checked against the generated suffixes rather than any shared prefix:
        # "EMAIL" and "EMAIL ME ON FAILURE" are both in the base vocabulary and
        # coexist honestly, which is a different thing from a manufactured twin.
        for seed in self.SEEDS:
            for widget in generate(seed).widgets:
                twinned = [
                    suffix for suffix in DISTRACTOR_SUFFIXES
                    if widget.label.endswith(f" {suffix}")
                ]
                self.assertFalse(twinned, f"world{seed:03d}: {widget.label!r}")

    def test_a_twin_is_never_twinned_again(self):
        # "EMAIL BACKUP ALERTS" is not a harder screen, it is an incoherent
        # one, and no real form is laid out that way.
        for seed in self.SEEDS:
            for widget in generate(seed, hard=True).widgets:
                doubled = [
                    a for a in DISTRACTOR_SUFFIXES
                    for b in DISTRACTOR_SUFFIXES
                    if f"{a} {b}" in widget.label
                ]
                self.assertFalse(doubled, f"world{seed:03d}: {widget.label!r}")

    def test_the_decoy_action_is_wrong_rather_than_harmless(self):
        # It sets a real but different flag, so pressing it is a mistake the
        # verifier can see. A decoy that did nothing would be free to click.
        found = False
        for seed in self.SEEDS:
            world = generate(seed, hard=True)
            decoys = [w for w in world.widgets if w.id.endswith("_go_decoy")]
            for decoy in decoys:
                found = True
                self.assertTrue(decoy.sets)
                self.assertNotEqual(decoy.sets, world.spec.flag)
        self.assertTrue(found, "no decoy action in any hard world")

    def test_ids_stay_unique_with_distractors(self):
        for seed in self.SEEDS:
            ids = [w.id for w in generate(seed, hard=True).widgets]
            self.assertEqual(len(ids), len(set(ids)), f"world{seed:03d}")

    def test_a_button_is_wide_enough_for_its_caption(self):
        """A control whose own text runs off it cannot be read from a frame.

        Distractor labels are longer than the ones the fixed button width was
        chosen for, so "PLACE ORDER LATER" lost its final letter off the right
        edge — unreadable for reasons that have nothing to do with difficulty.
        """
        for seed in self.SEEDS:
            for widget in generate(seed, hard=True).widgets:
                if widget.kind != "button":
                    continue
                self.assertGreaterEqual(
                    widget.width, text_width(widget.label, 3),
                    f"world{seed:03d}: {widget.label!r} overflows its button",
                )

    def test_every_hard_task_is_solved_by_its_generated_gold(self):
        tasks = curriculum(range(12), per_world=2, hard=True)
        self.assertTrue(tasks)
        for task in tasks:
            trajectory = asyncio.run(run_episode(
                task.instruction, task.env_factory(), ScriptedPolicy(list(task.gold)),
                verifier=task.verifier, reward_config=task.reward_config(),
            ))
            self.assertTrue(trajectory.succeeded, f"{task.name}: {task.instruction}")
            self.assertAlmostEqual(trajectory.reward, 1.0, places=6, msg=task.name)

    def test_hard_tasks_are_not_satisfied_by_doing_nothing(self):
        for task in curriculum(range(6), per_world=2, hard=True):
            idle = asyncio.run(run_episode(
                task.instruction, task.env_factory(), ScriptedPolicy([]),
                verifier=task.verifier,
            ))
            self.assertFalse(idle.succeeded, f"{task.name} passes trivially")


class TestSplit(unittest.TestCase):
    def test_split_holds_out_whole_worlds(self):
        train, test = split(train=range(0, 6), test=range(50, 54), per_world=2)
        train_worlds = {t.metadata["world_seed"] for t in train}
        test_worlds = {t.metadata["world_seed"] for t in test}
        self.assertTrue(train and test)
        self.assertFalse(train_worlds & test_worlds)

    def test_overlapping_splits_are_rejected(self):
        # A split that shares a world leaks the layout, the labels, and where
        # the button is — exactly what it exists to hold out.
        with self.assertRaises(ValueError):
            split(train=range(0, 5), test=range(3, 8))


def _overlaps(a, b) -> bool:
    return not (
        a.x + a.width <= b.x
        or b.x + b.width <= a.x
        or a.y + a.height <= b.y
        or b.y + b.height <= a.y
    )


def _first_click(env):
    from computer_use import Action, ActionKind

    widget = next(w for w in env._widgets if w.kind in ("field", "checkbox", "radio"))
    return Action(
        ActionKind.LEFT_CLICK,
        coordinate=(widget.x + widget.width // 2, widget.y + widget.height // 2),
    )


if __name__ == "__main__":
    unittest.main()
