"""Procedurally generated GUIs — so you can hold out an *application*, not a task.

`computer_use.synthesis` removed the human from writing tasks, but it still
searches environments somebody hand-built. That leaves the last and worst
version of the overfitting problem intact: an agent evaluated on tasks drawn
from the same three apps it was trained on is being measured on how well it
knows those apps.

This module generates the apps too. A seed produces a complete, coherent GUI —
screens, fields, toggles, radio groups, navigation, a primary action, and
optionally content below the fold — and `synthesis` then derives verified tasks
from it. Train on worlds 0–99, evaluate on worlds 100–119, and a score is
evidence about operating *a* GUI rather than *the* GUI.

Two invariants make generated worlds trustworthy, both enforced rather than
hoped for:

  **Layout is non-overlapping by construction.** Widgets are placed by a
  single downward flow cursor, so no two can occupy the same pixel. Overlap
  would make hit-testing ambiguous and quietly corrupt every gold path derived
  from the world.

  **Every world is task-bearing.** `generate` runs the search before returning
  and rejects a seed that yields nothing reachable. A world nobody can act in
  is a silent hole in a benchmark, and generation is the right place to catch
  it — the same trick as everywhere else here: verify by construction.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from computer_use._render import _GLYPHS, text_width
from computer_use.environments import MockComputer, Widget
from computer_use.tasks import GUITask

#: Field archetypes: the rendered label, a placeholder, and a value worth
#: typing. Keeping the value with the field is what stops synthesis generating
#: "set MAX RETRIES to ada@example.com" — the vocabulary is derived from the
#: world rather than supplied alongside it.
FIELD_KINDS: tuple[tuple[str, str, str, str], ...] = (
    ("email", "EMAIL", "NAME@EXAMPLE.COM", "ada@example.com"),
    ("code", "PROMO CODE", "CODE", "SPRING20"),
    ("retries", "MAX RETRIES", "3", "5"),
    ("filename", "FILE NAME", "UNTITLED", "q4-final.pdf"),
    ("city", "CITY", "CITY", "BERLIN"),
    ("phone", "PHONE", "555 0100", "5550142"),
)

TOGGLE_LABELS: tuple[str, ...] = (
    "EMAIL ME ON FAILURE",
    "REMEMBER THIS DEVICE",
    "SHARE USAGE DATA",
    "ENABLE DARK MODE",
    "AUTO RENEW",
    "REQUIRE APPROVAL",
)

RADIO_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SHIPPING SPEED", ("STANDARD - 5 DAYS", "EXPRESS - 1 DAY", "PICKUP")),
    ("PLAN", ("BASIC", "PRO", "TEAM")),
    ("VISIBILITY", ("PRIVATE", "UNLISTED", "PUBLIC")),
    ("FORMAT", ("PDF", "CSV")),
)

PRIMARY_ACTIONS: tuple[tuple[str, str], ...] = (
    ("SAVE", "saved"),
    ("SUBMIT", "submitted"),
    ("PLACE ORDER", "ordered"),
    ("CONFIRM", "confirmed"),
    ("APPLY", "applied"),
)

APP_TITLES: tuple[str, ...] = (
    "SETTINGS", "SHOP", "FILES", "ACCOUNT", "BILLING", "CONSOLE", "INBOX",
)

SCREEN_NAMES: tuple[str, ...] = ("details", "options", "review")

#: Suffixes that turn a caption into a near-duplicate of itself. A form with
#: both "EMAIL" and "EMAIL BACKUP" cannot be operated by matching words against
#: the instruction — every token of the goal appears on both controls — so an
#: agent has to prefer the caption that matches *exactly* over the one that
#: merely contains it. This is what real settings screens look like, and it is
#: the cheapest way to stop token overlap from being a complete strategy.
DISTRACTOR_SUFFIXES: tuple[str, ...] = ("BACKUP", "ALERTS", "ALT", "OVERRIDE")

#: Labels for a second, wrong primary action. "SAVE" beside "SAVE DRAFT" is the
#: decoy an agent takes when it is scanning for a word rather than reading the
#: screen — and both are filled, so the visual cue does not separate them.
DECOY_ACTIONS: tuple[str, ...] = ("DRAFT", "ALL", "LATER")

# Layout constants. A single downward cursor makes overlap impossible.
_X = 60
_TOP = 110
_GAP = 18
_LABEL_H = 26
_FIELD_H = 44
_BOX = 32
_BUTTON_W, _BUTTON_H = 200, 48


@dataclass(frozen=True)
class WorldSpec:
    """The shape of a generated app, before any pixels exist."""

    seed: int
    title: str
    screens: int
    fields: int
    toggles: int
    radio_groups: int
    scroll: bool
    action: str
    flag: str
    #: Near-duplicate controls added to defeat matching by word overlap.
    distractors: int = 0

    def summary(self) -> str:
        parts = [f"{self.screens} screen(s)", f"{self.fields} field(s)"]
        if self.toggles:
            parts.append(f"{self.toggles} toggle(s)")
        if self.radio_groups:
            parts.append(f"{self.radio_groups} radio group(s)")
        if self.scroll:
            parts.append("below-the-fold action")
        if self.distractors:
            parts.append(f"{self.distractors} distractor(s)")
        return ", ".join(parts)


@dataclass(frozen=True)
class World:
    """A generated app: widgets, geometry, and the vocabulary it implies."""

    spec: WorldSpec
    widgets: tuple[Widget, ...]
    vocabulary: Mapping[str, tuple[str, ...]]
    content_height: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"world{self.spec.seed:03d}"

    def factory(self) -> Callable[[], MockComputer]:
        """A zero-argument builder, which is what environments are passed as."""
        widgets, title, content_height = self.widgets, self.spec.title, self.content_height

        def _build() -> MockComputer:
            return MockComputer(widgets, title=title, content_height=content_height)

        return _build

    def build(self) -> MockComputer:
        return self.factory()()


class _Layout:
    """A downward flow cursor. Widgets cannot overlap because y only grows."""

    def __init__(self, screen: str) -> None:
        self.screen = screen
        self.y = _TOP
        self.widgets: list[Widget] = []

    def label(self, text: str) -> None:
        self.widgets.append(
            Widget(f"{self.screen}_lbl{len(self.widgets)}", "label", _X, self.y,
                   420, _LABEL_H, label=text, screen=self.screen)
        )
        self.y += _LABEL_H + _GAP

    def field(self, widget_id: str, label: str, placeholder: str) -> None:
        self.widgets.append(
            Widget(f"{widget_id}_label", "label", _X, self.y, 300, _LABEL_H,
                   label=label, screen=self.screen)
        )
        self.y += _LABEL_H + 4
        self.widgets.append(
            Widget(widget_id, "field", _X, self.y, 460, _FIELD_H,
                   placeholder=placeholder, screen=self.screen)
        )
        self.y += _FIELD_H + _GAP

    def checkbox(self, widget_id: str, label: str) -> None:
        self.widgets.append(
            Widget(widget_id, "checkbox", _X, self.y, _BOX, _BOX,
                   label=label, screen=self.screen)
        )
        self.y += _BOX + _GAP

    def radios(self, group: str, label: str, options: Sequence[str]) -> None:
        self.label(label)
        for index, option in enumerate(options):
            self.widgets.append(Widget(
                f"{group}_{index}", "radio", _X, self.y, _BOX, _BOX,
                label=option, group=group, checked=index == 0,
                screen=self.screen,
            ))
            self.y += _BOX + 10
        self.y += _GAP - 10

    def spacer(self, pixels: int) -> None:
        self.y += pixels

    def buttons(self, specs: Sequence[tuple[str, str, str | None, str | None]]) -> None:
        """A row of buttons: (id, label, sets, shows)."""
        x = _X
        for widget_id, label, sets, shows in specs:
            # Size to the caption. A fixed width clips a long label at the
            # button's edge, and a control whose own text runs off it cannot be
            # read back from the screen — which makes it unusable to an agent
            # working from pixels, for reasons that have nothing to do with the
            # task being hard.
            width = max(_BUTTON_W, text_width(label, 3) + 24)
            self.widgets.append(Widget(
                widget_id, "button", x, self.y, width, _BUTTON_H,
                label=label, sets=sets, shows=shows, screen=self.screen,
            ))
            x += width + 30
        self.y += _BUTTON_H + _GAP


def _spec(seed: int, *, hard: bool = False) -> WorldSpec:
    rng = random.Random(seed)
    screens = rng.choice((1, 1, 2, 3))
    action, flag = rng.choice(PRIMARY_ACTIONS)
    return WorldSpec(
        seed=seed,
        title=rng.choice(APP_TITLES),
        screens=screens,
        fields=rng.randint(1, 2),
        toggles=rng.randint(0, 2),
        radio_groups=rng.randint(0, 1),
        # Scrolling and multi-screen navigation are the two ways an app hides a
        # control. Requiring both at once makes most worlds deep and slow to
        # search, so a world gets one or the other.
        scroll=screens == 1 and rng.random() < 0.5,
        action=action,
        flag=flag,
        distractors=rng.randint(1, 2) if hard else 0,
    )


def _compose(spec: WorldSpec) -> tuple[list[Widget], dict[str, tuple[str, ...]], int]:
    rng = random.Random(spec.seed + 9973)
    screens = ["main", *SCREEN_NAMES[: spec.screens - 1]]
    widgets: list[Widget] = []
    vocabulary: dict[str, tuple[str, ...]] = {}

    field_kinds = rng.sample(FIELD_KINDS, k=min(spec.fields, len(FIELD_KINDS)))
    toggles = rng.sample(TOGGLE_LABELS, k=min(spec.toggles, len(TOGGLE_LABELS)))
    groups = rng.sample(RADIO_GROUPS, k=min(spec.radio_groups, len(RADIO_GROUPS)))

    # Deal the controls across screens so later screens are not empty.
    buckets: list[list[tuple[str, Any]]] = [[] for _ in screens]
    items: list[tuple[str, Any]] = (
        [("field", k) for k in field_kinds]
        + [("toggle", t) for t in toggles]
        + [("radios", g) for g in groups]
    )
    rng.shuffle(items)
    for index, item in enumerate(items):
        buckets[index % len(screens)].append(item)

    # Distractors go beside the control they imitate, not somewhere else on the
    # screen: a twin caption three screens away is a navigation problem, while
    # a twin caption in the same list is a reading problem, and reading is what
    # is being tested.
    # Snapshot first: a twin of a twin gives "EMAIL BACKUP ALERTS", which is
    # not a harder screen so much as an incoherent one, and no real form is
    # laid out that way.
    originals = [item for bucket in buckets for item in bucket
                 if item[0] in ("field", "toggle")]
    for slot, item in enumerate(originals[:spec.distractors]):
        decoy = _twin(item, DISTRACTOR_SUFFIXES[slot % len(DISTRACTOR_SUFFIXES)])
        bucket = next(b for b in buckets if item in b)
        bucket.insert(bucket.index(item) + 1, decoy)
        if decoy[0] == "field":
            widget_id, _, _, value = decoy[1]
            vocabulary[widget_id] = (value,)

    content_height = 720
    for index, screen in enumerate(screens):
        layout = _Layout(screen)
        layout.label(f"{spec.title} {screen.upper()}")

        for kind, payload in buckets[index]:
            if kind == "field":
                widget_id, label, placeholder, value = payload
                layout.field(widget_id, label, placeholder)
                vocabulary[widget_id] = (value,)
            elif kind == "toggle":
                layout.checkbox(_slug(payload), payload)
            else:
                group_label, options = payload
                layout.radios(_slug(group_label), group_label, options)

        last = index == len(screens) - 1
        if last and spec.scroll:
            # Push the primary action below the fold so reaching it requires a
            # scroll — the affordance the search has to discover.
            layout.spacer(max(0, 780 - layout.y))

        row: list[tuple[str, str, str | None, str | None]] = []
        if index > 0:
            row.append((f"{screen}_back", "BACK", None, screens[index - 1]))
        if last:
            row.append((f"{screen}_go", spec.action, spec.flag, None))
            if spec.distractors:
                # A second button carrying the same verb, and filled the same
                # way, so neither the words nor the styling picks the right one
                # on its own. It sets a real but different flag, which means
                # pressing it is a wrong answer the verifier can see.
                variant = DECOY_ACTIONS[spec.seed % len(DECOY_ACTIONS)]
                row.append((
                    f"{screen}_go_decoy", f"{spec.action} {variant}",
                    f"{spec.flag}_{variant.lower()}", None,
                ))
        else:
            row.append((f"{screen}_next", "CONTINUE", None, screens[index + 1]))
        layout.buttons(row)

        widgets.extend(layout.widgets)
        content_height = max(content_height, layout.y + 60)

    return widgets, vocabulary, content_height


def _twin(item: tuple[str, Any], suffix: str) -> tuple[str, Any]:
    """A near-duplicate of one control, to sit directly beneath it."""
    kind, payload = item
    if kind == "field":
        widget_id, label, placeholder, value = payload
        return ("field", (
            f"{widget_id}_{suffix.lower()}", f"{label} {suffix}", placeholder,
            _vary(value),
        ))
    return ("toggle", f"{payload} {suffix}")


def _vary(value: str) -> str:
    """A different value for the twin, so a task names one of them uniquely."""
    if value.isdigit():
        return str(int(value) + 2)
    head, _, tail = value.partition("@")
    return f"{head}.alt@{tail}" if tail else f"{value}-2"


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")[:24]


def generate(seed: int, *, hard: bool = False) -> World:
    """Generate one world. Deterministic in `seed` (and in `hard`)."""
    spec = _spec(seed, hard=hard)
    widgets, vocabulary, content_height = _compose(spec)
    return World(
        spec=spec,
        widgets=tuple(widgets),
        vocabulary=vocabulary,
        content_height=content_height,
        metadata={"generated": True, "seed": seed},
    )


def is_viable(world: World, *, max_depth: int = 4) -> bool:
    """Whether any goal is reachable — i.e. whether the world can host a task.

    Generation validates itself by running the same search that will later
    derive the tasks. A world nobody can act in would otherwise sit in a
    benchmark contributing nothing but a zero.
    """
    from computer_use.synthesis import SynthesisConfig, explore

    found = explore(
        world.build(),
        SynthesisConfig(vocabulary=world.vocabulary, max_depth=max_depth),
    )
    return any(d.depth >= 2 for d in found)


def generate_many(
    seeds: Iterable[int], *, require_viable: bool = True, max_depth: int = 4,
    hard: bool = False,
) -> list[World]:
    """Generate worlds for `seeds`, dropping any that cannot host a task."""
    worlds = []
    for seed in seeds:
        world = generate(seed, hard=hard)
        if require_viable and not is_viable(world, max_depth=max_depth):
            continue
        worlds.append(world)
    return worlds


def curriculum(
    seeds: Iterable[int],
    *,
    per_world: int | None = 4,
    max_depth: int = 5,
    min_depth: int = 2,
    sample_seed: int | None = 0,
    hard: bool = False,
) -> list[GUITask]:
    """Tasks across generated worlds — the unit you actually train or test on.

    Every task carries its world's seed in `metadata`, so a result can always
    be traced back to the app it came from and splits can be audited.
    """
    from computer_use.synthesis import SynthesisConfig, synthesize

    tasks: list[GUITask] = []
    for world in generate_many(seeds, hard=hard):
        generated = synthesize(
            world.factory(),
            SynthesisConfig(vocabulary=world.vocabulary, max_depth=max_depth),
            name_prefix=world.name,
            limit=per_world,
            min_depth=min_depth,
            seed=sample_seed,
        )
        for task in generated:
            task.metadata.update({
                "world_seed": world.spec.seed,
                "world": world.spec.summary(),
                "hard": hard,
            })
        tasks.extend(generated)
    return tasks


def split(
    *, train: range | Sequence[int], test: range | Sequence[int], **kwargs: Any
) -> tuple[list[GUITask], list[GUITask]]:
    """A train/test split over *worlds*, not tasks.

    This is the point of the module. Splitting tasks within an app leaks the
    app — the layout, the labels, where the button is — so a held-out task is
    still a familiar screen. Holding out whole worlds is what makes a score
    evidence of operating a GUI rather than of having seen this one.
    """
    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(
            f"train and test worlds overlap on seeds {sorted(overlap)} — the "
            "split would leak the environment it is meant to hold out"
        )
    return curriculum(train, **kwargs), curriculum(test, **kwargs)


def renderable(text: str) -> bool:
    """Whether every character has a glyph in the mock's bitmap font.

    A label the renderer cannot draw shows as filled blocks, which a vision
    model reads as noise — the task would be unreadable rather than merely
    hard.
    """
    return all(c in _GLYPHS for c in text.upper())


__all__ = [
    "APP_TITLES",
    "FIELD_KINDS",
    "PRIMARY_ACTIONS",
    "RADIO_GROUPS",
    "TOGGLE_LABELS",
    "World",
    "WorldSpec",
    "curriculum",
    "generate",
    "generate_many",
    "is_viable",
    "renderable",
    "split",
]
