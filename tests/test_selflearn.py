"""Fast tests for the self-learning loop's pure-Python parts (no model needed)."""
import random

from selflearn import tasks as T
from selflearn.curriculum import Curriculum, informative
from selflearn.grpo import group_advantages
from selflearn.tools import TOOLS, calc, last_call, run_call
from selflearn.verifier import parse_answer, reward


def test_tasks_are_deterministic_positive_and_hold_out_is_fixed():
    a, b = T.make_eval_set(5), T.make_eval_set(5)
    assert a == b and len(a) == 5 * len(T.LEVELS)
    assert all(t.answer > 0 and isinstance(t.answer, int) for t in a)


def test_level_controls_number_of_steps():
    rng = random.Random(0)
    for lv in T.LEVELS:
        t = T.make_task(lv, rng)
        assert t.question.count(".") - 1 == lv  # one sentence per step, plus the opening sentence


def test_train_tasks_never_hit_the_held_out_set():
    held = {t.question for t in T.make_eval_set(3)}
    rng = random.Random(1)
    assert all(T.make_train_task(lv, rng, held).question not in held for lv in T.LEVELS for _ in range(20))


def test_calculator_is_exact_and_safe():
    assert calc("12 + 3 * 25") == "87"
    assert calc("1,200 / 4") == "300"
    assert calc("7 / 2") == "3.5"
    assert calc("1/0").startswith("error")
    assert calc("__import__('os').system('ls')").startswith("error")
    assert calc("2 ** 10").startswith("error")


def test_tool_call_parsing_uses_the_last_closed_call():
    text = "<calc>1+1</calc>\n<result>2</result>\n<calc>2*21</calc>"
    assert last_call(text, TOOLS) == ("calc", "2*21")
    assert run_call(text, TOOLS) == "42"
    assert run_call("<calc>3+", TOOLS) is None


def test_verifier():
    assert parse_answer("<answer> 1,234 </answer>") == 1234
    assert reward("<answer>87</answer>", 87) == 1.0
    assert reward("<answer>86</answer>", 87) == 0.0
    assert reward("the answer is 87", 87) == 0.0


def test_group_advantages_drop_groups_without_signal():
    adv = group_advantages([1, 0, 0, 0, 1, 1, 1, 1], group=4)
    assert adv[4:] == [None] * 4
    assert adv[0] > 0 > adv[1] and abs(sum(adv[:4])) < 1e-6


def test_curriculum_prefers_the_learning_edge():
    assert informative(0.5, 8) > informative(0.9, 8) > informative(0.99, 8)
    cur = Curriculum({1: 0.99, 2: 0.5, 3: 0.01, 4: 0.5, 5: 0.5, 6: 0.5}, group=8)
    w = cur.weights()
    assert w[2] > w[1] and w[2] > w[3]
    cur.update(2, 1.0)
    assert cur.p[2] > 0.5


def test_coordinator_runs_the_loop_in_order_with_any_agents():
    """The Coordinator depends only on role interfaces: fake agents (no model) are enough to run it."""
    from selflearn.coordinator import Coordinator
    calls = []

    class Assessor:
        def assess(self, tag):
            calls.append(f"assess:{tag}")
            return {"accuracy": 0.5, "by_level": {1: 0.5}}, []

    class Proposer:
        def calibrate(self, by_level):
            calls.append("calibrate")

        def propose(self, n):
            calls.append("propose")
            return [T.Task("q", 1, 1)] * n

        def observe(self, trajs):
            calls.append("observe")

    class Actor:
        def act(self, tasks, group, temp):
            calls.append("act")
            return ["t"] * (len(tasks) * group)

    class Verifier:
        def score(self, trajs):
            calls.append("score")
            return 0.5

    class Learner:
        def learn(self, trajs):
            calls.append("learn")
            return {"informative_groups": 1}

    steps = []
    res = Coordinator(Assessor(), Proposer(), Actor(), Verifier(), Learner()).run(2, 3, 4, 1.0, lambda s, r: steps.append(r))
    loop = ["propose", "act", "score", "observe", "learn"]
    assert calls == ["assess:base", "calibrate"] + loop * 2 + ["assess:trained"]
    assert [s["step"] for s in steps] == [1, 2] and res["after"]["accuracy"] == 0.5
