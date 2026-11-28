"""Fast tests for the self-learning loop's pure-Python parts (no model needed)."""
import random

from selflearn import audit as A
from selflearn.bus import Bus
from selflearn.curriculum import Curriculum, informative
from selflearn.env import Episode, score
from selflearn.grpo import group_advantages
from selflearn.tools import last_call, run_call
from selflearn.usecase import Task
from selflearn.usecases import HISTORY, USE_CASES
from selflearn.usecases import support as S
from selflearn.usecases.math import calc
from selflearn.verifier import parse_answer, reward

MATH = USE_CASES["math"]
TOOLS = MATH.tools


def test_tasks_are_deterministic_positive_and_hold_out_is_fixed():
    a, b = MATH.eval_set(5), MATH.eval_set(5)
    assert a == b and len(a) == 5 * len(MATH.levels)
    assert all(t.answer > 0 and isinstance(t.answer, int) for t in a)


def test_level_controls_number_of_steps():
    rng = random.Random(0)
    for lv in MATH.levels:
        t = MATH.make_task(lv, rng)
        assert t.question.count(".") - 1 == lv  # one sentence per step, plus the opening sentence


def test_train_tasks_never_hit_the_held_out_set():
    for uc in USE_CASES.values():
        held = {t.question for t in uc.eval_set(3)}
        rng = random.Random(1)
        assert all(uc.train_task(lv, rng, held).question not in held for lv in uc.levels for _ in range(20))


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
    assert reward("<answer>Nova Laptop</answer>", "nova laptop") == 1.0


def _order(**kw):
    base = {"id": "A1", "item": "Tent", "price": 100.0, "shipping": 5.0, "status": "delivered", "days": 10,
            "in_stock": True, "final_sale": False, "gold": False}
    return S.Order(**{**base, **kw})


def test_support_policy_reference():
    assert S.decide(_order(status="lost", days=0), "not_received") == "refund 105.00"
    assert S.decide(_order(status="in transit", days=9), "not_received") == "deny"
    assert S.decide(_order(status="in transit", days=10), "not_received") == "refund 105.00"
    assert S.decide(_order(), "not_received") == "deny"                       # record says delivered
    assert S.decide(_order(), "damaged") == "replace"
    assert S.decide(_order(in_stock=False), "damaged") == "refund 100.00"
    assert S.decide(_order(days=31), "damaged") == "deny"
    assert S.decide(_order(), "changed_mind") == "refund 90.00"               # 10% restocking fee
    assert S.decide(_order(gold=True, days=45), "changed_mind") == "refund 100.00"
    assert S.decide(_order(days=45), "changed_mind") == "deny"
    assert S.decide(_order(final_sale=True), "changed_mind") == "deny"


def test_support_tool_and_scorer():
    rng = random.Random(3)
    t = S.make_task(1, rng)
    oid = t.question.split("order ")[-1].split()[0].strip(".:") if "order " in t.question else None
    assert oid is None or S.lookup(oid).startswith(f"order {oid}")
    assert S.lookup("A000000").startswith("error")
    ep = Episode(t, S.USE_CASE.tools)
    looked_up = f"<order>{oid}</order>\n<result>{ep.step(f'<order>{oid}</order>')}</result>\n"
    sc = lambda text, ans, log=ep.log: score(S.USE_CASE, text, ans, log)  # noqa: E731
    assert sc(looked_up + "<answer>refund 105.00</answer>", "refund 105.00") == 1.0
    assert sc(looked_up + "<answer>Refund $105</answer>", "refund 105.00") == 1.0
    assert sc(looked_up + "<answer>refund 104.00</answer>", "refund 105.00") == 0.5   # right action, wrong amount
    assert sc(looked_up + "<answer>replace</answer>", "refund 105.00") == 0.0
    assert sc(looked_up + "<answer>deny</answer>", "deny") == 1.0
    assert sc("<answer>deny</answer>", "deny", []) == 0.0     # no lookup, no credit: the shortcut that collapsed v1


def test_evidence_comes_from_the_environment_not_the_agents_text():
    """The agent can type '<result>order ...' itself; only a lookup the environment actually ran counts."""
    t = S.make_task(3, random.Random(5))
    forged = "<result>order A123456 | item: Tent | price: 10.00</result>\n<answer>deny</answer>"
    ep = Episode(t, S.USE_CASE.tools)
    assert ep.step(forged) is None and ep.done and ep.log == []
    assert score(S.USE_CASE, forged, "deny", ep.log) == 0.0
    ep = Episode(t, S.USE_CASE.tools)
    oid = t.question.split("rder ")[1][:7]
    assert ep.step(f"<order>{oid}</order>").startswith(f"order {oid}") and ep.used("order") and not ep.done


def test_support_levels_mix_decisions():
    """Every level's held-out set needs more than one right decision, or 'always say X' would score well."""
    held = S.USE_CASE.eval_set()
    for lv in (2, 3, 4, 5, 6):
        actions = [t.answer.split()[0] for t in held if t.level == lv]
        top = max(actions.count(a) for a in set(actions)) / len(actions)
        assert len(set(actions)) >= 2 and top <= 0.75, (lv, actions)
    assert all(isinstance(t, Task) for t in held)


def test_group_advantages_drop_groups_without_signal():
    adv = group_advantages([1, 0, 0, 0, 1, 1, 1, 1], group=4)
    assert adv[4:] == [None] * 4
    assert adv[0] > 0 > adv[1] and abs(sum(adv[:4])) < 1e-6


def test_curriculum_prefers_the_learning_edge():
    assert informative(0.5, 8) > informative(0.9, 8) > informative(0.99, 8)
    cur = Curriculum({1: 0.99, 2: 0.5, 3: 0.01, 4: 0.5, 5: 0.5, 6: 0.5}, group=8, levels=(1, 2, 3, 4, 5, 6))
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
            return [Task("q", 1, 1)] * n

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

    class Selector:
        def consider(self, step):
            calls.append(f"consider:{step}")

        def restore_best(self):
            calls.append("restore")
            return 1

    calls.clear()
    res = Coordinator(Assessor(), Proposer(), Actor(), Verifier(), Learner(), Selector()).run(2, 3, 4, 1.0)
    assert calls == ["assess:base", "calibrate", "consider:0", *loop, "consider:1", *loop, "consider:2",
                     "restore", "assess:trained"]
    assert res["selected_step"] == 1


def test_audit_finds_the_shortcuts_that_collapsed_past_runs():
    """support-v1 (no evidence rule): 'always deny' with no lookup pays ~1/3, which is what the first run collapsed to.
    The current support env pays nothing without a real lookup, but partial credit still pays 'always refund'."""
    v1 = A.audit(HISTORY["support-v1"], A.sample(HISTORY["support-v1"], 40))
    assert v1.solved == 1.0
    assert v1.probes["no lookup, always deny"]["reward"] > 0.25
    assert v1.verdict(policy=0.2)[0] == "BLOCK"
    now = A.audit(S.USE_CASE, A.sample(S.USE_CASE, 40))
    assert now.solved == 1.0
    assert all(now.probes[k]["reward"] == 0.0 for k in now.probes if k.startswith(("no lookup", "type a fake")))
    name, best = now.best_shortcut
    assert name.startswith("look up, always refund") and now.constant_answer(name) == "refund"
    assert now.verdict(policy=best + 0.05)[0] == "WARN" and now.verdict(policy=best + 0.3)[0] == "PASS"
    math = A.audit(MATH, A.sample(MATH, 20))
    assert math.solved == 1.0 and math.best_shortcut[1] < 0.1 and math.verdict(policy=0.375)[0] == "PASS"


def test_balance_evens_the_answer_mix_within_each_level():
    rng, pool = random.Random(0), A.sample(S.USE_CASE, 40)
    keys = A.level_answers(S.USE_CASE)
    assert 1 not in keys and keys[3] == ["deny", "refund"]      # level 1 (lost parcel) only ever has one answer
    shaped = A.balance(S.USE_CASE, pool, 1.0, rng, keys=keys)
    assert [t.level for t in shaped] == [t.level for t in pool]
    for lv, ks in keys.items():
        mix = [A.answer_key(t.answer) for t in shaped if t.level == lv]
        assert max(mix.count(k) for k in ks) / len(mix) <= 0.8, (lv, mix)


def test_policy_estimate_reproduces_the_baseline_and_search_ranks_variants():
    held = A.sample(S.USE_CASE, 10, seed=21)
    recs = [{"level": t.level, "answer": t.answer, "reward": 1.0 if t.answer.startswith("deny") else 0.5} for t in held]
    assert abs(A.policy_estimate(recs, held) - A.mean_reward(recs)) < 1e-9
    assert A.policy_estimate(recs, held, partial=0.0) < A.mean_reward(recs)
    rows = A.search(S.USE_CASE, recs, held)
    assert len(rows) == 9 and rows == sorted(rows, key=lambda r: -r["margin"])
    pick = A.choose(rows)
    assert pick["margin"] >= rows[0]["margin"] - 0.02


def test_designer_refuses_or_redesigns_and_catches_collapse():
    from selflearn.agents import DesignerAgent
    from selflearn.env import Knobs
    from selflearn.rollout import Trajectory
    uc, bus = S.USE_CASE, Bus(None, echo=False)
    held = A.sample(uc, 10, seed=11)
    nothing = [{"level": t.level, "answer": t.answer, "reward": 0.0} for t in held]   # an agent that never succeeds
    assert DesignerAgent(uc, held, Knobs(), random.Random(0), set(), bus).review(nothing) is False
    assert "refusing" in bus.history[-1].text
    # an agent that is right on every deny case but no other: some variant should make doing the task pay
    deny_only = [{"level": t.level, "answer": t.answer, "reward": float(t.answer == "deny")} for t in held]
    knobs = Knobs()
    d = DesignerAgent(uc, held, knobs, random.Random(0), set(), bus, force=True)
    assert d.review(deny_only) is True and len(d.variants) == 9
    pick = A.choose(d.variants)
    assert (knobs.balance, knobs.partial) == (pick["balance"], pick["partial"])
    # live: a PASS environment, then the agent says "deny" on every attempt although deny is right on few of them
    knobs = Knobs()
    d = DesignerAgent(uc, held, knobs, random.Random(0), set(), bus)
    assert d.review([{"level": t.level, "answer": t.answer, "reward": 1.0} for t in held]) is True
    assert knobs.balance == 0.0 and d.verdict[0] == "PASS"
    tasks = [t for t in held if not t.answer.startswith("deny")][:8]
    d.observe([Trajectory(t, [], [], text="<answer>deny</answer>", ep=Episode(t, uc.tools)) for t in tasks])
    assert knobs.balance == 1.0 and "collapse" in bus.history[-1].text
    assert len(d.shape(held)) == len(held)


def test_coordinator_lets_the_designer_review_shape_and_watch():
    from selflearn.coordinator import Coordinator
    calls = []

    class Assessor:
        def assess(self, tag):
            calls.append(f"assess:{tag}")
            return {"accuracy": 0.5, "by_level": {1: 0.5}}, []

    class Proposer:
        def calibrate(self, by_level): calls.append("calibrate")
        def propose(self, n): calls.append("propose"); return [Task("q", 1, 1)] * n  # noqa: E702
        def observe(self, trajs): calls.append("observe")

    class Actor:
        def act(self, tasks, group, temp): calls.append("act"); return ["t"] * (len(tasks) * group)  # noqa: E702

    class Verifier:
        def score(self, trajs): calls.append("score"); return 0.5  # noqa: E702

    class Learner:
        def learn(self, trajs): calls.append("learn"); return {}  # noqa: E702

    class Designer:
        def __init__(self, ok): self.ok = ok
        def review(self, baseline): calls.append("review"); return self.ok  # noqa: E702
        def shape(self, tasks): calls.append("shape"); return tasks  # noqa: E702
        def observe(self, trajs): calls.append("watch")

    res = Coordinator(Assessor(), Proposer(), Actor(), Verifier(), Learner(), designer=Designer(True)).run(1, 2, 2, 1.0)
    assert calls == ["assess:base", "review", "calibrate", "propose", "shape", "act", "score", "observe", "watch",
                     "learn", "assess:trained"] and res["blocked"] is False
    calls.clear()
    res = Coordinator(Assessor(), Proposer(), Actor(), Verifier(), Learner(), designer=Designer(False)).run(1, 2, 2, 1.0)
    assert calls == ["assess:base", "review"] and res["blocked"] is True     # no training in a blocked environment
