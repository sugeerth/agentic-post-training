"""Use case 2: a customer-support agent decides refunds, the way Sierra- or Decagon-style agents do.

A customer writes in about an order. The agent looks the order up with the <order> tool, does any arithmetic with
<calc>, applies the store's refund policy, and gives one decision: "refund AMOUNT", "replace" or "deny".
`decide()` is the reference implementation of the policy; the reward checks the agent's decision against it, so
there are still no human labels. Level = how many policy rules the case needs (6 = the customer's message
contradicts the order record, and the record wins).
"""
from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from selflearn.usecase import Task, UseCase
from selflearn.usecases.math import calc

SYSTEM_PROMPT = (
    "You are a customer-support agent for an online store. Look up the order, apply the policy, and give one decision.\n"
    "Tools: <order>ORDER_ID</order> looks up an order. <calc>EXPRESSION</calc> does arithmetic. "
    "Each result comes back as <result>...</result>. Use one tool at a time.\n"
    "Refund policy:\n"
    "1. Trust the order record over the customer's message.\n"
    "2. Damaged or wrong item reported within 30 days of delivery: replace if in stock, otherwise refund the price. "
    "After 30 days: deny.\n"
    "3. Changed mind: final-sale items are denied. Otherwise, within 30 days of delivery (60 for gold members), "
    "refund the price minus a 10% restocking fee (gold members pay no fee). After that: deny.\n"
    "4. Not received: if the status is lost, refund price + shipping. If in transit for fewer than 10 days, deny "
    "(ask them to wait). If in transit for 10 days or more, refund price + shipping. If the record says delivered, deny.\n"
    "Answer with <answer>refund AMOUNT</answer>, <answer>replace</answer> or <answer>deny</answer>.\n"
    "Example:\n"
    "Customer: My order B100200 never showed up.\n"
    "<order>B100200</order>\n<result>order B100200 | item: Desk Lamp | price: 30.00 | shipping: 5.00 | status: lost | "
    "in stock: yes | final sale: no | member: standard</result>\n"
    "<calc>30.00 + 5.00</calc>\n<result>35</result>\n<answer>refund 35.00</answer>"
)

ITEMS = ("Blender", "Running Shoes", "Backpack", "Coffee Maker", "Headphones", "Desk Chair", "Rain Jacket",
         "Water Bottle", "Yoga Mat", "Toaster", "Phone Case", "Wool Sweater", "Air Fryer", "Book Light", "Tent")


@dataclass(frozen=True)
class Order:
    id: str
    item: str
    price: float
    shipping: float
    status: str            # delivered | in transit | lost
    days: int              # days since delivery (delivered) or since shipping (in transit)
    in_stock: bool
    final_sale: bool
    gold: bool

    def record(self) -> str:
        when = {"delivered": f"delivered {self.days} days ago", "in transit": f"in transit for {self.days} days",
                "lost": "status: lost"}[self.status]
        return (f"order {self.id} | item: {self.item} | price: {self.price:.2f} | shipping: {self.shipping:.2f} | "
                + ("" if self.status == "lost" else "status: " + self.status + " | ") + when + " | "
                + f"in stock: {'yes' if self.in_stock else 'no'} | final sale: {'yes' if self.final_sale else 'no'} | "
                + f"member: {'gold' if self.gold else 'standard'}")


ORDERS: dict[str, Order] = {}   # every generated order, so the <order> tool can find it


def lookup(order_id: str) -> str:
    """The agent's <order> tool."""
    o = ORDERS.get(order_id.strip().upper())
    return o.record() if o else f"error: no order {order_id.strip()[:20]}"


def decide(o: Order, issue: str) -> str:
    """Reference policy. Returns 'refund X.XX', 'replace' or 'deny'."""
    if issue == "damaged":
        if o.status != "delivered" or o.days > 30:
            return "deny"
        return "replace" if o.in_stock else f"refund {o.price:.2f}"
    if issue == "changed_mind":
        if o.final_sale or o.status != "delivered" or o.days > (60 if o.gold else 30):
            return "deny"
        return f"refund {o.price if o.gold else round(o.price * 0.9, 2):.2f}"
    if issue == "not_received":
        if o.status == "lost" or (o.status == "in transit" and o.days >= 10):
            return f"refund {o.price + o.shipping:.2f}"
        return "deny"
    raise ValueError(issue)


MESSAGES = {
    "damaged": ("Order {id}: the {item} arrived broken.", "Hi, my {item} from order {id} came damaged. What can you do?",
                "I got the wrong item in order {id}. I ordered a {item}."),
    "changed_mind": ("I'd like to return the {item} from order {id}. I changed my mind.",
                     "Order {id}: I don't need the {item} anymore, can I get my money back?"),
    "not_received": ("My order {id} never arrived.", "Where is my {item}? Order {id} still hasn't come.",
                     "I never received order {id}. Please help."),
}
CLAIMS = {  # level 6: the message claims something the record contradicts
    "gold": " I'm a gold member, so no fee please.",
    "recent": " I only got it last week.",
}


def _order(rng: random.Random, **fixed) -> Order:
    while True:
        oid = f"A{rng.randint(100000, 999999)}"
        if oid not in ORDERS:
            break
    price = round(rng.uniform(12, 400), 2)
    o = Order(id=oid, item=rng.choice(ITEMS), price=price, shipping=rng.choice((0.0, 4.99, 7.50, 12.00)),
              status=fixed.get("status", "delivered"), days=fixed.get("days", rng.randint(1, 25)),
              in_stock=fixed.get("in_stock", rng.random() < 0.6), final_sale=fixed.get("final_sale", False),
              gold=fixed.get("gold", False))
    ORDERS[oid] = o
    return o


def make_task(level: int, rng: random.Random) -> Task:
    if level == 1:      # lost package: look up, add price + shipping
        issue, o = "not_received", _order(rng, status="lost", days=0)
    elif level == 2:    # damaged within 30 days: replace or refund, depending on stock
        issue, o = "damaged", _order(rng, days=rng.randint(1, 30))
    # Levels 3-6 are built about half "refund" and half "deny", so always giving one answer cannot score well.
    elif level == 3:    # changed mind, standard member: 10% fee, or past the 30-day window
        days = rng.randint(1, 30) if rng.random() < 0.5 else rng.randint(31, 90)
        issue, o = "changed_mind", _order(rng, days=days)
    elif level == 4:    # changed mind with an exception: gold member (no fee, 60 days) or final sale
        r = rng.random()
        if r < 0.5:
            issue, o = "changed_mind", _order(rng, gold=True, days=rng.randint(31, 60))
        elif r < 0.75:
            issue, o = "changed_mind", _order(rng, gold=True, days=rng.randint(61, 90))
        else:
            issue, o = "changed_mind", _order(rng, final_sale=True, days=rng.randint(1, 20))
    elif level == 5:    # timing: in transit < or >= 10 days, or damage reported too late
        r = rng.random()
        if r < 0.5:
            issue, o = "not_received", _order(rng, status="in transit", days=rng.randint(10, 16))
        elif r < 0.75:
            issue, o = "not_received", _order(rng, status="in transit", days=rng.randint(3, 9))
        else:
            issue, o = "damaged", _order(rng, days=rng.randint(31, 70))
    elif level == 6:    # the message contradicts the record; the record wins
        kind = rng.choices(("gold", "delivered", "recent"), weights=(2, 1, 1))[0]
        if kind == "delivered":
            issue, o = "not_received", _order(rng, status="delivered", days=rng.randint(2, 20))
        elif kind == "gold":
            issue, o = "changed_mind", _order(rng, days=rng.randint(1, 30))
        else:
            issue, o = "changed_mind", _order(rng, days=rng.randint(31, 90))
        msg = rng.choice(MESSAGES[issue]).format(id=o.id, item=o.item) + CLAIMS.get(kind, "")
        return Task(msg, decide(o, issue), level)
    else:
        raise ValueError("level must be 1-6")
    return Task(rng.choice(MESSAGES[issue]).format(id=o.id, item=o.item), decide(o, issue), level)


_DECISION = re.compile(r"(refund|replace|deny)\s*\$?\s*([\d,]*\.?\d*)", re.I)


def reward(text: str, answer: str) -> float:
    """1.0 for the right decision (action, and amount to the cent for refunds); 0.5 for the right action with a wrong
    amount; 0.0 otherwise. The environment adds the evidence rule (requires=("order",)): a decision made without a
    successful <order> lookup in the episode scores 0 (policy rule 1). See selflearn/env.py.

    Why: with a plain 0/1 reward and no lookup rule, the first run collapsed to always answering "deny" without
    looking anything up (right about a third of the time, no effort). Requiring evidence removes that shortcut, and
    partial credit gives refund cases a learning signal before the agent gets amounts exactly right.
    Held-out accuracy still counts only 1.0 as correct."""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if not m:
        return 0.0
    got = _DECISION.search(m[-1])
    if not got:
        return 0.0
    action, want_action = got.group(1).lower(), answer.split()[0]
    if action != want_action:
        return 0.0
    if action != "refund":
        return 1.0
    try:
        return 1.0 if abs(float(got.group(2).replace(",", "")) - float(answer.split()[1])) < 0.006 else 0.5
    except ValueError:
        return 0.5


# ----------------------------------------------------------------------------- shortcut probes for the env audit
# Scripted policies, each skipping part of the job. The Designer agent plays them in the environment before training:
# a reward that pays one of them about as much as the agent already earns is a reward the agent will learn to hack.
def _probe(decide_from: Callable[[Task, Order | None], str], look_up: bool = True):
    def play(task: Task, ep) -> str:
        text, o = "", None
        if look_up:
            m = re.search(r"[A-Z]\d{6}", task.question)
            oid = m.group() if m else "A000000"
            text = f"<order>{oid}</order>\n<result>{ep.step(f'<order>{oid}</order>')}</result>\n"
            o = ORDERS.get(oid)
        final = f"<answer>{decide_from(task, o)}</answer>"
        ep.step(final)
        return text + final
    return play


PROBES = {
    "oracle": _probe(lambda t, o: str(t.answer)),
    "no lookup, always deny": _probe(lambda t, o: "deny", look_up=False),
    "no lookup, always replace": _probe(lambda t, o: "replace", look_up=False),
    "look up, always deny": _probe(lambda t, o: "deny"),
    "look up, always replace": _probe(lambda t, o: "replace"),
    "look up, always refund the price": _probe(lambda t, o: f"refund {o.price if o else 0:.2f}"),
    "look up, always refund price + shipping": _probe(lambda t, o: f"refund {(o.price + o.shipping) if o else 0:.2f}"),
}


USE_CASE = UseCase(name="support", system_prompt=SYSTEM_PROMPT, tools={"order": lookup, "calc": calc},
                   levels=(1, 2, 3, 4, 5, 6), make_task=make_task, reward=reward, requires=("order",),
                   probes=PROBES, user_prefix="Customer: ",
                   max_new=96, eval_per_level=15,
                   summary="taught itself to handle refund requests: look up the order, apply the refund policy, decide",
                   level_note="level = how many policy rules the case needs (6 = the message contradicts the record)")


def exact_decision(text: str, answer: str) -> float:
    """The first run's reward: 1 for exactly the right decision, else 0, and no evidence rule."""
    return float(reward(text, answer) == 1.0)


# The environment the first run trained in (it collapsed to "deny" on everything). Kept so the Designer's audit can be
# checked against what actually happened: python3 -m selflearn.audit --usecase support-v1
USE_CASE_V1 = replace(USE_CASE, name="support-v1", requires=(), reward=exact_decision)
