"""Coding: write a Python function that passes hidden unit tests.

Code runs in a separate, isolated (`python -I`) subprocess with a timeout.
This is a correctness harness, not a security sandbox: only run policies you
trust, or run the whole eval inside a container.
"""

from __future__ import annotations

import re
import subprocess
import sys

from evals.environment import Environment
from evals.types import Action, Grade, Task, Trajectory

TIMEOUT_S = 5.0

# (function name, instruction, reference solution, hidden tests)
_CASES: list[tuple[str, str, str, str]] = [
    (
        "fizzbuzz",
        "Return 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, else str(n).",
        "def fizzbuzz(n):\n"
        "    if n % 15 == 0:\n        return 'FizzBuzz'\n"
        "    if n % 3 == 0:\n        return 'Fizz'\n"
        "    if n % 5 == 0:\n        return 'Buzz'\n"
        "    return str(n)\n",
        "assert fizzbuzz(3) == 'Fizz'\nassert fizzbuzz(10) == 'Buzz'\n"
        "assert fizzbuzz(30) == 'FizzBuzz'\nassert fizzbuzz(7) == '7'\n",
    ),
    (
        "is_palindrome",
        "Return True if s is a palindrome ignoring case and non-alphanumerics.",
        "def is_palindrome(s):\n"
        "    t = [c.lower() for c in s if c.isalnum()]\n"
        "    return t == t[::-1]\n",
        "assert is_palindrome('A man, a plan, a canal: Panama')\n"
        "assert not is_palindrome('hello')\nassert is_palindrome('')\n",
    ),
    (
        "merge_intervals",
        "Merge overlapping [start, end] intervals; return them sorted by start.",
        "def merge_intervals(xs):\n"
        "    out = []\n"
        "    for s, e in sorted(xs):\n"
        "        if out and s <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], e)\n"
        "        else:\n"
        "            out.append([s, e])\n"
        "    return out\n",
        "assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]\n"
        "assert merge_intervals([[5,7],[1,2]]) == [[1,2],[5,7]]\n"
        "assert merge_intervals([]) == []\n",
    ),
    (
        "word_count",
        "Return a dict mapping each lowercase word in text to its count.",
        "def word_count(text):\n"
        "    counts = {}\n"
        "    for w in text.lower().split():\n"
        "        counts[w] = counts.get(w, 0) + 1\n"
        "    return counts\n",
        "assert word_count('a b A') == {'a': 2, 'b': 1}\nassert word_count('') == {}\n",
    ),
    (
        "flatten",
        "Flatten an arbitrarily nested list of ints.",
        "def flatten(xs):\n"
        "    out = []\n"
        "    for x in xs:\n"
        "        out.extend(flatten(x) if isinstance(x, list) else [x])\n"
        "    return out\n",
        "assert flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]\nassert flatten([]) == []\n",
    ),
    (
        "binary_search",
        "Return the index of target in sorted list xs, or -1 if absent.",
        "def binary_search(xs, target):\n"
        "    lo, hi = 0, len(xs) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if xs[mid] == target:\n            return mid\n"
        "        if xs[mid] < target:\n            lo = mid + 1\n"
        "        else:\n            hi = mid - 1\n"
        "    return -1\n",
        "assert binary_search([1,3,5,7], 7) == 3\nassert binary_search([1,3,5,7], 4) == -1\n"
        "assert binary_search([], 1) == -1\n",
    ),
]

_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """First fenced block if present, else the raw text."""
    match = _FENCE.search(text)
    return match.group(1) if match else text


def run_tests(code: str, tests: str) -> str | None:
    """Return a failure tag, or None if all tests pass."""
    if not code.strip():
        return "no_code"
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError:
        return "syntax_error"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code + "\n" + tests],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    if proc.returncode == 0:
        return None
    return "test_failure" if "AssertionError" in proc.stderr else "runtime_error"


class CodingEnv(Environment):
    name = "coding"

    def tasks(self) -> list[Task]:
        return [
            Task(
                id=f"code-{fn}",
                prompt=f"Write a Python function `{fn}`. {instruction} Reply with one code block.",
                reference=(Action(text=f"```python\n{solution}```"),),
                data={"tests": tests},
            )
            for fn, instruction, solution, tests in _CASES
        ]

    def grade(self, task: Task, trajectory: Trajectory) -> Grade:
        text = trajectory.final.text if trajectory.final else ""
        failure = run_tests(extract_code(text), task.data["tests"])
        return Grade(
            task.id, self.name, failure is None, float(failure is None), failure, task.tags
        )
