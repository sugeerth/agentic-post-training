"""Gradient checks for the autograd in `computer_use.nn`.

Every op is compared against a central finite difference of the same scalar it
claims to differentiate. This is the only test in the package that would catch
the failure mode that matters here: a hand-written backward with a transposed
index still points roughly downhill, so the loss falls, the model trains, and
nothing looks wrong until the result is compared against something.

The tolerance is 1e-4 relative. The ops here land near 1e-10, so a failure at
1e-4 is a real derivative error rather than float noise.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

import pytest

from computer_use import nn
from computer_use.nn import Tensor


def _tensor(rows: int, cols: int, rng: random.Random) -> Tensor:
    return Tensor(
        [rng.gauss(0.0, 1.0) for _ in range(rows * cols)], (rows, cols), requires_grad=True
    )


def _scalarize(x: Tensor, seed: int = 1) -> Tensor:
    """Collapse a tensor to a scalar with fixed random weights.

    Random weights rather than a plain sum: summing gives every entry the same
    upstream gradient, which hides any backward that scrambles the order of its
    output. A distinct weight per column makes that visible.
    """
    rng = random.Random(seed)
    weights = Tensor([rng.gauss(0.0, 1.0) for _ in range(x.cols)], (x.cols, 1))
    column = nn.matmul(x, weights)
    ones = Tensor([1.0] * column.rows, (1, column.rows))
    return nn.matmul(ones, column)


def check_gradients(
    build: Callable[[], Tensor],
    params: Sequence[Tensor],
    *,
    eps: float = 1e-5,
    tolerance: float = 1e-4,
) -> float:
    """Worst relative error between analytic and numeric gradients."""
    for p in params:
        p.zero_grad()
    build().backward()
    analytic = [list(p.grad) if p.grad is not None else [0.0] * len(p.data) for p in params]

    worst = 0.0
    for index, p in enumerate(params):
        for i in range(len(p.data)):
            original = p.data[i]
            p.data[i] = original + eps
            up = build().data[0]
            p.data[i] = original - eps
            down = build().data[0]
            p.data[i] = original
            numeric = (up - down) / (2 * eps)
            scale = max(1.0, abs(numeric), abs(analytic[index][i]))
            worst = max(worst, abs(numeric - analytic[index][i]) / scale)
    assert worst < tolerance, f"worst relative gradient error {worst:.2e}"
    return worst


class TestGradients:
    """Each op differentiates what it says it differentiates."""

    def test_matmul_accumulate_shape(self) -> None:
        rng = random.Random(0)
        a, b = _tensor(4, 3, rng), _tensor(3, 5, rng)
        check_gradients(lambda: _scalarize(nn.matmul(a, b)), [a, b])

    def test_matmul_wide_inner_dimension(self) -> None:
        """The other loop order — output narrower than the shared dimension."""
        rng = random.Random(1)
        a, b = _tensor(5, 9, rng), _tensor(9, 3, rng)
        check_gradients(lambda: _scalarize(nn.matmul(a, b)), [a, b])

    def test_add_bias(self) -> None:
        rng = random.Random(2)
        x, bias = _tensor(4, 3, rng), _tensor(1, 3, rng)
        check_gradients(lambda: _scalarize(nn.add_bias(x, bias)), [x, bias])

    def test_add(self) -> None:
        rng = random.Random(3)
        a, b = _tensor(4, 3, rng), _tensor(4, 3, rng)
        check_gradients(lambda: _scalarize(nn.add(a, b)), [a, b])

    def test_scale(self) -> None:
        rng = random.Random(4)
        x = _tensor(4, 3, rng)
        check_gradients(lambda: _scalarize(nn.scale(x, 0.37)), [x])

    def test_relu(self) -> None:
        rng = random.Random(5)
        x = _tensor(4, 3, rng)
        check_gradients(lambda: _scalarize(nn.relu(x)), [x])

    def test_transpose(self) -> None:
        rng = random.Random(6)
        x = _tensor(4, 3, rng)
        check_gradients(lambda: _scalarize(nn.transpose(x), seed=3), [x])

    def test_slice_cols(self) -> None:
        rng = random.Random(7)
        x = _tensor(4, 6, rng)
        check_gradients(lambda: _scalarize(nn.slice_cols(x, 2, 3), seed=4), [x])

    def test_concat_cols(self) -> None:
        rng = random.Random(8)
        a, b = _tensor(4, 2, rng), _tensor(4, 3, rng)
        check_gradients(lambda: _scalarize(nn.concat_cols([a, b])), [a, b])

    def test_layer_norm(self) -> None:
        rng = random.Random(9)
        x, gain, bias = _tensor(4, 5, rng), _tensor(1, 5, rng), _tensor(1, 5, rng)
        check_gradients(lambda: _scalarize(nn.layer_norm(x, gain, bias)), [x, gain, bias])

    def test_softmax_rows(self) -> None:
        rng = random.Random(10)
        x = _tensor(4, 4, rng)
        check_gradients(lambda: _scalarize(nn.softmax_rows(x), seed=7), [x])

    def test_softmax_causal(self) -> None:
        rng = random.Random(11)
        x = _tensor(4, 4, rng)
        check_gradients(lambda: _scalarize(nn.softmax_rows(x, causal=True), seed=8), [x])

    def test_embed_repeated_id(self) -> None:
        """A row used twice must receive both gradients, not the last one."""
        rng = random.Random(12)
        weight = _tensor(6, 3, rng)
        check_gradients(lambda: _scalarize(nn.embed(weight, [0, 3, 3, 5])), [weight])

    def test_select_rows_repeated(self) -> None:
        rng = random.Random(13)
        x = _tensor(6, 4, rng)
        check_gradients(lambda: _scalarize(nn.select_rows(x, [4, 1, 1]), seed=11), [x])

    def test_cross_entropy_with_mask(self) -> None:
        rng = random.Random(14)
        logits = _tensor(4, 7, rng)
        check_gradients(
            lambda: nn.cross_entropy(logits, [2, 0, 6, 1], [1.0, 0.0, 1.0, 1.0]), [logits]
        )

    def test_shared_tensor_accumulates(self) -> None:
        """A tensor read twice gets the sum of both paths, not one of them."""
        rng = random.Random(15)
        x = _tensor(3, 3, rng)
        check_gradients(lambda: _scalarize(nn.add(nn.relu(x), nn.scale(x, 2.0))), [x])


class TestPolicyGradient:
    """The RL objective, and the sign that decides which way it learns."""

    def test_gradient_matches_finite_difference(self) -> None:
        rng = random.Random(21)
        logits = _tensor(4, 6, rng)
        actions = [0, 3, 5, 1]
        advantages = [1.2, -0.7, 0.0, -0.4]

        check_gradients(
            lambda: nn.policy_gradient(logits, actions, advantages), [logits]
        )

    def test_a_positive_advantage_raises_that_action_s_probability(self) -> None:
        """The sign convention, pinned. Backwards, this trains a policy to
        reproduce its own worst samples — and the loss curve looks normal
        the whole time."""
        logits = Tensor([0.0] * 4, (1, 4), requires_grad=True)
        before = _probability(logits, action=2)

        for _ in range(40):
            logits.zero_grad()
            nn.policy_gradient(logits, [2], [1.0]).backward()
            for i in range(len(logits.data)):
                logits.data[i] -= 0.5 * (logits.grad or [0.0] * 4)[i]

        assert _probability(logits, action=2) > before

    def test_a_negative_advantage_lowers_it(self) -> None:
        logits = Tensor([0.0] * 4, (1, 4), requires_grad=True)
        before = _probability(logits, action=2)

        for _ in range(40):
            logits.zero_grad()
            nn.policy_gradient(logits, [2], [-1.0]).backward()
            for i in range(len(logits.data)):
                logits.data[i] -= 0.5 * (logits.grad or [0.0] * 4)[i]

        assert _probability(logits, action=2) < before

    def test_a_zero_advantage_moves_nothing(self) -> None:
        """A group where every sample scored alike teaches nothing, and has to
        contribute nothing rather than a small arbitrary push."""
        rng = random.Random(23)
        logits = _tensor(3, 5, rng)
        logits.zero_grad()

        nn.policy_gradient(logits, [0, 1, 2], [0.0, 0.0, 0.0]).backward()

        assert all(g == 0.0 for g in (logits.grad or []))

    def test_mismatched_batch_shapes_are_refused(self) -> None:
        rng = random.Random(24)
        logits = _tensor(3, 5, rng)

        with pytest.raises(ValueError, match="different batches"):
            nn.policy_gradient(logits, [0, 1], [1.0, 1.0])


def _probability(logits: Tensor, *, action: int) -> float:
    row = logits.data
    top = max(row)
    exps = [math.exp(v - top) for v in row]
    return exps[action] / sum(exps)


class TestSemantics:
    """Forward values, not just their derivatives."""

    def test_causal_softmax_ignores_the_future(self) -> None:
        x = Tensor([0.0] * 9, (3, 3))
        out = nn.softmax_rows(x, causal=True)
        assert out.row(0) == pytest.approx([1.0, 0.0, 0.0])
        assert out.row(1) == pytest.approx([0.5, 0.5, 0.0])
        assert out.row(2) == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_causal_softmax_is_unaffected_by_masked_values(self) -> None:
        """Changing a masked entry must not change any output."""
        base = Tensor([0.0] * 9, (3, 3))
        poisoned = Tensor([0.0, 99.0, -99.0, 0.0, 0.0, 42.0, 0.0, 0.0, 0.0], (3, 3))
        assert nn.softmax_rows(base, causal=True).data == pytest.approx(
            nn.softmax_rows(poisoned, causal=True).data
        )

    def test_layer_norm_standardizes_each_row(self) -> None:
        x = Tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], (2, 3))
        gain = Tensor([1.0, 1.0, 1.0], (1, 3))
        bias = Tensor([0.0, 0.0, 0.0], (1, 3))
        out = nn.layer_norm(x, gain, bias)
        for row in (out.row(0), out.row(1)):
            assert sum(row) == pytest.approx(0.0, abs=1e-9)
            assert sum(v * v for v in row) / 3 == pytest.approx(1.0, abs=1e-4)

    def test_cross_entropy_of_uniform_logits_is_log_vocab(self) -> None:
        logits = Tensor([0.0] * 12, (3, 4))
        assert nn.cross_entropy(logits, [0, 1, 2]).data[0] == pytest.approx(math.log(4))

    def test_masked_rows_do_not_contribute(self) -> None:
        """A row the mask zeroes cannot change the loss, however wrong it is."""
        good = Tensor([5.0, 0.0, 0.0, 5.0, 0.0, 0.0], (2, 3))
        bad = Tensor([5.0, 0.0, 0.0, -9.0, 9.0, 0.0], (2, 3))
        mask = [1.0, 0.0]
        assert nn.cross_entropy(good, [0, 0], mask).data[0] == pytest.approx(
            nn.cross_entropy(bad, [0, 0], mask).data[0]
        )

    def test_cross_entropy_rejects_an_empty_mask(self) -> None:
        logits = Tensor([0.0] * 6, (2, 3))
        with pytest.raises(ValueError, match="unmasked"):
            nn.cross_entropy(logits, [0, 0], [0.0, 0.0])

    def test_matmul_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="cannot multiply"):
            nn.matmul(Tensor([0.0] * 6, (2, 3)), Tensor([0.0] * 6, (2, 3)))

    def test_backward_requires_a_scalar(self) -> None:
        with pytest.raises(ValueError, match="scalar"):
            Tensor([0.0] * 4, (2, 2), requires_grad=True).backward()


class TestAdam:
    """The optimizer moves downhill and respects its clip."""

    def test_minimizes_a_quadratic(self) -> None:
        x = Tensor([3.0, -4.0], (1, 2), requires_grad=True)
        opt = nn.Adam([x], lr=0.1)
        for _ in range(300):
            opt.zero_grad()
            loss = nn.matmul(x, nn.transpose(x))  # x . x
            loss.backward()
            opt.step()
        assert abs(x.data[0]) < 1e-2
        assert abs(x.data[1]) < 1e-2

    def test_clipping_bounds_the_applied_gradient(self) -> None:
        """A huge gradient must move the parameter by about one lr, not more."""
        x = Tensor([0.0], (1, 1), requires_grad=True)
        opt = nn.Adam([x], lr=0.01, clip=1.0)
        x.grad = [1e6]
        opt.step()
        assert abs(x.data[0]) < 0.02

    def test_reports_the_pre_clip_norm(self) -> None:
        x = Tensor([0.0, 0.0], (1, 2), requires_grad=True)
        opt = nn.Adam([x], lr=0.01, clip=1.0)
        x.grad = [3.0, 4.0]
        assert opt.step() == pytest.approx(5.0)
