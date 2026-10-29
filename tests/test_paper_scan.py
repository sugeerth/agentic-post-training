"""Tests for the daily paper-scan pipeline.

Covers extraction correctness (no title-derived noise, known techniques
filtered out), codegen shape (files import + step), and pipeline
idempotence (running twice on the same day doesn't churn).
"""

from __future__ import annotations

import asyncio
import importlib
import tempfile
from pathlib import Path

import pytest

from agents.mimic_writer import MimicWriter
from agents.paper_scanner import PaperScanner
from agents.technique_scout import TechniqueScout, _looks_like_algorithm


# --------------------------------------------------------------------------- #
# Algorithm-name filter
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("word", ["GRPO", "MT-AGRPO", "VG-RFT", "MC-PRM", "SPIN-2", "TrajDPO"])
def test_looks_like_algorithm_accepts_real_acronyms(word):
    assert _looks_like_algorithm(word)


@pytest.mark.parametrize("word", ["Multi-Turn", "Tool-Using", "Trajectory-Aware",
                                  "Adaptive", "Self-Play", "Group-Wise"])
def test_looks_like_algorithm_rejects_descriptive_compounds(word):
    assert not _looks_like_algorithm(word)


# --------------------------------------------------------------------------- #
# PaperScanner
# --------------------------------------------------------------------------- #

def test_canned_papers_have_expected_shape():
    scanner = PaperScanner()
    papers = scanner.canned()
    assert len(papers) >= 5
    for p in papers:
        assert p.arxiv_id and p.title and p.abstract
        # published is YYYY-MM-DD
        assert len(p.published) == 10 and p.published[4] == "-"


def test_scanner_run_no_network_uses_canned():
    scanner = PaperScanner()
    result = asyncio.run(scanner.run(allow_network=False))
    assert result["source"] == "canned"
    assert result["count"] >= 5


# --------------------------------------------------------------------------- #
# TechniqueScout — one technique per paper, no title-derived noise
# --------------------------------------------------------------------------- #

def test_scout_extracts_one_technique_per_canned_paper():
    scanner = PaperScanner()
    scout = TechniqueScout()
    papers = [p.__dict__ for p in scanner.canned()]
    per_paper = [len(scout.scout(p)) for p in papers]
    # Every canned abstract has exactly one "we propose X" — should yield 1.
    assert per_paper == [1] * len(papers), f"got {per_paper}"


def test_scout_skips_known_techniques():
    """An abstract that only names GRPO/DPO should produce nothing new."""
    scout = TechniqueScout()
    fake = {
        "arxiv_id": "0000.0001", "title": "Comparing GRPO and DPO",
        "abstract": "We compare GRPO and DPO. GRPO wins.",
        "url": "https://arxiv.org/abs/0000.0001",
    }
    assert scout.scout(fake) == []


# --------------------------------------------------------------------------- #
# MimicWriter — codegen produces importable, working stubs
# --------------------------------------------------------------------------- #

def test_mimic_writer_produces_importable_stub():
    with tempfile.TemporaryDirectory() as tmp:
        w = MimicWriter(root=tmp)
        technique = {
            "name": "MT-AGRPO",
            "slug": "mt_agrpo",
            "paper_id": "2501.01234v1",
            "paper_title": "Test",
            "paper_url": "https://arxiv.org/abs/2501.01234",
            "novelty": "test",
            "outperforms": [("GRPO", 8.0)],
            "gains": [],
            "mentions_kl": True,
            "mentions_reward_hacking": False,
            "mentions_process_reward": False,
        }
        path = w.write(technique)
        assert path.exists()

        # The stub must be actually importable + callable.
        import importlib.util
        spec = importlib.util.spec_from_file_location("mimic_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cls = mod.MTAGRPO   # _class_name preserves all-caps hyphenated acronyms
        assert cls.name == "mt_agrpo"
        m = cls()
        step_metrics = m.step(1)
        assert set(step_metrics) == {"loss", "reward", "kl_divergence"}
        assert 0 < step_metrics["reward"] <= 1
        # Bigger "outperforms by 8%" claim should raise gain above baseline.
        assert m.gain >= 0.10


def test_mimic_writer_is_idempotent_without_overwrite():
    """Second run of the pipeline on the same day should skip existing files."""
    with tempfile.TemporaryDirectory() as tmp:
        w = MimicWriter(root=tmp)
        techniques = [{
            "name": "FOO", "slug": "foo",
            "paper_id": "1", "paper_title": "t", "paper_url": "u",
            "novelty": "n", "outperforms": [], "gains": [],
            "mentions_kl": False, "mentions_reward_hacking": False,
            "mentions_process_reward": False,
        }]
        r1 = asyncio.run(w.run(techniques=techniques))
        r2 = asyncio.run(w.run(techniques=techniques))
        assert r1["written"] == ["foo"] and r1["skipped"] == []
        assert r2["written"] == [] and r2["skipped"] == ["foo"]


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def test_full_pipeline_writes_mimics_and_a_digest():
    from pipeline.paper_scan_pipeline import PaperScanPipeline
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        mimics = Path(tmp) / "mimics"
        pipe = PaperScanPipeline(out_dir=str(out), mimic_root=str(mimics))
        summary = asyncio.run(pipe.run(allow_network=False))

        assert summary["papers"] >= 5
        assert summary["mimics_written"] >= 1
        assert (out / "digest.md").exists()
        assert (out / "digest.json").exists()
        # Each mimic file loads cleanly.
        for f in mimics.glob("*.py"):
            spec = importlib.util.spec_from_file_location(f.stem, f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
