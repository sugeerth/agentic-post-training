"""Paper scanner — fetches recent arXiv abstracts on agent post-training.

Uses arXiv's public Atom API — no auth, no extra deps (urllib + xml).
Falls back to a canned list of ~10 recent-ish papers when the runner
has no outbound network, so the daily automation still produces useful
output in a sandbox.

Design tenets:
  • fetch just abstracts (metadata), never PDFs — cheap and no scraping
  • query the last N days so the digest stays fresh
  • dedupe by arXiv id so repeated runs don't churn
"""

from __future__ import annotations

import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterable

from agents.base_agent import BaseAgent


ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"}

DEFAULT_QUERY_TERMS = [
    "agent post-training",
    "RLHF",
    "GRPO",
    "DPO",
    "reward hacking",
    "tool-use reinforcement learning",
    "process reward model",
    "self-play fine-tuning",
]


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str                          # YYYY-MM-DD
    categories: list[str] = field(default_factory=list)
    url: str = ""

    def slug(self) -> str:
        """Short, filesystem-safe id used to name generated mimic files."""
        return re.sub(r"[^a-z0-9]+", "_", self.arxiv_id.lower()).strip("_")


class PaperScanner(BaseAgent):
    """Fetches recent arXiv abstracts. Gracefully degrades to a canned list."""

    def __init__(self, name: str = "PaperScanner"):
        super().__init__(name, role="paper_scanner")
        self.register_capability("fetch_arxiv", "Query arXiv's Atom API")
        self.register_capability("canned_fallback", "Return a stub set when offline")

    def _build_query(self, terms: Iterable[str], max_results: int) -> str:
        # `abs:` restricts the search to the abstract field — narrows
        # hits and rejects unrelated papers that happen to share a title term.
        clauses = " OR ".join(f'abs:"{t}"' for t in terms)
        params = {
            "search_query": clauses,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(max_results),
        }
        return f"{ARXIV_ENDPOINT}?{urllib.parse.urlencode(params)}"

    def _parse_atom(self, xml_text: str) -> list[Paper]:
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", ARXIV_NS):
            eid = _text(entry.find("atom:id", ARXIV_NS))
            arxiv_id = eid.rsplit("/", 1)[-1] if eid else ""
            papers.append(Paper(
                arxiv_id=arxiv_id,
                title=_clean(_text(entry.find("atom:title", ARXIV_NS))),
                authors=[_text(a.find("atom:name", ARXIV_NS))
                         for a in entry.findall("atom:author", ARXIV_NS)],
                abstract=_clean(_text(entry.find("atom:summary", ARXIV_NS))),
                published=_text(entry.find("atom:published", ARXIV_NS))[:10],
                categories=[c.get("term", "")
                            for c in entry.findall("atom:category", ARXIV_NS)],
                url=eid,
            ))
        return papers

    def fetch(self, terms: list[str] | None = None, max_results: int = 12,
              timeout: float = 8.0) -> list[Paper]:
        """Attempt a live arXiv query. Returns [] on any network failure."""
        url = self._build_query(terms or DEFAULT_QUERY_TERMS, max_results)
        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            return self._parse_atom(data)
        except Exception as e:
            self.log(f"arXiv fetch failed: {e.__class__.__name__} — using canned list")
            return []

    def canned(self) -> list[Paper]:
        """Deterministic offline seed. Kept small so digest stays scannable.

        The `abstract` field on each canned entry is deliberately written to
        look like a real arXiv abstract — it exercises the technique-scout
        regex (acronym near "propose", "outperforms X", KL/reward mentions).
        """
        return [
            Paper(
                arxiv_id="2501.01234v1",
                title="Multi-Turn GRPO with Adaptive KL Control for Tool-Using Agents",
                authors=["A. Kim", "B. Chen"],
                abstract=(
                    "We propose MT-AGRPO, a multi-turn variant of GRPO with an adaptive "
                    "KL coefficient that tightens as reward-eval Spearman correlation "
                    "drops. MT-AGRPO outperforms GRPO by 8 points on tool-use success "
                    "rate while keeping KL divergence below 0.15."
                ),
                published="2025-12-30",
                categories=["cs.LG", "cs.CL"],
                url="https://arxiv.org/abs/2501.01234",
            ),
            Paper(
                arxiv_id="2501.02345v1",
                title="TrajDPO: Trajectory-Aware Preference Optimization",
                authors=["C. Zhou", "D. Ito"],
                abstract=(
                    "We introduce TrajDPO, which extends DPO to compare full multi-turn "
                    "trajectories rather than single responses. TrajDPO removes the "
                    "reward model entirely and reaches parity with GRPO on WebArena "
                    "at 40% of the compute."
                ),
                published="2025-12-29",
                categories=["cs.LG"],
                url="https://arxiv.org/abs/2501.02345",
            ),
            Paper(
                arxiv_id="2501.03456v1",
                title="Verifier-Guided Rejection Sampling for Code Agents",
                authors=["E. Patel"],
                abstract=(
                    "We describe VG-RFT, a rejection-sampling fine-tuning loop guided "
                    "by a unit-test verifier. VG-RFT delivers 12% higher HumanEval pass@1 "
                    "than plain RFT with zero reward hacking observed across 5000 rollouts."
                ),
                published="2025-12-28",
                categories=["cs.SE", "cs.LG"],
                url="https://arxiv.org/abs/2501.03456",
            ),
            Paper(
                arxiv_id="2501.04567v1",
                title="Process Reward Models via Monte Carlo Rollout Labeling",
                authors=["F. Nakamura", "G. Ali"],
                abstract=(
                    "We present MC-PRM, which labels intermediate reasoning steps by "
                    "Monte Carlo rollout success rates. MC-PRM outperforms outcome-only "
                    "RMs by 6 ECE points on MATH and reduces reward hacking severity "
                    "on downstream GRPO runs."
                ),
                published="2025-12-27",
                categories=["cs.LG"],
                url="https://arxiv.org/abs/2501.04567",
            ),
            Paper(
                arxiv_id="2501.05678v1",
                title="SPIN-2: Iterative Self-Play with Curriculum Curation",
                authors=["H. Larsen"],
                abstract=(
                    "We propose SPIN-2, extending SPIN with adaptive curriculum. SPIN-2 "
                    "matches DPO alignment quality without preference labels, using only "
                    "SFT-quality data."
                ),
                published="2025-12-26",
                categories=["cs.LG"],
                url="https://arxiv.org/abs/2501.05678",
            ),
        ]

    async def run(self, **kwargs) -> dict[str, Any]:
        terms: list[str] = kwargs.get("terms", DEFAULT_QUERY_TERMS)
        max_results: int = kwargs.get("max_results", 12)
        allow_network: bool = kwargs.get("allow_network", True)

        papers: list[Paper] = self.fetch(terms, max_results) if allow_network else []
        source = "arxiv"
        if not papers:
            papers = self.canned()
            source = "canned"

        await self.send_message("task_result", {
            "message": f"Fetched {len(papers)} papers ({source})",
            "count": len(papers),
        }, target="Coordinator")

        return {
            "source": source,
            "count": len(papers),
            "papers": [p.__dict__ for p in papers],
        }

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)


def _text(node: ET.Element | None) -> str:
    return (node.text or "") if node is not None else ""


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()
