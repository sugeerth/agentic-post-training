"""Technique scout — extracts algorithm names + claims from paper abstracts.

Deliberately regex-based, not learned. Two reasons:

  1. Auditable — every extraction rule shows up as one line of the digest.
  2. Robust to abstract style drift — an LLM extractor at 3 AM every day
     with no eval is a bug factory.

Extraction pipeline:
  • find acronyms (2-10 uppercase letters, optionally hyphenated) near
    signal verbs — "we propose", "we introduce", "our method", "we present"
  • find "outperforms X by N%" and "N% higher on TASK" claims
  • flag mentions of KL / reward / success / calibration
  • dedupe against known techniques so we don't reinvent GRPO
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import BaseAgent


# Techniques we already ship — abstracts mentioning only these produce
# no new mimic. Add the acronym form users would actually see.
KNOWN_TECHNIQUES: set[str] = {
    "GRPO", "DPO", "PPO", "RLHF", "RLAIF", "KTO", "ORPO", "SPIN", "SimPO",
    "IPO", "SPO", "SFT", "RFT", "STAR", "PRM", "ORM",
}


# Signal phrases that immediately precede the method acronym in most
# NeurIPS/ICLR abstracts. Order-sensitive: more specific first.
_PROPOSE = re.compile(
    r"\b(?:we\s+(?:propose|introduce|present|describe)|our\s+(?:method|approach))\b"
    r"[^.]{0,80}?"
    r"\b([A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*){0,3})\b",
    re.IGNORECASE,
)
_OUTPERFORMS = re.compile(
    r"\b(?:outperforms?|beats?|exceeds?)\s+([A-Z][A-Za-z0-9-]*)"
    r"(?:\s+by\s+(\d+(?:\.\d+)?)\s*(?:%|points?|pp))?",
    re.IGNORECASE,
)
_GAIN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|points?|pp)\s+(?:higher|better|improvement|gain)\s+"
    r"(?:on|in)\s+([A-Za-z0-9-@]+)",
    re.IGNORECASE,
)


@dataclass
class ScoutedTechnique:
    """One candidate technique extracted from a single paper."""
    name: str                                       # e.g. "MT-AGRPO"
    slug: str                                       # snake_case, file-safe
    paper_id: str
    paper_title: str
    paper_url: str
    novelty: str                                    # ≤120-char one-liner
    outperforms: list[tuple[str, float | None]] = field(default_factory=list)
    gains: list[tuple[float, str]] = field(default_factory=list)
    mentions_kl: bool = False
    mentions_reward_hacking: bool = False
    mentions_process_reward: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class TechniqueScout(BaseAgent):
    def __init__(self, name: str = "TechniqueScout"):
        super().__init__(name, role="technique_scout")
        self.register_capability("extract_names", "Find algorithm acronyms")
        self.register_capability("extract_claims", "Parse outperforms/gain phrases")

    def scout(self, paper: dict[str, Any]) -> list[ScoutedTechnique]:
        """Extract candidate techniques from a paper.

        Precedence: abstract's "we propose X" wins. Title mining only runs
        as a fallback (when the abstract yielded nothing), and applies a
        tighter filter that rejects descriptive-adjective compounds like
        "Multi-Turn" or "Tool-Using".
        """
        abstract = paper.get("abstract", "") or ""
        title = paper.get("title", "") or ""

        found: list[ScoutedTechnique] = []
        seen_names: set[str] = set()

        for m in _PROPOSE.finditer(abstract):
            candidate = m.group(1).strip("-")
            if not _looks_like_algorithm(candidate):
                continue
            up = candidate.upper()
            if up in KNOWN_TECHNIQUES or up in seen_names:
                continue
            seen_names.add(up)
            found.append(_build(candidate, paper, abstract))

        if found:
            return found  # abstract was informative — don't muddy with title guesses

        # Fallback: mine the title. Tighter filter — must clear the
        # algorithm-shape test that rejects "Multi-Turn" style compounds.
        for word in re.findall(r"\b([A-Z][A-Za-z0-9]*(?:-[A-Z0-9][A-Za-z0-9]*){0,3})\b", title):
            up = word.upper()
            if up in KNOWN_TECHNIQUES or up in seen_names:
                continue
            if not _looks_like_algorithm(word):
                continue
            seen_names.add(up)
            found.append(_build(word, paper, abstract))

        return found

    async def run(self, **kwargs) -> dict[str, Any]:
        papers: list[dict] = kwargs.get("papers", [])
        techniques: list[ScoutedTechnique] = []
        for p in papers:
            techniques.extend(self.scout(p))

        await self.send_message("task_result", {
            "message": f"Found {len(techniques)} candidate techniques across {len(papers)} papers",
            "count": len(techniques),
        }, target="Coordinator")

        return {
            "papers_scanned": len(papers),
            "techniques_found": len(techniques),
            "techniques": [t.to_dict() for t in techniques],
        }

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)


_DESCRIPTIVE_HEADS = {
    "MULTI", "SELF", "TOOL", "TRAJECTORY", "VERIFIER", "ADAPTIVE", "ONLINE",
    "OFFLINE", "MODEL", "REWARD", "PROCESS", "OUTCOME", "GROUP", "GROUPED",
    "REGULARIZED", "STABLE",
}


def _looks_like_algorithm(word: str) -> bool:
    """Filter for algorithm names: 3+ chars, mostly caps or contains digits,
    and not a descriptive-adjective compound like Multi-Turn / Tool-Using.

    Real method names have a high uppercase ratio (GRPO, MT-AGRPO, VG-RFT).
    Descriptive-adjective compounds have a Title-Case-With-Hyphens shape
    where each chunk starts uppercase but is mostly lowercase.
    """
    if len(word) < 3:
        return False
    # Reject if the leading chunk is a descriptive adjective ("Multi-*", "Tool-*").
    first_chunk = word.split("-", 1)[0].upper()
    if first_chunk in _DESCRIPTIVE_HEADS:
        return False
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    # Algorithms are ≥50% uppercase; "Multi-Turn" is ~40%, "GRPO"/"MT-AGRPO"/"TrajDPO" are ≥60%.
    return upper_ratio >= 0.5 or any(c.isdigit() for c in word)


def _build(name: str, paper: dict, abstract: str) -> ScoutedTechnique:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    outperforms = [
        (opp.group(1).upper(), float(opp.group(2)) if opp.group(2) else None)
        for opp in _OUTPERFORMS.finditer(abstract)
    ]
    gains = [(float(g.group(1)), g.group(2)) for g in _GAIN.finditer(abstract)]
    novelty = _first_sentence(abstract)[:120]

    return ScoutedTechnique(
        name=name,
        slug=slug,
        paper_id=paper.get("arxiv_id", ""),
        paper_title=paper.get("title", ""),
        paper_url=paper.get("url", ""),
        novelty=novelty,
        outperforms=outperforms,
        gains=gains,
        mentions_kl="kl" in abstract.lower() or "divergence" in abstract.lower(),
        mentions_reward_hacking="reward hacking" in abstract.lower(),
        mentions_process_reward="process reward" in abstract.lower(),
    )


def _first_sentence(text: str) -> str:
    m = re.search(r"[^.]+\.", text)
    return m.group(0).strip() if m else text.strip()
