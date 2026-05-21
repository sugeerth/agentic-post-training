"""`agentic-train` command-line interface.

Replaces the broken `pipeline.cli:main` entry point that was declared in
pyproject but never existed. Three subcommands:

  - `agentic-train run <run.yaml> [--backend X] [--dry-run]`
  - `agentic-train list techniques|backends`
  - `agentic-train inspect <name>`

Kept deliberately small. Heavy logic lives in `core/`, `backends/`,
and the technique modules; this file is just argparse + dispatch.
"""

from __future__ import annotations

import argparse
import json
import sys

# Side-effect imports: each module's @register_* decorator runs on import.
# Without these the registries appear empty when the CLI starts.
import backends  # noqa: F401  - registers LocalBackend
import techniques  # noqa: F401  - imports each technique, populating registry
from core.registry import (
    get_backend,
    get_technique,
    list_backends,
    list_techniques,
)
from pipeline.config import PipelineConfig


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.from_yaml(args.spec)
    errs = cfg.validate()
    if errs:
        print("Run-spec failed validation:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 2

    backend_cls = get_backend(args.backend)
    backend = backend_cls()

    from backends.local import make_job
    job = make_job(
        technique=cfg.technique,
        model_name=cfg.model_name,
        output_dir=cfg.output_dir,
        technique_config={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "max_length": cfg.max_length,
            **cfg.technique_config,
        },
    )

    if args.dry_run:
        plan = backend.dry_run(job)
        print(json.dumps(plan, indent=2, default=str))
        return 0

    handle = backend.launch(job)
    print(json.dumps(
        {
            "job_id": handle.job_id,
            "status": handle.status.value,
            "backend": handle.backend,
            "last_checkpoint": handle.last_checkpoint,
            "metrics": handle.metrics,
            "error": handle.error,
        },
        indent=2,
        default=str,
    ))
    return 0 if handle.status.value == "completed" else 1


def _cmd_list(args: argparse.Namespace) -> int:
    if args.kind == "techniques":
        for name in list_techniques():
            cls = get_technique(name)
            marker = "  [experimental]" if getattr(cls, "is_experimental", False) else ""
            print(f"  {name}{marker}")
    elif args.kind == "backends":
        for name in list_backends():
            print(f"  {name}")
    else:
        print(f"unknown kind: {args.kind}", file=sys.stderr)
        return 2
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        cls = get_technique(args.name)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 2
    info = {
        "name": getattr(cls, "name", args.name),
        "class": cls.__name__,
        "experimental": getattr(cls, "is_experimental", False),
    }
    print(json.dumps(info, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a YAML run-spec")
    p_run.add_argument("spec", help="Path to a run.yaml")
    p_run.add_argument("--backend", default="local", help="Backend to dispatch to (default: local)")
    p_run.add_argument("--dry-run", action="store_true", help="Print the plan without spending compute")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list", help="List registered components")
    p_list.add_argument("kind", choices=["techniques", "backends"])
    p_list.set_defaults(func=_cmd_list)

    p_inspect = sub.add_parser("inspect", help="Show details for a technique")
    p_inspect.add_argument("name")
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
