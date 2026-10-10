"""Command line for the training and validation pipeline.

    python -m forewarn.cli all --customers 100000

Steps (each can be run alone, in this order):
    simulate   synthetic bank -> data/raw/*.parquet
    features   point-in-time features and labels -> data/work/
    train      delinquency and bankruptcy models with comparators
    recommend  action value models and cross-fitted policy evaluation
    export     model bundle and parity cases -> artifacts/bundle/
    validate   out-of-time validation evidence -> artifacts/metrics/validation.json
    report     validation report (HTML) and model card (Markdown)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .store import Store

STEPS = ["simulate", "features", "train", "recommend", "export", "validate", "report"]


def step_simulate(store: Store, args) -> None:
    from .simulate import SimConfig, simulate

    tables = simulate(SimConfig(n_customers=args.customers, seed=args.seed))
    store.write_raw(tables)
    print(f"  {len(tables['customers']):,} accounts, {len(tables['dda_transactions']):,} deposit and {len(tables['card_transactions']):,} card transactions")


def step_features(store: Store, args) -> None:
    from .features import assign_split, build_features, build_labels

    stmts = store.read_raw("card_statements")
    snap, panel = build_features(store.read_raw("customers"), store.read_raw("dda_transactions"),
                                 store.read_raw("card_transactions"), stmts)
    labels = build_labels(snap, panel, stmts, store.read_raw("outcomes"), store.read_raw("interventions"))
    labels["split"] = assign_split(labels["snapshot_month"].to_numpy(), panel.n_months)
    snap.to_parquet(store.work / "features.parquet", index=False)
    labels.to_parquet(store.work / "labels.parquet", index=False)
    print(f"  {len(snap):,} account-month snapshots over {panel.n_months} months")


def step_train(store: Store, args) -> None:
    from .train import train

    train(store, train_sample=args.train_sample, seed=args.seed, log=lambda m: print("  " + m))


def step_recommend(store: Store, args) -> None:
    from .recommend import evaluate

    evaluate(store, seed=args.seed, log=lambda m: print(m))


def step_export(store: Store, args) -> None:
    from .export import export

    export(store, seed=args.seed, log=lambda m: print("  " + m))


def step_validate(store: Store, args) -> None:
    from .validate import validate

    validate(store, seed=args.seed, log=lambda m: print("  " + m))


def step_report(store: Store, args) -> None:
    from .report import render

    for p in render(store):
        print(f"  wrote {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="forewarn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]), help="project root (data/ and artifacts/ live here)")
    ap.add_argument("--customers", type=int, default=100_000, help="simulated customers (20000 is enough for a quick look)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-sample", type=float, default=1.0, help="fraction of training rows to use")
    args = ap.parse_args(argv)
    store = Store(args.root)
    for step in (STEPS if args.step == "all" else [args.step]):
        t0 = time.time()
        print(f"[{step}]", flush=True)
        globals()[f"step_{step}"](store, args)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
