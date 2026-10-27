"""
Calibrate the Verifier against MedAESQA human evidence-support labels (issue #32).

Each test case is one (evidence excerpt, answer sentence) pair that a human labeled:
  supporting               -> human_label 1 (sentence is supported)
  neutral / contradicting  -> human_label 0 (sentence is not supported)
Pairs labeled "not relevant" or "invalid citation" have no evidence text, so they're skipped.

The Verifier sees the excerpt as original_text and the sentence as simplified_text.
verifier_label is 1 only if both its faithfulness and unsupported_additions checks pass.

Run from the project root:
  python -m notebooks.calibrate_verifier --n 100      # call Gemini, then print metrics
  python -m notebooks.calibrate_verifier --report     # metrics only, from saved results
  python -m notebooks.calibrate_verifier --fresh      # start over after changing the Verifier
"""
import argparse
import asyncio
import csv
import random
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.verifier import check_verifier_output, make_verifier_input, verifier_agent
from notebooks.eval_harness import load_medaesqa, verifier_agreement

OUT = Path(__file__).resolve().parent.parent / "data" / "verifier_calibration.csv"
APP, USER = "verifier_calibration", "calibration"
SUPPORTED, UNSUPPORTED = {"supporting"}, {"neutral", "contradicting"}
CITATION = re.compile(r"\s*\[[\d,\s]+\]")
FIELDS = ["id", "relation", "human_label", "verifier_label",
          "faithfulness_issues", "unsupported_issues", "evidence", "sentence"]


def build_pairs(data):
    pairs = {}
    for q in data:
        for method, answer in q["machine_generated_answers"].items():
            for s in answer["answer_sentences"]:
                for c in s["citation_assessment"] or []:
                    relation, evidence = c["evidence_relation"], c["evidence_support"]
                    if not evidence or relation not in SUPPORTED | UNSUPPORTED:
                        continue
                    pid = f'{q["question_id"]}-{method}-{s["answer_sentence_id"]}-{c["cited_pmid"]}'
                    pairs[pid] = {
                        "id": pid,
                        "relation": relation,
                        "human_label": int(relation in SUPPORTED),
                        "evidence": evidence.strip(),
                        "sentence": CITATION.sub("", s["answer_sentence"]).strip(),
                    }
    return list(pairs.values())


def sample_balanced(pairs, n, seed=42):
    rng = random.Random(seed)
    pos = [p for p in pairs if p["human_label"] == 1]
    neg = [p for p in pairs if p["human_label"] == 0]
    k = min(n // 2, len(pos), len(neg))
    sample = rng.sample(pos, k) + rng.sample(neg, k)
    rng.shuffle(sample)
    return sample


async def verify(runner, evidence, sentence, retries=3):
    message = types.Content(role="user", parts=[types.Part(text=make_verifier_input(evidence, sentence))])
    for attempt in range(retries):
        try:
            session = await runner.session_service.create_session(app_name=APP, user_id=USER)
            async for _ in runner.run_async(user_id=USER, session_id=session.id, new_message=message):
                pass
            session = await runner.session_service.get_session(
                app_name=APP, user_id=USER, session_id=session.id)
            # Re-derive pass flags from the issues so the model can't contradict itself
            return check_verifier_output(session.state["verifier_feedback"])
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  retrying in {wait}s ({type(e).__name__})")
            await asyncio.sleep(wait)


def load_results():
    if not OUT.exists():
        return []
    with open(OUT, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def run(n, seed):
    sample = sample_balanced(build_pairs(load_medaesqa()), n, seed)
    done = {r["id"] for r in load_results()}
    todo = [p for p in sample if p["id"] not in done]
    print(f"{len(sample)} pairs sampled, {len(done)} already saved, {len(todo)} to run")

    runner = InMemoryRunner(agent=verifier_agent, app_name=APP)
    new_file = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, p in enumerate(todo, 1):
            try:
                fb = await verify(runner, p["evidence"], p["sentence"])
            except Exception as e:
                print(f"[{i}/{len(todo)}] {p['id']} failed: {e}")
                continue
            writer.writerow({
                **p,
                "verifier_label": int(fb.faithfulness.passed and fb.unsupported_additions.passed),
                "faithfulness_issues": " | ".join(x.description for x in fb.faithfulness.issues),
                "unsupported_issues": " | ".join(x.description for x in fb.unsupported_additions.issues),
            })
            f.flush()
            print(f"[{i}/{len(todo)}] {p['id']} done")


def report(rows):
    if not rows:
        print(f"No results yet in {OUT}")
        return
    human = [int(r["human_label"]) for r in rows]
    verifier = [int(r["verifier_label"]) for r in rows]
    stats = verifier_agreement(human, verifier)
    cells = Counter(zip(human, verifier))

    print(f"\nn = {stats['n']}")
    print(f"Agreement: {stats['accuracy']:.1%}  (target >= 80%)  "
          f"{'PASS' if stats['accuracy'] >= 0.8 else 'FAIL'}")
    print(f"Cohen's kappa: {stats['cohen_kappa']:.3f}  (target >= 0.6)  "
          f"{'PASS' if stats['cohen_kappa'] >= 0.6 else 'FAIL'}")

    print("\n                       Verifier: supported   Verifier: not supported")
    print(f"Human: supported       {cells[(1, 1)]:>19}   {cells[(1, 0)]:>23}")
    print(f"Human: not supported   {cells[(0, 1)]:>19}   {cells[(0, 0)]:>23}")

    print("\nAgreement by human label:")
    for relation in ["supporting", "neutral", "contradicting"]:
        group = [r for r in rows if r["relation"] == relation]
        if group:
            agree = sum(r["human_label"] == r["verifier_label"] for r in group)
            print(f"  {relation:<14} {agree}/{len(group)} ({agree / len(group):.0%})")

    print(f"\nReview disagreements in {OUT} (rows where human_label != verifier_label)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="pairs to sample (half supported, half not)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true", help="skip Gemini, just print metrics")
    parser.add_argument("--fresh", action="store_true", help="delete saved results first (use after editing the Verifier)")
    args = parser.parse_args()

    load_dotenv()
    if args.fresh:
        OUT.unlink(missing_ok=True)
    if not args.report:
        asyncio.run(run(args.n, args.seed))
    report(load_results())