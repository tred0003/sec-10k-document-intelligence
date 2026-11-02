"""Step 5d: score retrieval modes and answers on the auto-labeled questions.

Retrieval: a hit when any of the top-k chunks is on a page that prints the
answer (results/questions.json relevant_pages). MRR@10 = 1/rank of the first hit.
Answers: correct within rounding of the official value, wrong, or abstained
(found=false). The mode for answers is picked on dev only, then run on test.

Run:  python rag_eval.py retrieval        (all modes, dev + test)
      python rag_eval.py answers <mode>   (answers with that mode, dev + test)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rag
from extract import SCALE

RESULTS = Path(__file__).parent / "results"
MODES = ["bm25", "dense", "dense_rank", "hybrid_rank"]


def questions():
    return json.loads((RESULTS / "questions.json").read_text())


def run_retrieval(mode, qs):
    cache = RESULTS / f"retrieval_{mode}.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [q for q in qs if q["id"] not in done]

    def one(q):
        return q["id"], [r["page"] for r in rag.search(q["question"], q["ticker"], mode)[:10]]

    with ThreadPoolExecutor(6) as pool:
        for qid, pages in pool.map(one, todo):
            done[qid] = pages
    cache.write_text(json.dumps(done))
    return done


def retrieval_report(modes=MODES):
    qs = questions()
    print(f"{'mode':12} {'split':5} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} {'MRR@10':>7}")
    for mode in modes:
        got = run_retrieval(mode, qs)
        for split in ("dev", "test"):
            sub = [q for q in qs if q["split"] == split]
            hit = {k: 0 for k in (1, 5, 10)}
            mrr = 0.0
            for q in sub:
                rel = set(q["relevant_pages"])
                ranks = [i for i, p in enumerate(got[q["id"]], 1) if p in rel]
                for k in hit:
                    hit[k] += bool(ranks and ranks[0] <= k)
                mrr += 1 / ranks[0] if ranks else 0
            n = len(sub)
            print(f"{mode:12} {split:5} {hit[1] / n:6.1%} {hit[5] / n:6.1%} {hit[10] / n:7.1%} {mrr / n:7.3f}")


def correct(got, want):
    # Compare magnitudes: cash flow statements print payments as outflows, "(90,711)",
    # while XBRL stores PaymentsFor... as positive, and the questions ask "how much".
    if got is None:
        return False
    got, want = abs(got), abs(want)
    return abs(got - want) <= max(0.001 * want, 0.5e6 if want >= 1e9 else 1)


def rounded_ok(a, want):
    """Right to the precision printed: "$5.8 billion" for 5,787 million. Kept as its own
    bucket so strict accuracy is still reported, not quietly loosened."""
    printed = (a.get("printed") or "").strip("()$ ").replace(",", "")
    if a.get("value") is None or not printed:
        return False
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    half_unit = 0.5 * 10 ** -decimals * SCALE[a["scale"]]
    return abs(abs(a["value"]) - abs(want)) <= half_unit + 1


def answers_report(mode):
    qs = questions()
    cache = RESULTS / f"answers_{mode}.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [q for q in qs if q["id"] not in done]

    def one(q):
        try:
            return q["id"], rag.answer(q["question"], q["ticker"], q["company"], mode)
        except Exception as e:  # one bad call should not lose the run
            return q["id"], {"error": str(e)[:200]}

    with ThreadPoolExecutor(6) as pool:
        for qid, a in pool.map(one, todo):
            done[qid] = a
    cache.write_text(json.dumps(done, indent=1))

    for split in ("dev", "test"):
        sub = [q for q in qs if q["split"] == split]
        c = {"correct": 0, "rounded": 0, "wrong": 0, "abstained": 0, "error": 0}
        wrong_flagged = correct_flagged = 0
        for q in sub:
            a = done[q["id"]]
            if "error" in a:
                c["error"] += 1
                continue
            if not a["found"] or a["value"] is None:
                c["abstained"] += 1
            elif correct(a["value"], q["answer"]):
                c["correct"] += 1
                correct_flagged += bool(a["checks"])
            elif rounded_ok(a, q["answer"]):
                c["rounded"] += 1
            else:
                c["wrong"] += 1
                wrong_flagged += bool(a["checks"])
        n = len(sub)
        answered = c["correct"] + c["rounded"] + c["wrong"]
        print(f"[{split}] {n} questions: exact {c['correct']} ({c['correct'] / n:.1%}), "
              f"right at printed precision {c['rounded']} (together {(c['correct'] + c['rounded']) / n:.1%}), "
              f"wrong {c['wrong']}, abstained {c['abstained']}, errors {c['error']}")
        print(f"   precision when it answers: {(c['correct'] + c['rounded']) / max(answered, 1):.1%}; "
              f"citation checks flagged {wrong_flagged}/{c['wrong']} wrong, {correct_flagged}/{c['correct']} correct")


if __name__ == "__main__":
    if sys.argv[1] == "retrieval":
        retrieval_report()
    else:
        answers_report(sys.argv[2])
