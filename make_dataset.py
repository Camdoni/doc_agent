"""
make_dataset.py

WHAT THIS SCRIPT DOES:
Builds the training data for the LoRA fine-tune, in three steps you run in order:

    python make_dataset.py draft     # llama3.2 drafts question/answer pairs from your notes
    python make_dataset.py review    # YOU accept, edit, or reject every draft
    python make_dataset.py build     # writes train.jsonl and val.jsonl from what you accepted

Everything is saved in the "dataset" folder inside your docagent data folder.

WHAT A TRAINING EXAMPLE LOOKS LIKE:
One example is a whole conversation, exactly like a real run of the agent:
    system prompt -> your question -> the model calls search_docs ->
    the real search result comes back -> the model writes the answer
Training on the real format (including the real search results) teaches the
model the habit we want: answer ONLY from what search_docs returned.

THREE KINDS OF EXAMPLES:
    answerable  question drafted from a chunk; answer sticks to that chunk
    refusal     off-topic question; the right answer is "I couldn't find that"
    smalltalk   "hi" -> a short greeting, with no search (fixes the "{}" reply)

WHY YOU REVIEW: the drafting model sometimes adds facts that are not in your
notes. If those get into the training data, the tuned model learns to do it too.
"""

import os
import re
import sys
import json
import random
import argparse

import requests

import agent
from config import resolve_vault, data_dir_for
from ingest import ensure_index, load_embedder
from ollama_setup import ensure_ollama

DATASET_DIR = "dataset"
DRAFTS_FILE = "drafts.jsonl"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"

VAL_FRACTION = 0.2
FLAG_THRESHOLD = 0.35   # flag an answer if more than 35% of its words are not in the passage
MIN_PASSAGE_WORDS = 15  # skip chunks too short to ask anything about

DRAFT_PROMPT = """You write study questions from a student's lecture notes.
Below is one passage from the notes. Write {n} different questions that this passage fully answers.
For each question write a short answer of 1 to 3 sentences.

Rules:
- Use ONLY facts stated in the passage. Never add outside knowledge.
- The notes are fragments and bullet points. Rewrite them into clear sentences.
- Each question must make sense on its own, without seeing the passage.

Passage:
{passage}"""

PAIRS_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                "required": ["question", "answer"],
            },
        }
    },
    "required": ["pairs"],
}

# Questions that no lecture note should cover. The right answer is a polite "not in your notes".
OFF_TOPIC = [
    "What is the capital of France?",
    "Who won the 2018 World Cup?",
    "How do I bake sourdough bread?",
    "What is the tallest mountain in the world?",
    "Who wrote Pride and Prejudice?",
    "What is a good workout for building muscle?",
    "How far away is the moon?",
    "What is the best way to learn guitar?",
    "When did the Roman Empire fall?",
    "How many calories are in an avocado?",
]
REFUSALS = [
    "I couldn't find that in your notes.",
    "That doesn't seem to be in your notes, so I can't answer it from them.",
    "I couldn't find anything about that in your notes.",
]

# (what the user says, a short friendly reply). No search is needed for these.
SMALL_TALK = [
    ("hi", "Hi! Ask me anything about your notes."),
    ("hello", "Hello! What would you like to know from your notes?"),
    ("hey", "Hey! Ask me a question about your notes."),
    ("good morning", "Good morning! What would you like to look up in your notes?"),
    ("thanks", "You're welcome! Ask me anything else about your notes."),
    ("thank you", "Happy to help. Ask me another question about your notes any time."),
    ("what can you do?", "I answer questions using your own notes. Ask me about something you wrote down and I'll look it up."),
    ("who are you?", "I'm docagent. I answer questions using only your own notes."),
]


def dataset_path(data_dir, name):
    folder = os.path.join(data_dir, DATASET_DIR)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def draft_pairs(passage, n, model):
    """Asks the local model for n question/answer pairs. Returns [] if it replies with junk."""
    response = requests.post(
        agent.OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": DRAFT_PROMPT.format(n=n, passage=passage)}],
            "format": PAIRS_SCHEMA,  # makes Ollama return JSON that matches this shape
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=300,
    )
    response.raise_for_status()
    try:
        return json.loads(response.json()["message"]["content"])["pairs"]
    except (ValueError, KeyError, TypeError):
        return []


def unsupported_fraction(answer, passage):
    """
    Rough hallucination check: what share of the answer's longer words never
    appear in the passage? Paraphrasing causes some misses, so this only FLAGS
    answers for you to look at harder. It never rejects anything by itself.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if len(w) > 3]
    if not words:
        return 1.0
    known = set(re.findall(r"[a-z0-9]+", passage.lower()))
    return sum(w not in known for w in words) / len(words)


def cmd_draft(args):
    vault = resolve_vault(args.notes)
    data_dir = data_dir_for(vault)
    drafts_path = dataset_path(data_dir, DRAFTS_FILE)

    existing = read_jsonl(drafts_path)
    if existing and not args.fresh:
        reviewed = sum(d["status"] != "pending" for d in existing)
        raise SystemExit(
            f"{drafts_path} already has {len(existing)} drafts ({reviewed} reviewed).\n"
            "Run `review` to continue, or add --fresh to start over (your reviews will be lost)."
        )

    model = args.model or agent.DEFAULT_MODEL
    ensure_ollama(args.model, agent.DEFAULT_MODEL)
    ensure_index(vault, data_dir)
    index, chunks = agent.load_index(data_dir)
    embed_model = load_embedder()

    drafts = []
    seen_questions = set()
    skipped_unretrieved = 0

    for chunk in chunks:
        if len(chunk["text"].split()) < MIN_PASSAGE_WORDS:
            continue
        print(f"Drafting from {chunk['id']} ...")
        for pair in draft_pairs(chunk["text"], args.per_chunk, model):
            question = str(pair.get("question", "")).strip()
            answer = str(pair.get("answer", "")).strip()
            if not question or not answer or question.lower() in seen_questions:
                continue
            seen_questions.add(question.lower())

            # Run the REAL search, as the agent would. If the right chunk does not
            # come back, an example built on it would teach the model to answer
            # from passages that it would never actually be shown.
            result, sources = agent.search_docs(question, index, chunks, embed_model)
            if not any(s["text"] == chunk["text"] for s in sources):
                skipped_unretrieved += 1
                continue

            fraction = unsupported_fraction(answer, chunk["text"])
            drafts.append({
                "id": f"{chunk['id']}#{len(drafts)}",
                "kind": "answerable",
                "group": chunk["id"],
                "source": chunk["source"],
                "question": question,
                "answer": answer,
                "gold": chunk["text"],
                "context": result,
                "flag": f"{int(fraction * 100)}% of the answer's words are not in the passage" if fraction > FLAG_THRESHOLD else "",
                "status": "pending",
            })

    for i, question in enumerate(OFF_TOPIC):
        result, _ = agent.search_docs(question, index, chunks, embed_model)
        drafts.append({
            "id": f"refusal#{i}", "kind": "refusal", "group": f"refusal#{i}", "source": "",
            "question": question, "answer": REFUSALS[i % len(REFUSALS)],
            "gold": "", "context": result, "flag": "", "status": "pending",
        })

    for i, (question, reply) in enumerate(SMALL_TALK):
        drafts.append({
            "id": f"smalltalk#{i}", "kind": "smalltalk", "group": f"smalltalk#{i}", "source": "",
            "question": question, "answer": reply,
            "gold": "", "context": "", "flag": "", "status": "pending",
        })

    write_jsonl(drafts_path, drafts)
    kinds = {k: sum(d["kind"] == k for d in drafts) for k in ("answerable", "refusal", "smalltalk")}
    flagged = sum(bool(d["flag"]) for d in drafts)
    print(f"\nSaved {len(drafts)} drafts to {drafts_path}")
    print(f"  {kinds['answerable']} answerable ({flagged} flagged for a closer look), "
          f"{kinds['refusal']} refusals, {kinds['smalltalk']} small talk")
    if skipped_unretrieved:
        print(f"  Dropped {skipped_unretrieved} drafts whose passage the real search did not find.")
    print("Next: python make_dataset.py review")


def show_draft(d, position, total):
    print("\n" + "=" * 70)
    print(f"[{position}/{total}]  {d['kind']}   {d['source']}")
    if d["kind"] == "answerable":
        print(f"\nPASSAGE FROM YOUR NOTES:\n{d['gold']}")
    elif d["kind"] == "refusal":
        print(f"\nWHAT SEARCH RETURNS: {d['context'][:200].replace(chr(10), ' ')}")
    print(f"\nQUESTION: {d['question']}")
    print(f"ANSWER:   {d['answer']}")
    if d["flag"]:
        print(f"\n  !! CHECK CAREFULLY: {d['flag']}")


def cmd_review(args):
    vault = resolve_vault(args.notes)
    data_dir = data_dir_for(vault)
    drafts_path = dataset_path(data_dir, DRAFTS_FILE)
    drafts = read_jsonl(drafts_path)
    if not drafts:
        raise SystemExit("No drafts yet. Run: python make_dataset.py draft")

    pending = [d for d in drafts if d["status"] == "pending"]
    print(f"{len(pending)} drafts to review. Your progress is saved after every answer.")
    print("Keys: a = accept, e = edit the answer, r = reject, q = quit and continue later")

    for position, d in enumerate(pending, start=1):
        show_draft(d, position, len(pending))
        while True:
            try:
                key = input("\n[a]ccept [e]dit [r]eject [q]uit > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                key = "q"
            if key == "a":
                d["status"] = "accepted"
            elif key == "e":
                new_answer = input("New answer: ").strip()
                if not new_answer:
                    print("Empty answer, keeping the old one.")
                    continue
                d["answer"] = new_answer
                d["status"] = "accepted"
            elif key == "r":
                d["status"] = "rejected"
            elif key == "q":
                write_jsonl(drafts_path, drafts)
                print("\nSaved. Run review again to continue.")
                return
            else:
                continue
            write_jsonl(drafts_path, drafts)
            break

    print("\nAll drafts reviewed. Next: python make_dataset.py build")


def to_messages(d):
    """Turns one accepted draft into the conversation the model should learn from."""
    messages = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": d["question"]},
    ]
    if d["kind"] != "smalltalk":
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "search_docs", "arguments": {"query": d["question"]}}}],
        })
        messages.append({"role": "tool", "tool_name": "search_docs", "content": d["context"]})
    messages.append({"role": "assistant", "content": d["answer"]})
    return messages


def cmd_build(args):
    vault = resolve_vault(args.notes)
    data_dir = data_dir_for(vault)
    accepted = [d for d in read_jsonl(dataset_path(data_dir, DRAFTS_FILE)) if d["status"] == "accepted"]
    if not accepted:
        raise SystemExit("Nothing accepted yet. Run: python make_dataset.py review")

    # Split by GROUP, not by example. All questions drafted from one chunk stay
    # together, so the validation set never contains a question about a passage
    # the model already trained on. That would make the score look better than it is.
    groups = sorted({d["group"] for d in accepted})
    random.Random(42).shuffle(groups)
    n_val = max(1, round(len(groups) * VAL_FRACTION)) if len(groups) >= 2 else 0
    val_groups = set(groups[:n_val])

    train, val = [], []
    for d in accepted:
        row = {"messages": to_messages(d), "kind": d["kind"], "group": d["group"]}
        (val if d["group"] in val_groups else train).append(row)

    write_jsonl(dataset_path(data_dir, TRAIN_FILE), train)
    write_jsonl(dataset_path(data_dir, VAL_FILE), val)
    print(f"Wrote {len(train)} training and {len(val)} validation examples to "
          f"{os.path.join(data_dir, DATASET_DIR)}")
    for kind in ("answerable", "refusal", "smalltalk"):
        print(f"  {kind}: {sum(r['kind'] == kind for r in train)} train, {sum(r['kind'] == kind for r in val)} val")
    if len(train) < 100:
        print(f"\nHeads up: {len(train)} training examples is enough to test the pipeline, but a "
              "useful adapter usually needs 100 or more. Add more notes and draft again.")


def main():
    parser = argparse.ArgumentParser(prog="make_dataset", description="Build LoRA training data from your notes.")
    parser.add_argument("--notes", help="path to your notes folder (defaults to the remembered one)")
    sub = parser.add_subparsers(dest="command", required=True)

    draft = sub.add_parser("draft", help="have the local model draft question/answer pairs")
    draft.add_argument("--model", help="Ollama model that writes the drafts (default llama3.2)")
    draft.add_argument("--per-chunk", type=int, default=3, help="questions to draft per chunk (default 3)")
    draft.add_argument("--fresh", action="store_true", help="replace existing drafts, losing your reviews")
    draft.set_defaults(func=cmd_draft)

    sub.add_parser("review", help="accept, edit, or reject each draft").set_defaults(func=cmd_review)
    sub.add_parser("build", help="write train.jsonl and val.jsonl").set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
