"""
STEP 3 – ENTITY RECOGNITION
I train a spaCy Named Entity Recognition (NER) model to find the important details
in a resume: NAME, EMAIL, PHONE, LOCATION, JOB_TITLE, COMPANY, DATE, DEGREE,
COLLEGE and SKILL.

How the training works:
  1. Load the labelled resumes and clean them with the SAME cleaning as Step 2.
  2. Turn each resume into a spaCy Example (text + correct entities).
  3. Train on the train set. After every epoch I score the model on the dev set
     and keep the best version (early stopping stops when it stops improving).
  4. Test the best model ONCE on the test set it has never seen, and report
     precision, recall and F1 for every entity type.

Run:  python src/step3_ner.py           (train + test)
      python src/step3_ner.py --test    (only test the saved model)
"""
import argparse
import json
import random
import re
import time

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import compile_suffix_regex, fix_random_seed, minibatch

from config import DATA_DIR, LABELS, MODEL_DIR, RESULTS_DIR, SEED
from step2_clean import clean_text, move_spans


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_split(name):
    with open(DATA_DIR / f"{name}.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def to_cleaned(record):
    """Clean the text and move the entity spans to match the cleaned text."""
    cleaned, offsets = clean_text(record["text"], return_offsets=True)
    return cleaned, move_spans(record["entities"], offsets, cleaned)


def make_examples(nlp, records):
    """Build spaCy Examples. Spans that do not line up with token boundaries are counted."""
    examples, skipped = [], 0
    for record in records:
        text, spans = to_cleaned(record)
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in spans:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None or span.text != text[start:end]:
                skipped += 1
                continue
            ents.append(span)
        reference = doc.copy()
        reference.ents = ents
        examples.append(Example(nlp.make_doc(text), reference))
    return examples, skipped


def save_docbin(examples, path):
    db = DocBin(store_user_data=False)
    for ex in examples:
        db.add(ex.reference)
    db.to_disk(path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def create_blank_nlp():
    """A blank English pipeline with one small tokenizer fix.

    spaCy treats a capital letter followed by a full stop ("R.") as one token,
    like an initial in a name. In resumes "R." is usually the skill R at the end of
    a sentence, so I remove those special cases. The fix is saved with the model.
    """
    nlp = spacy.blank("en")
    rules = nlp.tokenizer.rules
    nlp.tokenizer.rules = {k: v for k, v in rules.items() if not re.fullmatch(r"[A-Z]\.", k)}
    suffixes = list(nlp.Defaults.suffixes) + [r"(?<=\b[A-Z])\."]
    nlp.tokenizer.suffix_search = compile_suffix_regex(suffixes).search
    return nlp


def train(max_epochs=30, patience=5, dropout=0.25, batch_size=16):
    fix_random_seed(SEED)
    random.seed(SEED)

    nlp = create_blank_nlp()
    ner = nlp.add_pipe("ner")
    for label in LABELS:
        ner.add_label(label)

    train_examples, skip_train = make_examples(nlp, load_split("train"))
    dev_examples, skip_dev = make_examples(nlp, load_split("dev"))
    print(f"Train: {len(train_examples)} resumes | Dev: {len(dev_examples)} resumes "
          f"| misaligned spans skipped: {skip_train + skip_dev}")
    save_docbin(train_examples, DATA_DIR / "train.spacy")
    save_docbin(dev_examples, DATA_DIR / "dev.spacy")

    optimizer = nlp.initialize(lambda: train_examples)
    history, best_f, bad_epochs = [], -1.0, 0

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        random.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=batch_size):
            nlp.update(batch, drop=dropout, sgd=optimizer, losses=losses)

        scores = nlp.evaluate(dev_examples)
        dev_f = scores["ents_f"]
        history.append({"epoch": epoch, "loss": round(float(losses["ner"]), 2),
                        "dev_precision": round(scores["ents_p"], 4),
                        "dev_recall": round(scores["ents_r"], 4),
                        "dev_f1": round(dev_f, 4)})
        flag = ""
        if dev_f > best_f:
            best_f, bad_epochs, flag = dev_f, 0, "  <- best, saved"
            with nlp.use_params(optimizer.averages):   # averaged weights generalise better
                nlp.to_disk(MODEL_DIR)
        else:
            bad_epochs += 1
        print(f"epoch {epoch:2d} | loss {losses['ner']:9.1f} | dev P {scores['ents_p']:.3f} "
              f"R {scores['ents_r']:.3f} F1 {dev_f:.3f} | {time.time() - start:.0f}s{flag}")
        if bad_epochs >= patience:
            print(f"No improvement for {patience} epochs -> stopping early.")
            break

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_json = json.dumps(history, indent=2)
    (RESULTS_DIR / "training_history.json").write_text(history_json, encoding="utf-8")
    (MODEL_DIR / "training_history.json").write_text(history_json, encoding="utf-8")  # travels with the model
    print(f"Best dev F1: {best_f:.3f}. Model saved to {MODEL_DIR}")


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
def held_out_skill_recall(nlp, records, held_out):
    """Recall on skills the model never saw in training vs. skills it did see."""
    found = {"seen": [0, 0], "unseen": [0, 0]}   # [found, total]
    for record in records:
        text, spans = to_cleaned(record)
        predicted = {(e.start_char, e.end_char) for e in nlp(text).ents if e.label_ == "SKILL"}
        for start, end, label in spans:
            if label != "SKILL":
                continue
            group = "unseen" if text[start:end] in held_out else "seen"
            found[group][1] += 1
            found[group][0] += (start, end) in predicted
    return {g: round(f / t, 4) if t else None for g, (f, t) in found.items()}, \
           {g: t for g, (_, t) in found.items()}


def test():
    nlp = spacy.load(MODEL_DIR)
    records = load_split("test")
    examples, skipped = make_examples(nlp, records)
    scores = nlp.evaluate(examples)

    held_out = set(json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))["held_out_skills"])
    skill_recall, skill_counts = held_out_skill_recall(nlp, records, held_out)

    per_type = {label: {k: round(v, 4) for k, v in s.items()}
                for label, s in scores["ents_per_type"].items()}
    support = {}
    for ex in examples:
        for ent in ex.reference.ents:
            support[ent.label_] = support.get(ent.label_, 0) + 1

    result = {
        "overall": {"precision": round(scores["ents_p"], 4), "recall": round(scores["ents_r"], 4),
                    "f1": round(scores["ents_f"], 4)},
        "per_entity": {k: {**per_type[k], "support": support.get(k, 0)} for k in LABELS if k in per_type},
        "skill_recall_seen_vs_unseen": skill_recall,
        "skill_mentions_seen_vs_unseen": skill_counts,
        "test_resumes": len(examples),
        "misaligned_spans_skipped": skipped,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ner_test_scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nNER TEST RESULTS ({len(examples)} unseen resumes, clean text)")
    print(f"{'Entity':<10} {'Precision':>9} {'Recall':>7} {'F1':>6} {'Count':>6}")
    for label, s in result["per_entity"].items():
        print(f"{label:<10} {s['p']:>9.3f} {s['r']:>7.3f} {s['f']:>6.3f} {s['support']:>6}")
    o = result["overall"]
    print(f"{'OVERALL':<10} {o['precision']:>9.3f} {o['recall']:>7.3f} {o['f1']:>6.3f}")
    print(f"Skill recall – seen in training: {skill_recall['seen']}, "
          f"never seen: {skill_recall['unseen']} ({skill_counts['unseen']} mentions)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="only test the saved model")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if not args.test:
        train(max_epochs=args.epochs)
    test()
