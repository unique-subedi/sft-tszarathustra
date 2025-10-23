#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds an instruction-style SFT dataset from Project Gutenberg #1998
(Thomas Common, 1909 translation of "Thus Spake Zarathustra" — Public Domain).

Now supports target counts:
  --train_n 1000 --eval_n 100

Outputs:
  data/pg1998_raw.txt
  data/zarathustra_clean.txt
  data/sft_train.jsonl
  data/sft_eval.jsonl
"""

import argparse, json, os, re, urllib.request, random
from pathlib import Path
from typing import List, Tuple

GUT_URL = "https://www.gutenberg.org/cache/epub/1998/pg1998.txt"

SYSTEM_STYLE = (
    "You are a helpful literary assistant channeling the biblical-prophetic cadence of "
    "Nietzsche's *Thus Spake Zarathustra* (Thomas Common translation). Use direct address "
    "('my friend', 'ye'), parallelism, aphoristic turns, and cosmic imagery (sun, mountain, flame, abyss). "
    "Avoid long verbatim quotations. If you quote briefly, keep it under 20 words."
)

THEMES = [
    "self-overcoming", "eternal recurrence", "the herd vs. the creator",
    "the will to power", "solitude", "joy and laughter", "the abyss", "the sun",
    "friendship and enmity", "the Übermensch", "pity and pride", "creation and destruction",
    "freedom and burden", "dance and gravity"
]

STYLE_TOPICS = [
    "a sleepless city at dawn", "a laboratory in winter", "a crowded marketplace at noon",
    "a lone runner at dusk", "a quiet library in summer rain", "a storm over the sea",
    "a programmer facing an error", "a student before an exam", "a parent’s farewell at a train station",
    "a lighthouse in fog", "a mountain pass at twilight", "a ship leaving harbor"
]

QA_FORMS = [
    ("What does Zarathustra teach about '{theme}' in '{title}'?",
     "Answer concisely in his cadence; avoid long quotes, use paraphrase."),
    ("How does '{title}' treat the contrast between solitude and community?",
     "Write 5–7 sentences, prophetic tone, include one brief cited fragment (<20 words)."),
    ("Explain the parable-like imagery in '{title}'.",
     "Name two images and what each signifies. Keep it under 150 words."),
]

HEADER_START_RE = re.compile(r"^\*{3}\s*START OF.*PROJECT GUTENBERG", re.I | re.M)
HEADER_END_RE   = re.compile(r"^\*{3}\s*END OF.*PROJECT GUTENBERG",   re.I | re.M)

# -------------------- IO helpers --------------------

def fetch_text(path: Path) -> str:
    if not path.exists():
        print(f"[download] {GUT_URL}")
        data = urllib.request.urlopen(GUT_URL).read().decode("utf-8", errors="ignore")
        path.write_text(data, encoding="utf-8")
    return path.read_text(encoding="utf-8", errors="ignore")

def strip_gutenberg_boilerplate(raw: str) -> str:
    m_start = HEADER_START_RE.search(raw)
    m_end   = HEADER_END_RE.search(raw)
    if m_start and m_end and m_end.start() > m_start.end():
        core = raw[m_start.end():m_end.start()]
    else:
        core = raw
    core = core.replace("\r\n", "\n").replace("\r", "\n")
    core = re.sub(r"\n{3,}", "\n\n", core)
    return core.strip()

# -------------------- segmentation --------------------

def looks_like_title(line: str) -> bool:
    s = line.strip()
    if len(s) < 4 or len(s) > 80: return False
    if s.lower().startswith(("produced by", "transcribed", "translator", "editor",
                             "license", "preface", "project gutenberg")):
        return False
    if s.endswith("."): return False
    if "  " in s: return False
    upper = sum(ch.isupper() for ch in s)
    lower = sum(ch.islower() for ch in s)
    return (upper >= max(5, lower * 2))

def split_into_sections(clean: str) -> List[Tuple[str, str]]:
    lines = clean.split("\n")
    sections: List[Tuple[str, str]] = []
    cur_title, cur_buf = None, []
    started = False
    for i, ln in enumerate(lines):
        if not started and "THUS SPAKE ZARATHUSTRA" in ln.upper():
            started = True
        if not started: continue
        is_title = looks_like_title(ln) and (i == 0 or lines[i-1].strip() == "")
        if is_title:
            if cur_title and cur_buf:
                content = "\n".join(cur_buf).strip()
                if len(content.split()) >= 80:
                    sections.append((cur_title, content))
            cur_title, cur_buf = ln.strip(), []
        else:
            if cur_title:
                cur_buf.append(ln)
    if cur_title and cur_buf:
        content = "\n".join(cur_buf).strip()
        if len(content.split()) >= 80:
            sections.append((cur_title, content))
    # filter TOC and tiny
    out = []
    for t, c in sections:
        if t.upper().startswith(("CONTENTS", "LIST OF")): continue
        if len(c.split()) < 80: continue
        out.append((t, c))
    return out

# -------------------- generation --------------------

def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def trim(text: str, max_chars: int) -> str:
    return squash(text)[:max_chars]

def make_summary(title: str, content: str, max_chars: int):
    instr = f"Summarize '{title.title()}' in 5–7 bullet points in Zarathustra's prophetic cadence. Avoid long quotes."
    out = "• " + trim(content, max_chars)
    return dict(system=SYSTEM_STYLE, instruction=instr, input="", output=out)

def make_theme(title: str, content: str, theme: str, max_chars: int):
    instr = (f"In the style of Zarathustra, explain how the theme of '{theme}' appears in '{title.title()}'. "
             f"Use brief quoted fragments only if necessary; prefer paraphrase.")
    out = trim(content, max_chars)
    return dict(system=SYSTEM_STYLE, instruction=instr, input="", output=out)

def make_style_transfer(topic: str, seed: str, max_chars: int):
    instr = (f"Write ~120 words about '{topic}' in the cadence of Zarathustra "
             f"(direct address, parallelism, aphoristic turns). Do not reference the book's plot.")
    out = trim(seed, max_chars)
    return dict(system=SYSTEM_STYLE, instruction=instr, input="", output=out)

def make_aphorisms(seed: str, max_chars: int):
    instr = ("Compose 3 aphorisms (1–2 sentences each) in Zarathustra's tone about solitude, creation, and laughter. "
             "Number them 1–3. Avoid clichés and direct quotations.")
    out = trim(seed, max_chars)
    return dict(system=SYSTEM_STYLE, instruction=instr, input="", output=out)

def make_qa(title: str, content: str, theme: str, form_idx: int, max_chars: int):
    q_tmpl, guide = QA_FORMS[form_idx % len(QA_FORMS)]
    question = q_tmpl.format(theme=theme, title=title.title())
    instr = f"{question} {guide}"
    out = trim(content, max_chars)
    return dict(system=SYSTEM_STYLE, instruction=instr, input="", output=out)

def build_pool(sections: List[Tuple[str,str]], rng: random.Random, max_chars: int, min_pool: int) -> List[dict]:
    """
    Build a large diverse pool of examples to sample from.
    We cycle sections and vary templates until we exceed `min_pool`.
    """
    pool: List[dict] = []
    i = 0
    while len(pool) < min_pool:
        title, content = sections[i % len(sections)]
        # Per iteration, add a small mix
        pool.append(make_summary(title, content, max_chars))
        pool.append(make_theme(title, content, rng.choice(THEMES), max_chars))
        pool.append(make_style_transfer(rng.choice(STYLE_TOPICS), content, max_chars))
        pool.append(make_aphorisms(content, max_chars))
        pool.append(make_qa(title, content, rng.choice(THEMES), i, max_chars))
        i += 1
    # shuffle
    rng.shuffle(pool)
    return pool

def sample_split(pool: List[dict], train_n: int, eval_n: int, rng: random.Random) -> Tuple[List[dict], List[dict]]:
    # simple take: first eval_n for eval, next train_n for train (after a fresh shuffle)
    rng.shuffle(pool)
    need = train_n + eval_n
    if len(pool) < need:
        raise RuntimeError(f"Pool size {len(pool)} < requested total {need}. Increase min_pool.")
    eval_set = pool[:eval_n]
    train_set = pool[eval_n:eval_n+train_n]
    return train_set, eval_set

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=GUT_URL)
    ap.add_argument("--raw_out", default="data/pg1998_raw.txt")
    ap.add_argument("--clean_out", default="data/zarathustra_clean.txt")
    ap.add_argument("--train_out", default="data/sft_train.jsonl")
    ap.add_argument("--eval_out", default="data/sft_eval.jsonl")
    ap.add_argument("--train_n", type=int, default=1000, help="target number of training examples")
    ap.add_argument("--eval_n", type=int, default=100, help="target number of eval examples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_output_chars", type=int, default=1100, help="cap for supervision targets")
    ap.add_argument("--pool_multiplier", type=float, default=1.6,
                    help="build pool of ~multiplier*(train_n+eval_n) before sampling")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs("data", exist_ok=True)

    # 1) fetch & clean
    raw = fetch_text(Path(args.raw_out))
    clean = strip_gutenberg_boilerplate(raw)
    Path(args.clean_out).write_text(clean, encoding="utf-8")

    # 2) segment
    sections = split_into_sections(clean)
    if not sections:
        raise RuntimeError("Could not segment chapters—check heuristics.")
    print(f"[info] sections detected: {len(sections)}")

    # 3) build large pool
    total_target = args.train_n + args.eval_n
    min_pool = max(total_target, int(total_target * args.pool_multiplier))
    pool = build_pool(sections, rng, args.max_output_chars, min_pool=min_pool)
    print(f"[info] built pool: {len(pool)} examples (target total {total_target})")

    # 4) sample exact sizes
    train, eval_ = sample_split(pool, args.train_n, args.eval_n, rng)

    # 5) write
    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for r in eval_: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] Train: {len(train)} | Eval: {len(eval_)} → {args.train_out}, {args.eval_out}")
    print("Next:  BOOKSFT_BASE=meta-llama/Meta-Llama-3.1-8B-Instruct python scripts/01_train_sft.py --config configs/sft.yaml")

if __name__ == "__main__":
    main()

