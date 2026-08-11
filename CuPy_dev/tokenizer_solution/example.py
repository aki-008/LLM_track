"""
End-to-end demo of the fixed tokenizer in tokenizer_solution/.

Run from anywhere:
    python CuPy_dev/tokenizer_solution/example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe import BPETokenizer, GPT4_SPLIT_PATTERN

SAMPLE = (
    "The Imperial Russian Navy (Russian: \u0420\u043e\u0441\u0441\u0438\u0439\u0441\u043a\u0438\u0439 "
    "\u0438\u043c\u043f\u0435\u0440\u0430\u0442\u043e\u0440\u0441\u043a\u0438\u0439 \u0444\u043b\u043e\u0442) "
    "operated as the navy of the Russian Tsardom from 1696 to 1917. "
    "Nikolai II r. 1894-1917<|endoftext|>"
)


def check(label, condition):
    if condition:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}")
        raise SystemExit(1)


print("== 1. Fresh instance already round-trips (no module side effects) ==")
tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
ids = tok.encode(SAMPLE, allowed_special="all")
print(f"encoded {len(ids)} tokens")
check("fresh encode/decode round-trip", tok.decode(ids) == SAMPLE)

print("\n== 2. Train, save, reload, re-encode ==")
tok.train(SAMPLE * 5, 512, verbose=False)
demo_dir = Path(__file__).resolve().parent
tok.save(str(demo_dir / "demo"))

reloaded = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
reloaded.load(str(demo_dir / "demo.model"))
ids2 = reloaded.encode(SAMPLE, allowed_special="all")
check("trained save/load round-trip", reloaded.decode(ids2) == SAMPLE)

print("\n== 3. Load the OLD corrupted model that used to break encode/decode ==")
old_model = Path(__file__).resolve().parents[1] / "Dataloader" / "bpe.model"
if old_model.exists():
    old_tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
    old_tok.load(str(old_model))
    ids3 = old_tok.encode(SAMPLE)
    text3 = old_tok.decode(ids3)
    printable = text3[:60].encode("ascii", "backslashreplace").decode("ascii")
    print(f"old model now encodes {len(ids3)} tokens")
    print(f"decoded: {printable}...")
    check("old corrupted model decode is non-empty", len(ids3) > 0 and len(text3) > 0)

    text_only = SAMPLE.split("<|endoftext|>")[0]
    ids4 = old_tok.encode(text_only, allowed_special="none")
    check(
        "old corrupted model round-trips (split pattern recovered)",
        old_tok.decode(ids4) == text_only,
    )
else:
    print("skipped (Dataloader/bpe.model not found)")

print("\nAll checks passed.")
print(f"\nImport this from anywhere with:\n"
      f"  sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})"
      f"\n  from bpe import BPETokenizer, GPT4_SPLIT_PATTERN")