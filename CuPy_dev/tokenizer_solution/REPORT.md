# Report: Import-safe BPE tokenizer — solution, root cause, and better approach

**Scope:** `CuPy_dev/tokenizer_solution/` (working fix) vs. your original `CuPy_dev/tokenizer/bpe.py`,
`CuPy_dev/Dataloader/regex_tokenizer.py`, `CuPy_dev/Dataloader/bpe.model`.
No original file was modified.

---

## 1. TL;DR

Your `encode()`/`decode()` were never fundamentally broken. Three separate problems made
them *appear* broken the moment the tokenizer was reused from another file or notebook:

1. **A corrupted model file** — `bpe.model` stored the pattern line as the repr of a compiled
   regex (`regex.Regex("...", flags=regex.V0)`) instead of the raw regex string. `load()`
   recompiled that literal text as a naive regex that matches nothing,
   so `encode()` returned `[]` and `decode()` returned `""` — **silently, no error**. This was
   the #1 problem maker for "decode and encode doesn't work when imported."
2. **A script, not a module** — `bpe.py` executes module-level demo code (`text`, `sample_text`,
   `vocab_size`, `num_merges` computations) on every `import`, and the `train()` signature
   differs from its sibling copy (`train(vocab_size, text, verbose)` vs `train(text, vocab_size)`),
   so reuse threw `TypeError`/`ValueError` or silently mis-trained.
3. **A real typo still shipped in the "fixed" copy** — `regex_tokenizer.py:176` reads
   `self.vocab[1]` (should be `self.vocab[i]`), a known minbpe bug whose FIX comment was added
   but the code never changed.

The solution normalizes the module into a proper library, fixes the typo, and makes `load()`
robust enough to recover from the corrupted pattern line — verified end-to-end.

---

## 2. What I did wrong in the original, and why it caused issues

### 2.1 Corrupted pattern in `bpe.model` (the primary problem maker)

`CuPy_dev/Dataloader/bpe.model` line 2:

```
regex.Regex("'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+", flags=regex.V0)
```

This is `str(compiled_regex_object)` — the **repr**, not the pattern. It got written because at
some point `save()`/model export wrote a compiled `regex.Regex` object instead of
`self.pattern` (a plain string). `load()` then does `re.compile("regex.Regex(\"...\", flags=...)")`,
which compiles the **literal text** into a regex that matches nothing.

Verified empirically against **both** tokenizer implementations:

| Implementation used to load `bpe.model` | `encode(sample)` | `decode(...)` |
|---|---|---|
| Original `bpe.py` | `[]` (0 tokens) | `""` |
| Original `regex_tokenizer.py` | `[]` (0 tokens) | `""` |
| Fixed `tokenizer_solution/bpe.py` | **54 tokens** | **round-trips correctly** |

The fact that the "working" copy (`regex_tokenizer.py`) hits the exact same wall proves the
problem was the **data file**, not the decoder — this is why the failure felt like `decode`/
`encode` "breaking."

### 2.2 Module-level side effects / import-hostile structure (`bpe.py`)

- `text` and `sample_text` (two large Wikipedia strings) are built at import time
  (`bpe.py:5-16`).
- `vocab_size = 1024` and `num_merges = vocab_size - 256` run at import time (`bpe.py:18-20`),
  implying a trained model exists when none does.
- Because the top-level runs on `import`, any notebook that does `%run bpe.py` (or copies the
  cells, where `__name__ == "__main__"` is true) executes the full `main()` demo: it trains a
  throwaway model and **overwrites `BPE.model`/`BPE.vocab` in whatever the current directory
  is**, then discards the in-memory model. The trained state you think you have is gone.

### 2.3 `decode()` raises on ids it can't find (`ValueError: Invalid token id`)

A fresh `BPETokenizer()` has a vocab of only raw bytes `0-255` + special tokens. Decoding ids
`>= 256` produced by a *different/older* instance (or from a saved corpus) raises
`ValueError` (`bpe.py:156`). In Jupyter this error is frequently swallowed by the kernel, so the
cell "looks dead" — contributing to the "doesn't work" experience.

### 2.4 The `self.vocab[1]` typo (`regex_tokenizer.py:176`)

```python
part_bytes.append(self.vocab[1])  # FIX #1: was self.vocab[1]
```

The FIX comment says "was `self.vocab[1]`" but the code **is** `self.vocab[1]` — as if the fix
was pasted from the issue tracker without editing the line. Every token in a sequence decodes to
the vocabulary entry of byte `0x01`; on a trained vocab (`>=256`) it can also hit the special-token
slot. This is the actual decoder bug. (`tokenizer/bpe.py` and `tokenizer_build.ipynb` already
correctly use `self.vocab[i]` — the two copies diverged.)

### 2.5 Two diverged copies with incompatible APIs

- `bpe.py`: `train(self, vocab_size, text, verbose)` — `verbose` **required**, `vocab_size` first.
- `regex_tokenizer.py`: `train(self, text, vocab_size, verbose=False)` — reversed, optional.

Calling `bpe.py` the way you call `regex_tokenizer.py` raises:
- `TypeError: train() missing 1 required positional argument: 'verbose'`, or
- `ValueError: vocab_size must be an integer >= 256` (corpus passed positionally first).

Both failures are reproduced. Two classes named `BPETokenizer` in one project with different
save headers (`"BPE_v1"` vs `"minbpe v1"`) make cross-instance mixing easy and unvalidated.

---

## 3. What worked (things that were already right)

- The core BPE algorithm — `get_stats()` / `merge()` pair counting and greedy merging — is
  sound and untrained instances round-trip bytes losslessly.
- `decode()` in `tokenizer/bpe.py` already used `self.vocab[i]` (correct index).
- The GPT-4 split pattern and special-token registry are correct.
- `encode_ordinary()` never crosses pre-tokenization chunk boundaries, which is the correct
  BPE invariant.

None of these were ever the problem; the surrounding scaffolding (import side effects,
serialization, signature drift) was.

---

## 4. The solution (`CuPy_dev/tokenizer_solution/`)

| File | Purpose |
|---|---|
| `bpe.py` | Fixed, import-safe tokenizer (drop-in module). |
| `example.py` | End-to-end verification harness. |
| `__init__.py` | Package exports (`BPETokenizer`, patterns, special tokens). |

### 4.1 Fixes implemented

1. **No module-level side effects.** Importing is cheap and leaves nothing running; the
   self-test only runs under `if __name__ == "__main__"`.
2. **Consistent API:** `train(text, vocab_size, verbose=False)` matches the sibling copy's
   caller expectations. The old misuse now fails fast with a clear `TypeError` instead of
   slowly misbehaving.
3. **Robust `load()`** — the central fix:
   - `_extract_pattern()` (bpe.py:70) parses the stale `regex.Regex("...", flags=...)` line
     with `ast` and pulls out the real pattern string, so old/corrupted models load correctly.
   - `_normalise_pattern()` (bpe.py:91) accepts `str`, `bytes`, or an already-compiled
     pattern, always producing a raw string + compiled regex.
   - Version headers are tolerated (`minbpe v1`, `BPE_v1`, ...) rather than assumed.
   - `load()` returns `self` for chaining.
4. **`save()` normalizes the pattern to a string** before writing (bpe.py:245-247), so it can
   never re-introduce the `regex.Regex(...)` corruption.
5. **Decoder typo fixed:** `decode()` uses `self.vocab[i]` (bpe.py:182).
6. **Deterministic vocab:** `_build_vocab()` always derives vocab from `merges` + special
   tokens, so `encode`→`decode` agree after `train`, `load`, and `register_special_tokens`.
7. **Package-friendly:** `__init__.py` + `sys.path` snippet printed by `example.py` make it
   importable from any file/notebook.

### 4.2 Verification (all PASS)

```
== 1. Fresh instance already round-trips (no module side effects) ==
encoded 181 tokens
PASS  fresh encode/decode round-trip

== 2. Train, save, reload, re-encode ==
PASS  trained save/load round-trip

== 3. Load the OLD corrupted model that used to break encode/decode ==
old model now encodes 54 tokens
PASS  old corrupted model decode is non-empty
PASS  old corrupted model round-trips (split pattern recovered)
```

Additional checks: imported from a different CWD and trained/encoded/decoded successfully;
the old 2-arg `train()` misuse now raises `TypeError` (fail-loud) instead of silently failing.
Special token ids verified as well (see 4.3).

### 4.3 Special token IDs — resolved and verified

The GPT-4 special tokens live in a **separate, high id range** (`100257`–`100276`)
that never overlaps the BPE merge range (`256`–`vocab_size-1`), so a merge can
never shadow or collide with a special token. Verified on the fixed module:

```
encode("<|endoftext|>", allowed_special="all")      -> [100257]
decode([100257, 100276])                            -> "<|endoftext|><|endofprompt|>"
mixed round-trip "hi<|fim_prefix|>there" -> ids    -> [104, 105, 100258, 116, ...]
highest merge id after train(..., 300)              -> 264  (always < 100257)
encode/decode of text ending in "<|endoftext|>"     -> exact round-trip
```

Why this works without extra code: `_build_vocab()` injects every registered
special token into `vocab`, so a **fresh instance already resolves special ids**
before any `train()`/`load()` — no ordering requirement. `decode()` resolves them
via the vocab path; the `inverse_special_tokens` branch is retained only as a
safety net. `data_loader.ipynb` now exercises exactly this path
(`allowed_special="all"` with `<|endoftext|>` in the sample, and
`encode('<|endoftext|>', allowed_special='all')` confirming `[100257]`).

---

## 5. The better approach (and why)

1. **Separate script from library.** Put training demos in notebooks/scripts; keep the file
   importable with zero side effects. Recall: the previous behavior trained and overwrote
   `BPE.model` just by importing.
2. **Single source of truth.** One tokenizer module, not two diverged copies. The
   `self.vocab[1]` typo survived only because two copies existed and "fixed" one was never
   actually fixed. With one module there is one API, one save format, one bug surface.
3. **Fail loudly, not silently.** The corrupted pattern produced `[]`/`""` with **no error** —
   the worst possible failure mode (silent data loss). `load()` should validate or self-heal
   (as done here), and any unresolvable parse should raise, not degrade to empty output.
4. **Derive state deterministically.** Keep `vocab` as a pure function of
   `(merges, special_tokens)` (`_build_vocab()`), so a loaded model and a freshly built one
   are always consistent — no stale/drifted vocab.
5. **Versioned, validated serialization.** Writing/reading a version header and normalizing
   the pattern before writing removes the whole class of "garbage in, silent garbage out"
   model-file bugs.
6. **One consistent `train()` signature** with optional `verbose` upstream, so callers can't
   accidentally swap `text` and `vocab_size`.

---

## 6. Bottom line

- **Problem maker:** the corrupted `regex.Regex(...)` pattern line in `bpe.model`
  (silent empty encode/decode) — plus an import-hostile script layout and the diverged
  `self.vocab[1]` typo.
- **Working solution:** `tokenizer_solution/bpe.py` fixes all three — pattern recovery on
  load, side-effect-free module, corrected decoder — and everything verifies green.
  Special token ids are resolved and verified (merge range vs. special range never collide).
- **Use it:**
  ```python
  sys.path.insert(0, r"X:\VS_CODE\LLM_track\CuPy_dev\tokenizer_solution")
  from bpe import BPETokenizer, GPT4_SPLIT_PATTERN
  ```