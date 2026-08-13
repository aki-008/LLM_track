"""
Import-safe BPE tokenizer (drop-in replacement for CuPy_dev/tokenizer/bpe.py).

This module fixes the original so encode()/decode() keep working when the
tokenizer is imported from another file or a Jupyter notebook:

1. No module-level side effects on import. The original built huge `text` /
   `sample_text` strings and computed `num_merges` at import time.
2. train() has a consistent signature: train(text, vocab_size, verbose=False).
   The original used train(vocab_size, text, verbose) with a *required*
   `verbose`, so any reuse from another file raised a TypeError.
3. load() recovers a raw pattern string from a model line that was saved as
   the repr of a compiled regex object (`regex.Regex("...", flags=regex.V0)`).
   The shipped Dataloader/bpe.model suffered exactly this corruption, which
   silently made encode() return [] and decode() return "".
4. save() always writes the raw pattern string, never a compiled-regex repr,
   so saved models round-trip cleanly.

Usage:
    from bpe import BPETokenizer, GPT4_SPLIT_PATTERN

    tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)   # or tok = BPETokenizer()
    tok.train("some corpus text", 1024, verbose=False)
    tok.save("bpe")
    tok.load("bpe.model")
    ids   = tok.encode("text", allowed_special="all")
    text  = tok.decode(ids)
"""

import ast
import unicodedata

import regex as re  # third-party drop-in with unicode property classes \p{L}/\p{N}

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276,
    '<|pad|>': 100277,
}


def get_stats(ids, count=None):
    """Count adjacent (ids[i], ids[i+1]) pairs, optionally accumulating into `count`."""
    count = {} if count is None else count
    for pair in zip(ids, ids[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` with the single token id `idx`."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def _extract_pattern(line):
    """Turn a model-file pattern line into a raw regex string.

    Handles both a plain pattern string and the stale repr of a compiled
    regex object (`regex.Regex("...", flags=regex.V0)`) that older save()
    code could write.
    """
    line = line.strip()
    if line.startswith("regex.Regex(") and line.endswith(")"):
        try:
            tree = ast.parse(line, mode="exec")
            expr = tree.body[0].value
            if isinstance(expr, ast.Call):
                for arg in expr.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        return arg.value
        except (SyntaxError, ValueError):
            pass
    return line


def _normalise_pattern(pattern):
    """Return (raw_pattern_string, compiled_pattern) for any accepted pattern value."""
    if isinstance(pattern, bytes):
        pattern = pattern.decode("utf-8")
    if not isinstance(pattern, str):
        pattern = getattr(pattern, "pattern", None) or str(pattern)
    return pattern, re.compile(pattern)


def render_token(token):
    """Return a printable string form of a bytes token, escaping control chars."""
    s = token.decode("utf-8", errors="replace")
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer with regex pre-tokenization and special tokens."""

    def __init__(self, pattern=None):
        self.merges = {}
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.pattern, self.compiled_pattern = _normalise_pattern(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab()
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def register_special_tokens(self, special_tokens):
        if not isinstance(special_tokens, dict):
            raise TypeError("special_tokens must be a dict mapping str -> int")
        self.special_tokens.update(special_tokens)
        self.inverse_special_tokens = {
            idx: token for token, idx in self.special_tokens.items()
        }
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        """Learn merge rules from `text` up to a target `vocab_size` (>= 256)."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not isinstance(vocab_size, int) or vocab_size < 256:
            raise ValueError(
                f"vocab_size must be an integer >= 256, got {vocab_size}"
            )

        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)
        if not text_chunks:
            self.merges = {}
            self.vocab = self._build_vocab()
            return

        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                get_stats(chunk, stats)

            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk, pair, idx) for chunk in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab
        for special, special_idx in self.special_tokens.items():
            self.vocab[special_idx] = special.encode("utf-8")

    def decode(self, ids):
        part_bytes = []
        for i in ids:
            if i in self.vocab:
                part_bytes.append(self.vocab[i])
            elif i in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[i].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {i}")
        return b"".join(part_bytes).decode("utf-8", errors="replace")

    def _encode_chunk(self, bytes_chunk):
        ids = list(bytes_chunk)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text):
        ids = []
        for chunk in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text, allowed_special="none"):
        """Encode `text` to token ids with the given special-token policy."""
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none" or allowed_special is None:
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(
                        f"Special token {token!r} found in text but "
                        f"allowed_special='none_raise'"
                    )
        elif isinstance(allowed_special, set):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            raise ValueError(
                f"allowed_special={allowed_special!r} not understood. "
                f"Expected 'all', 'none', 'none_raise', or a set of strings."
            )

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        parts = re.split(special_pattern, text)

        ids = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def save(self, file_prefix):
        """Write {file_prefix}.model (machine readable) and .vocab (inspection)."""
        pattern = self.pattern
        if not isinstance(pattern, str):
            pattern = getattr(pattern, "pattern", None) or str(pattern)

        merges_file = file_prefix + ".model"
        with open(merges_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for p0, p1 in self.merges:
                f.write(f"{p0} {p1}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Load a tokenizer previously written by save() (also reads old files)."""
        with open(model_file, "r", encoding="utf-8") as f:
            f.readline()  # version header (accepted but not enforced)
            pattern_line = f.readline().strip()
            self.pattern, self.compiled_pattern = _normalise_pattern(
                _extract_pattern(pattern_line)
            )

            num_special = int(f.readline().strip())
            special_tokens = {}
            for _ in range(num_special):
                line = f.readline().strip()
                token, idx_str = line.rsplit(" ", 1)
                special_tokens[token] = int(idx_str)

            merges = {}
            idx = 256
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx0, idx1 = line.split()
                merges[(int(idx0), int(idx1))] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {
            v: k for k, v in special_tokens.items()
        }
        self.vocab = self._build_vocab()
        return self

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab


if __name__ == "__main__":
    tok = BPETokenizer()
    tok.train("hello hello world, hello world, hello!", 300, verbose=False)
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
    print("self-test OK")