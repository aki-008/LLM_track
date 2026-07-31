import regex as re          # third-party 're' drop-in replacement that supports \p{L}/\p{N} unicode property classes (stdlib 're' does not)
import unicodedata          # used to classify unicode characters (e.g. detect control characters)

# GPT-2 style pre-tokenization regex: splits text into chunks (contractions, words, numbers, punctuation, whitespace)
# before byte-pair merges are ever applied, so merges never cross chunk boundaries.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# GPT-4 style pre-tokenization regex: refines GPT-2's pattern (case-insensitive contractions,
# caps numbers to 1-3 digits, handles newlines specially, uses possessive quantifiers for speed).
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}


def get_stats(ids, count=None):
    """Count consecutive pairs in the list of token ids."""
    count = {} if count is None else count            # start with an empty dict unless an existing count dict was passed in (allows accumulating across chunks)
    for pair in zip(ids, ids[1:]):                     # iterate over every adjacent (ids[i], ids[i+1]) pair
        count[pair] = count.get(pair, 0) + 1           # increment the occurrence count for this pair
    return count                                       # return the pair -> frequency mapping


def merge(ids, pair, idx):
    """Replace all occurrences of `pair` in `ids` with the single token `idx`."""
    newids = []                                         # output list holding the merged sequence
    i = 0                                                # index cursor into the input `ids`
    while i < len(ids):                                  # scan through every position in ids
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            # current position starts the target pair and there's room for a second element that matches
            newids.append(idx)                           # replace the pair with the new merged token id
            i += 2                                        # skip both consumed elements
        else:
            newids.append(ids[i])                        # no match here; keep the original id
            i += 1                                        # advance by one
    return newids                                        # return the new id sequence with merges applied


def replace_control_characters(s: str) -> str:
    """Escape unicode control characters for safe printing."""
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []                                           # accumulator list of characters/escape sequences
    for ch in s:                                         # iterate over each character in the input string
        if unicodedata.category(ch)[0] != "C":           # category codes starting with "C" are control-ish (Cc, Cf, Co, Cs, etc.)
            chars.append(ch)                              # not a control character: keep it as-is
        else:
            chars.append(f"\\u{ord(ch):04x}")             # control character: replace with a printable \uXXXX escape
    return "".join(chars)                                 # reassemble into a single safe-to-print string


def render_token(t: bytes) -> str:
    """Pretty print a token, escaping control characters."""
    s = t.decode('utf-8', errors='replace')               # decode raw bytes to text, substituting invalid sequences rather than raising
    s = replace_control_characters(s)                     # escape any control characters so the token prints safely
    return s                                               # return the human-readable token string


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer with regex-based pre-tokenization.

    Supports training from raw text, encoding/decoding, special tokens,
    and model serialization (save/load).

    Note on performance: The training loop is O(n * num_merges) where n is the
    total number of bytes in the corpus. For large corpora (100MB+), consider
    using optimized implementations like HuggingFace `tokenizers` or
    SentencePiece. This implementation prioritises clarity and correctness.
    """

    def __init__(self, pattern=None):
        self.merges = {}  # (int, int) -> int                      # learned merge rules: byte/token-id pair -> new token id
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern   # use GPT-4 split pattern by default, or a custom one if provided
        self.compiled_pattern = re.compile(self.pattern)            # pre-compile the regex for fast repeated use
        self.special_tokens = {}       # str -> int                 # registry of special tokens (e.g. "<|endoftext|>") to their reserved ids
        self.inverse_special_tokens = {}  # int -> str               # reverse lookup: id -> special token string, used during decode
        self.vocab = self._build_vocab()                            # build the id -> bytes vocabulary from (currently empty) merges/specials
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text, vocab_size, verbose=False):
        """
        Train BPE merges from raw text.

        Args:
            text: The training corpus as a single string.
            vocab_size: Target vocabulary size (must be >= 256).
            verbose: If True, print progress for every merge.
        """
        if not isinstance(text, str):                                    # guard: training text must be a plain string
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not isinstance(vocab_size, int) or vocab_size < 256:           # guard: vocab must be at least the 256 raw byte values
            raise ValueError(f"vocab_size must be an integer >= 256, got {vocab_size}")

        num_merges = vocab_size - 256                                    # number of new merged tokens to learn beyond the base 256 bytes

        # Pre-tokenize with the regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)            # split the corpus into chunks (words/punctuation/etc.) using the regex
        if not text_chunks:
            # Nothing to train on — leave merges empty
            self.merges = {}                                             # no chunks found (e.g. empty text): reset merges
            self.vocab = self._build_vocab()                             # rebuild vocab to reflect the empty merge set
            return                                                       # nothing further to do

        # Encode each chunk into its raw UTF-8 byte ids
        ids = [list(ch.encode('utf-8')) for ch in text_chunks]           # convert each text chunk into a list of raw byte values (0-255)

        merges = {}                                                      # local dict of learned (pair -> new id) merges built up during training
        vocab = {idx: bytes([idx]) for idx in range(256)}                # local vocab starting with the 256 single-byte tokens

        for i in range(num_merges):                                     # perform num_merges merge iterations
            # Gather pair statistics across all chunks
            stats = {}                                                   # frequency count of adjacent pairs, accumulated over all chunks
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)                              # accumulate pair counts from this chunk into the shared stats dict

            if not stats:
                # No more pairs to merge (all chunks are length-1)
                break                                                    # stop early if there are no more pairs left to merge

            pair = max(stats, key=stats.get)                             # pick the most frequent pair to merge next
            idx = 256 + i                                                # assign the next available token id for this merge
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]     # apply this merge across every chunk's id sequence
            merges[pair] = idx                                           # record the merge rule: pair -> new id
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]                 # the new token's bytes are the concatenation of the merged pair's bytes

            if verbose:
                print(                                                  # optionally log progress for this merge step
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]}) had {stats[pair]} occurrences"
                )

        self.merges = merges                                            # store the learned merges on the instance
        self.vocab = vocab                                              # store the resulting vocab on the instance

        # Re-incorporate any previously registered special tokens
        for special, special_idx in self.special_tokens.items():
            self.vocab[special_idx] = special.encode("utf-8")           # make sure special tokens still map to their bytes after retraining vocab

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    def register_special_tokens(self, special_tokens):
        """
        Register a dict of special tokens: {"<|endoftext|>": 100257, ...}.

        These are *added* to any existing special tokens.  To replace all
        special tokens, pass a complete dict (old entries are kept unless
        their keys collide).
        """
        if not isinstance(special_tokens, dict):                        # guard: input must be a dict of str -> int
            raise TypeError("special_tokens must be a dict mapping str -> int")
        self.special_tokens.update(special_tokens)                      # merge the new special tokens into the existing registry
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}  # rebuild the id -> token reverse lookup
        # Rebuild vocab to include the new special tokens
        self.vocab = self._build_vocab()                                # regenerate the full vocab so it includes the new special tokens

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids):
        """Decode a list of token ids back into a string."""
        part_bytes = []                                                 # list of byte fragments to be concatenated
        for i in ids:                                                   # process each token id in order
            if i in self.vocab:
                part_bytes.append(self.vocab[i])  # FIX #1: was self.vocab[1]   # normal token: look up its raw bytes
            elif i in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[i].encode("utf-8"))  # special token: encode its string form back to bytes
            else:
                raise ValueError(f"Invalid token id: {i}")               # unknown id: cannot decode, fail loudly
        text_bytes = b"".join(part_bytes)                                # concatenate all byte fragments into one byte string
        text = text_bytes.decode("utf-8", errors="replace")              # decode to text, substituting any invalid UTF-8 sequences
        return text                                                      # return the fully decoded string

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_chunk(self, text_bytes):
        """Encode a single pre-tokenized chunk (bytes) into merged token ids."""
        ids = list(text_bytes)                                          # start from the raw byte values (0-255) of this chunk
        while len(ids) >= 2:                                            # need at least 2 ids to have a mergeable pair
            stats = get_stats(ids)                                      # get all adjacent pair counts (counts themselves are unused here)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))  # pick the pair with the lowest merge index (i.e. earliest-learned merge)
            if pair not in self.merges:
                break                                                    # no candidate pair has a known merge rule: nothing left to merge
            idx = self.merges[pair]                                      # look up the token id this pair merges into
            ids = merge(ids, pair, idx)                                  # apply that merge across the whole chunk
        return ids                                                       # return the final list of token ids for this chunk

    def encode_ordinary(self, text):
        """Encode text into token ids, ignoring special tokens."""
        text_chunks = re.findall(self.compiled_pattern, text)            # split text into pre-tokenization chunks using the regex
        ids = []                                                         # accumulator for the final token id sequence
        for chunk in text_chunks:                                       # process each chunk independently (merges never cross chunks)
            chunk_bytes = chunk.encode("utf-8")                          # convert the chunk text to raw UTF-8 bytes
            chunk_ids = self._encode_chunk(chunk_bytes)                  # BPE-encode this chunk's bytes into token ids
            ids.extend(chunk_ids)                                        # append this chunk's ids to the overall sequence
        return ids                                                       # return the complete list of token ids

    def encode(self, text, allowed_special="none"):
        """
        Encode text into token ids with special-token handling.

        Args:
            text: The string to encode.
            allowed_special: Controls how special tokens in the text are handled.
                - "all"        : recognise every registered special token.
                - "none"       : ignore special tokens (treat as ordinary text).
                - "none_raise" : raise if any special token is found in text.
                - a set(...)   : recognise only the listed special tokens.
        """
        # FIX #2 & #3: was comparing to the string "None"; now uses sensible
        # string sentinel "none" and also accepts None for backwards compat.
        if allowed_special == "all":
            special = self.special_tokens                               # recognise every registered special token
        elif allowed_special == "none" or allowed_special is None:
            special = {}                                                 # treat all special tokens as ordinary text (none recognised)
        elif allowed_special == "none_raise":
            special = {}                                                 # none recognised, but validate none are present in the text
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(                                   # a special token substring was found where none was allowed: error out
                        f"Special token {token!r} found in text but "
                        f"allowed_special='none_raise'"
                    )
        elif isinstance(allowed_special, set):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special  # only allow the explicitly listed subset of special tokens
            }
        else:
            raise ValueError(                                           # unrecognised value for allowed_special: fail with a helpful message
                f"allowed_special={allowed_special!r} not understood. "
                f"Expected 'all', 'none', 'none_raise', or a set of strings."
            )

        if not special:
            return self.encode_ordinary(text)                           # fast path: no special tokens to handle, just do ordinary BPE encoding

        # FIX #6: use local variables, not instance attributes
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"  # build a regex alternation of escaped special-token strings, capturing them
        special_chunks = re.split(special_pattern, text)                # split text into a list alternating normal text and special-token matches

        ids = []                                                        # accumulator for the final token id sequence
        for part in special_chunks:                                    # process each piece of the split text
            if part in special:
                ids.append(special[part])                              # this piece IS a special token: append its reserved id directly
            else:
                ids.extend(self.encode_ordinary(part))                 # ordinary text piece: run it through normal BPE encoding
        return ids                                                      # return the complete list of token ids

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, file_prefix):
        """
        Save the tokenizer to disk.

        Writes two files:
            {file_prefix}.model  — pattern, special tokens, and merges
            {file_prefix}.vocab  — human-readable vocabulary (for inspection)
        """
        # --- .model file (machine-readable) ---
        model_file = file_prefix + ".model"                             # construct the path for the machine-readable model file
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")                                      # write a version header so load() can validate format compatibility
            f.write(f"{self.pattern}\n")                                 # write the regex pattern used for pre-tokenization
            f.write(f"{len(self.special_tokens)}\n")                     # write how many special tokens follow
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")                           # write each special token as "<token> <id>"
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")                              # write each merge rule as "<id1> <id2>" (order defines the new id via loading)

        # --- .vocab file (human-readable) ---
        vocab_file = file_prefix + ".vocab"                             # construct the path for the human-readable vocab file
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}  # reverse mapping: new token id -> the pair it was merged from
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():                      # iterate over every token in the vocabulary
                s = render_token(token)                                  # get a safe, printable string representation of this token
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]                    # this token came from merging two earlier tokens
                    s0 = render_token(self.vocab[idx0])                  # printable form of the first component token
                    s1 = render_token(self.vocab[idx1])                  # printable form of the second component token
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")            # show the merge relationship for readability
                else:
                    f.write(f"[{s}] {idx}\n")                            # base byte token or special token: just show it directly

    def load(self, model_file):
        """
        Load a tokenizer from a .model file previously written by save().
        """
        assert model_file.endswith(".model"), f"Expected .model file, got {model_file}"  # guard: only accept the expected file extension

        merges = {}                                                     # local dict to rebuild the merge rules into
        special_tokens = {}                                             # local dict to rebuild the special token registry into
        idx = 256                                                       # merged token ids start right after the 256 raw byte values

        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()                              # read the version header line
            assert version == "minbpe v1", f"Unknown model version: {version}"  # verify the file format version matches what we support

            self.pattern = f.readline().strip()                         # read back the pre-tokenization regex pattern
            self.compiled_pattern = re.compile(self.pattern)            # recompile the regex for use

            num_special = int(f.readline().strip())                     # read how many special token lines follow
            for _ in range(num_special):
                line = f.readline().strip()                             # read one special-token line
                # Special token format: "<|token|> 100257"
                # The token may contain spaces, so split from the right
                special, special_idx = line.rsplit(" ", 1)               # split off the id from the right in case the token text contains spaces
                special_tokens[special] = int(special_idx)               # register this special token with its id

            for line in f:                                              # remaining lines are all merge rules
                line = line.strip()                                     # strip whitespace/newline
                if not line:
                    continue                                             # skip any blank lines
                idx1, idx2 = line.split()                                # parse the two component ids of this merge
                merges[(int(idx1), int(idx2))] = idx                     # this pair merges into the next sequential id
                idx += 1                                                 # advance to the next available merged-token id

        self.merges = merges                                            # store the reconstructed merges
        self.special_tokens = special_tokens                            # store the reconstructed special tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}  # rebuild the id -> special token reverse lookup
        self.vocab = self._build_vocab()                                # rebuild the full vocab from merges + special tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_vocab(self):
        """Deterministically derive the vocab from merges + special tokens."""
        vocab = {idx: bytes([idx]) for idx in range(256)}               # start with the 256 single-byte base tokens
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]                          # each merged token's bytes = concatenation of its two component tokens' bytes
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")                        # special tokens map directly to their UTF-8 encoded string
        return vocab                                                    # return the complete id -> bytes vocabulary