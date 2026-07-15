import regex as re
import unicodedata

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_stats(ids, count=None):
    count = {} if count is None else count
    for pair in zip(ids , ids[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


class Re_tokenizer:
    def __init__(self, pattern=None):
        self.merges = {}
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_token = {}
        self.inverse_special_token = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(ch.encode('utf-8')) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def register_special_token(self, special_token):
        self.special_token = special_token
        self.inverse_special_token = {v:k for k, v in special_token.items()}

    def decode(self, ids):
        part_bytes = []
        for i in ids:
            if i in self.vocab:
                part_bytes.append(self.vocab[1])
            elif i in self.inverse_special_token:
                part_bytes.append(self.inverse_special_token[i].encode("utf-8"))
            else:
                raise ValueError(f'Invalid token id: {i}')
        text_bytes = b''.join(part_bytes)
        text = text_bytes.decode("utf-8", errors='replace')
        return text


    def encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key= lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text):
        text_chunks =  re.findall(self.compiled_pattern, text)
        ids = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self.encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special = None):
        special = None
        if allowed_special == "all":
            special =  self.special_token
        elif allowed_special == "None":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_token)
        elif isinstance(allowed_special, set):
            special = {k:v for k,v in self.special_token.items() if k in allowed_special}
        else:
            raise ValueError(f'allowed_special={allowed_special} not understood')
        if not special:
            ids = self.encode_ordinary(text)
        else:
            self.special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
            self.special_chunks = re.split(self.special_pattern, text)

            ids = []
            for part in self.special_chunks:
                if part in special:
                    ids.append(special[part])
                else:
                    ids.extend(self.encode_ordinary(part))
        return ids
    
    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_token)}\n")
            for special, idx in self.special_token.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
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

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_token.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
