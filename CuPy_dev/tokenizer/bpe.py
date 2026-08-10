# %%
import regex as re
import unicodedata

text = """The Imperial Russian Navy (Russian: Российский императорский флот) operated as the navy of the Russian Tsardom and later the Russian Empire from 1696 to 1917.[c] Formally established in 1696, it lasted until being dissolved in the wake of the February Revolution and the declaration of the Russian Republic in 1917. It developed from a smaller force that had existed prior to Tsar Peter the Great's founding of the modern Russian navy during the Second Azov campaign in 1696[3], and expanded in the second half of the 18th century before reaching its peak strength by the early part of the 19th century, behind only the British and French fleets in terms of size.

The Imperial Navy drew its officers from the aristocracy of the Empire, who belonged to the state Russian Orthodox Church. Young aristocrats began to be trained for leadership at a national naval boarding school, the Naval Cadet Corps. From 1818 on, only officers of the Imperial Russian Navy were appointed to the position of Chief Manager of the Russian-American Company, based in Russian America (present-day Alaska) for colonization and fur-trade development. Although the early Imperial Navy initially employed paid foreign sailors, the government began to recruit native-born sailors as conscripts, drafted (as were men to serve in the army). Service in the navy was lifelong before the 1874 decree on conscription limited the service term to six years at most. Many naval commanders and recruits came from Imperial Russia's non-Russian lands with maritime traditions—Finland and (especially) the Baltic governorates.[citation needed]

The Russian Navy went into a period of decline due to the Empire's slow technical and economic development in the first half of the 19th century. It had a revival in the latter part of the century during the reign of Emperor Nicholas II (r. 1894–1917), but most of its Pacific Fleet (along with the Baltic Fleet sent to the Far East) was destroyed in the disastrous Russo-Japanese War of 1904–1905.[4] Nicholas II, who was a naval enthusiast, had a major role in both the build up of the navy before the war with Japan and the rebuilding of it in the decade after.[5]

The navy had mixed experiences during the First World War, with the Germans generally gaining the upper hand in the Baltic Sea, while the Russians took control of the Black Sea. The Russian Baltic Fleet mostly stayed on the defensive, but the Black Sea Fleet's attacks on Ottoman merchant shipping nearly cut off the coal supply to Constantinople and threatened the Ottoman Empire's ability to stay in the war.[6][7] The Russian Revolution marked the end of the Imperial Navy; the Russian Provisional Government carried out reforms to the navy and its command structure, including the removal of imperial references from its rank insignia. Its officers had mostly aligned with the emperor, and the sailors split to fight on either side during the Russian Civil War of 1917–1922. The Soviet Navy, established as the Red Fleet in 1918 after the Revolution, took over the available surviving ships that did not evacuate from Crimea."""


sample_text = '''The Imperial Russian Navy (Russian: Российский императорский флот) operated as the navy of the Russian Tsardom and later the Russian Empire from 1696 to 1917.[c] Formally established in 1696, it lasted until being dissolved in the wake of the February Revolution and the declaration of the Russian Republic in 1917. It developed from a smaller force that had existed prior to Tsar Peter the Great's founding of the modern Russian navy during the Second Azov campaign in 1696[3], and expanded in the second half of the 18th century before reaching its peak strength by the early part of the 19th century, behind only the British and French fleets in terms of size.

The Imperial Navy drew its officers from the aristocracy of the Empire'''
# %%
vocab_size = 1024
num_merges = vocab_size - 256
num_merges

# %% [markdown]
# ### Text chunking

# %%
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
    count = {} if count is None else count
    for pair in zip(ids, ids[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count


def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def token_handler(t:bytes):
    """
    Convert a byte token to a printable string.

    Decodes the token as UTF-8 and escapes Unicode control characters
    as ``\\uXXXX``. Used when saving BPE merge files.
    """    
    s = t.decode('utf-8', errors='replace')
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0]  != 'C':
            chars.append(ch)
        else:
            chars.append(f'\\u{ord(ch):04x}')

    s = ''.join(chars)
    return s


class BPETokenizer:
    def __init__(self, pattern):
        self.merges = {}
        self.pattern = pattern
        self.compiled_pattern = re.compile(self.pattern)

        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab()
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def register_special_tokens(self, special_tokens):

        if not isinstance(special_tokens, dict):
            raise TypeError("special_tokens must be a dict mapping str -> int")
        self.special_tokens.update(special_tokens)
        self.inverse_special_tokens = {
            idx: token
            for token, idx in self.special_tokens.items()
        }

        self.vocab = self._build_vocab()


    def train(self, vocab_size, text, verbose):

        if not isinstance(text, str):                                    # guard: training text must be a plain string
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not isinstance(vocab_size, int) or vocab_size < 256:           # guard: vocab must be at least the 256 raw byte values
            raise ValueError(f"vocab_size must be an integer >= 256, got {vocab_size}")

        assert vocab_size > 256, "vocab size too small"

        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)
        if not text_chunks:
            self.merges = {}
            self.vocab = self._build_vocab()
            return
        
        ids = [list(ch.encode('utf-8')) for ch in text_chunks]

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                get_stats(chunk, stats)

            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids , pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] 
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]}) had {stats[pair]} occurrences"
                )     
        self.merges = merges
        self.vocab = vocab
        # Preserves the registered special tokens
        for special, special_idx in self.special_tokens.items():
            self.vocab[special_idx] = special.encode('utf-8')

    def decode(self, ids):
        text = []
        for i in ids:
            if i in self.vocab:
                text.append(self.vocab[i])
            elif i in self.inverse_special_tokens:
                text.append(
                    self.inverse_special_tokens[i].encode('utf-8')
                )
            else:
                raise ValueError(f'Invalid token id: {i}')
            
        return b''.join(text).decode('utf-8', errors='replace')
    

    def encode_ordinary(self, text):
        pattern = self.compiled_pattern
        text_chunks = re.findall(pattern, text)

        enc_ids = []

        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))

            while len(chunk_ids) >= 2:
                stats = get_stats(chunk_ids)

                pair = min(stats.keys(), key=lambda p: self.merges.get(p , float('inf')))

                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                chunk_ids = merge(chunk_ids, pair, idx)
            enc_ids.extend(chunk_ids)
        return enc_ids


    def encode(self, text, allowed_special='none'):
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
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")" 
        parts = re.split(special_pattern, text)

        ids = []

        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part)) #extend prevent nested list, dont use append
        return ids

    def save(self, file_prefix):

        merges_file = file_prefix + '.model'
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("BPE_v1 \n")
            f.write(f'{self.pattern}\n')
            f.write(f'{len(self.special_tokens)}\n')

            for special , idx in self.special_tokens.items():
                f.write(f'{special} {idx}\n')

            for p0, p1 in self.merges:
                f.write(f'{p0} {p1}\n')

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair , idx in self.merges.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx , token in self.vocab.items():
                s = token_handler(token)
                if idx in inverted_merges:
                    idx1, idx2  = inverted_merges[idx]
                    s0 = token_handler(self.vocab[idx1])        
                    s1 = token_handler(self.vocab[idx2])
                    f.write(f'[{s0}] [{s1}] -> [{s}] {idx}\n')
                else:
                    f.write(f'[{s}] {idx}\n')

    def load(self, model_file):
        merges = {}
        special_tokens = {}
        idx = 256

        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            self.pattern = f.readline().strip()                         # read back the pre-tokenization regex pattern
            self.compiled_pattern = re.compile(self.pattern) 

            num_special = int(f.readline().strip())

            for _ in range(num_special):
                line = f.readline().strip()
                special, special_idx = line.rsplit(" ", 1)
                special_tokens[special] = int(special_idx)

            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx1, idx2 = line.split()
                merges[(int(idx1), int(idx2))] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {
            v: k for k, v in special_tokens.items()
        }
        self.vocab = self._build_vocab()


    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')

        return vocab


def main():
    model = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
    model.train(1024, text, True)
    model.register_special_tokens(GPT4_SPECIAL_TOKENS)
    model.save('BPE')
    result = model.encode(sample_text, allowed_special='all')
    print(result)
    result = model.decode(result)
    print(result)
    print("Script Complete")
    
    # sample_text = '''The Imperial Russian Navy (Russian: Российский императорский флот) operated as the navy of 
    # the Russian Tsardom and later the Russian Empire from 1696 to 1917.[c] Formally established in 1696, it lasted
    # until being dissolved in the wake of the February Revolution and the declaration of the Russian Republic in 
    # 917. It developed from a smaller force that had existed prior to Tsar Peter the Great's founding of the modern 
    # Russian navy during the Second Azov campaign in 1696[3], and expanded in the second half of the 18th century 
    # before reaching its peak strength by the early part of the 19th century, behind only the British and French 
    # fleets in terms of size.The Imperial Navy drew its officers from the aristocracy of the Empire'''

    # model = BPETokenizerv2(pattern=GPT4_SPLIT_PATTERN)
    # model.load('model\\bpe.model')

    # ids = model.encode(sample_text)
    # print(ids)
    # print(model.decode(ids))

if __name__ == "__main__":
    main()
# %%
