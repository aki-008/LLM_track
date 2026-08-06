# %%
text = """The Imperial Russian Navy (Russian: Российский императорский флот) operated as the navy of the Russian Tsardom and later the Russian Empire from 1696 to 1917.[c] Formally established in 1696, it lasted until being dissolved in the wake of the February Revolution and the declaration of the Russian Republic in 1917. It developed from a smaller force that had existed prior to Tsar Peter the Great's founding of the modern Russian navy during the Second Azov campaign in 1696[3], and expanded in the second half of the 18th century before reaching its peak strength by the early part of the 19th century, behind only the British and French fleets in terms of size.

The Imperial Navy drew its officers from the aristocracy of the Empire, who belonged to the state Russian Orthodox Church. Young aristocrats began to be trained for leadership at a national naval boarding school, the Naval Cadet Corps. From 1818 on, only officers of the Imperial Russian Navy were appointed to the position of Chief Manager of the Russian-American Company, based in Russian America (present-day Alaska) for colonization and fur-trade development. Although the early Imperial Navy initially employed paid foreign sailors, the government began to recruit native-born sailors as conscripts, drafted (as were men to serve in the army). Service in the navy was lifelong before the 1874 decree on conscription limited the service term to six years at most. Many naval commanders and recruits came from Imperial Russia's non-Russian lands with maritime traditions—Finland and (especially) the Baltic governorates.[citation needed]

The Russian Navy went into a period of decline due to the Empire's slow technical and economic development in the first half of the 19th century. It had a revival in the latter part of the century during the reign of Emperor Nicholas II (r. 1894–1917), but most of its Pacific Fleet (along with the Baltic Fleet sent to the Far East) was destroyed in the disastrous Russo-Japanese War of 1904–1905.[4] Nicholas II, who was a naval enthusiast, had a major role in both the build up of the navy before the war with Japan and the rebuilding of it in the decade after.[5]

The navy had mixed experiences during the First World War, with the Germans generally gaining the upper hand in the Baltic Sea, while the Russians took control of the Black Sea. The Russian Baltic Fleet mostly stayed on the defensive, but the Black Sea Fleet's attacks on Ottoman merchant shipping nearly cut off the coal supply to Constantinople and threatened the Ottoman Empire's ability to stay in the war.[6][7] The Russian Revolution marked the end of the Imperial Navy; the Russian Provisional Government carried out reforms to the navy and its command structure, including the removal of imperial references from its rank insignia. Its officers had mostly aligned with the emperor, and the sailors split to fight on either side during the Russian Civil War of 1917–1922. The Soviet Navy, established as the Red Fleet in 1918 after the Revolution, took over the available surviving ships that did not evacuate from Crimea."""

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

# %%
import regex as re

pattern = re.compile(GPT4_SPLIT_PATTERN)
text_chunks = re.findall(pattern, text)
text_chunks

# %% [markdown]
# ### Raw bytes Conversion

# %%
ids = [list(ch.encode('utf-8')) for ch in text_chunks]
print(ids)

# %% [markdown]
# ### Vocab init

# %%
merges = {}
vocab = {i: bytes([i]) for i in range(256)}

print(vocab)

# %% [markdown]
# ### Get_stats helper func

# %%
def get_stats(ids, count=None):
    count = {} if count is None else count
    for pair in zip(ids, ids[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count


# %%
# sample = [32, 82, 117, 115, 115, 115, 105, 97, 110]
# sample

# %%
# stats = {}
# result = get_stats(sample, stats)

# %% [markdown]
# ### Merge helper functions

# %%
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


# %%
# sample_ids = [32, 82, 117, 115, 115, 115, 105, 97, 110]
# pair = max(stats, key=stats.get)
# idx = 256

# result = merge(sample_ids, pair, idx)
# result


# %% [markdown]
# ### Training loop

# %%
for i in range(num_merges):
    stats = {}
    for j in ids:
        get_stats(j, stats)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

# %%
import pprint

pprint.pprint(vocab)

# %%
pprint.pprint(merges)

# %% [markdown]
# ### Decoding

# %%
sample_ids = [84, 257, 32, 110, 117, 109, 98, 101, 114, 274, 102, 32, 112, 97, 114, 116, 105, 99, 105, 112, 259, 116, 115, 270, 97, 115, 268, 99, 114, 263, 115, 261, 265, 114, 266, 256, 264, 256, 272, 115, 256, 111, 256, 119, 101, 108, 118, 101, 44, 270, 104, 105, 99, 104, 268, 99, 108, 117, 100, 261, 32, 97, 32, 104, 111, 271, 256, 272, 44, 256, 111, 112, 265, 105, 118, 101, 256, 272, 115, 265, 114, 266, 260, 32, 112, 114, 101, 118, 105, 111, 117, 115, 32, 261, 105, 116, 105, 111, 110, 44, 260, 256, 119, 111, 32, 104, 105, 103, 257, 271, 45, 114, 259, 107, 261, 256, 272, 115, 268, 260, 32, 73, 67, 67, 269, 266, 264, 39, 115, 32, 84, 273, 73, 32, 84, 272, 32, 82, 259, 107, 258, 103, 115, 32, 110, 111, 116, 32, 267, 114, 263, 100, 121, 32, 113, 117, 267, 105, 102, 105, 261, 44, 32, 262, 265, 111, 117, 114, 274, 116, 257, 114, 256, 272, 115, 32, 100, 101, 116, 101, 114, 109, 258, 261, 256, 104, 114, 111, 117, 103, 104, 32, 97, 32, 115, 101, 114, 105, 101, 115, 274, 102, 32, 113, 117, 267, 105, 102, 105, 101, 114, 115, 46, 32, 78, 101, 116, 257, 114, 275, 115, 32, 113, 117, 267, 105, 102, 105, 261, 265, 111, 114, 260, 269, 266, 264, 39, 115, 32, 84, 273, 269, 111, 114, 108, 100, 32, 67, 117, 112, 265, 111, 114, 260, 265, 105, 114, 271, 256, 105, 109, 101, 46, 60, 124, 264, 100, 111, 102, 116, 101, 120, 116, 124, 62]

# %%
enc = []
for i in sample_ids:
    enc.append(vocab[i].decode("utf-8"))

txt = ''.join(enc)
txt

# %% [markdown]
# ### Encoding

# %%
sample_text ='''The Russian Navy went into a period of decline due to the Empire's slow technical and economic development in the first half of the 19th century. It had a revival in the latter part of the century during the reign of Emperor Nicholas II (r. 1894–1917), but most of its Pacific Fleet (along with the Baltic Fleet sent to the Far East) was destroyed in the disastrous Russo-Japanese War of 1904–1905.[4] Nicholas II, who was a naval enthusiast, had a major role in both the build up of the navy before the war with Japan and the rebuilding of it in the decade after.'''


# %%
txt_chunks = re.findall(pattern, sample_text)
pprint.pprint(txt_chunks)

# %%
enc_ids = []

for chunk in text_chunks:
    chunk_ids = list(chunk.encode('utf-8'))

    while len(chunk_ids) >= 2:
        stats = get_stats(chunk_ids)

        pair = min(stats.keys(), key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
                break
        idx = merges[pair]
        chunk_ids = merge(chunk_ids, pair, idx)
    enc_ids.extend(chunk_ids)
print(enc_ids)


# %% [markdown]
# ### saving and loading

# %%
import unicodedata

def token_handler(t:bytes):
    s = t.decode('utf-8', errors='replace')
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0]  != 'C':
            chars.append(ch)
        else:
            chars.append(f'\\u{ord(ch):04x}')

    s = ''.join(chars)
    return s

# %%
def save(file_prefix):

    # machine readable merges
    model_file = file_prefix + ".model"
    with open(model_file, "w" ,encoding="utf-8") as f:
        f.write('Bpe v1\n')
        f.write(f'{model.pattern}\n')
        for idx0, idx1 in merges:
            f.write(f"{idx0} {idx1}\n")

    # Human-readable vocab
    vocab_file = file_prefix + ".vocab"

    inverted_merges = {idx: pair for pair, idx in model.merges.items()}
    with open (vocab_file, 'w', encoding='utf-8') as f:
        for idx, token in model.vocab.items():
            s = token_handler(token)
            if idx in inverted_merges:
                idx1, idx2 = inverted_merges[idx]
                s0 = token_handler(model.vocab[idx1])
                s1 = token_handler(model.vocab[idx2])
                f.write(f'[{s0}] [{s1}] -> [{s}] {idx}\n') #idx missing
            else:
                f.write(f'[{s}] {idx}\n')

# save('bpe')

# %% [markdown]
# ### BPE Tokenizer

# %%
class BPETokenizer:
    def __init__(self, pattern):
        self.pattern = re.compile(GPT4_SPLIT_PATTERN)
        self.merges = {}


    def train(self, vocab_size, text, verbose):
        assert vocab_size > 256, "vocab size too small"

        num_merges = vocab_size - 256

        text_chunks = re.findall(self.pattern, text)
        ids = [list(ch.encode('utf-8')) for ch in text_chunks]
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                get_stats(chunk, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids , pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] 
            if verbose:
                print(
                    "merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]}) had {stats[pair]} occurrences"
                )     
        self.merges = merges
        self.vocab = vocab

    # def decode(self, ids):
    #     text = []
    #     for i in ids:
    #         text.append(self.vocab[i].decode('utf-8'))
    #     txt = ''.join(text)
    #     return txt

    def decode(self, ids):
        text = []
        for i in ids:
            text.append(self.vocab[i])
        txt = b''.join(text)
        return txt

    def encode(self, text):
        pattern = self.pattern
        text_chunks = re.findall(pattern, text)

        enc_ids = []

        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))

            while len(chunk_ids) >= 2:
                stats = get_stats(chunk_ids)

                pair = min(stats.keys(), key=lambda p: self.merges.get(p , float('inf')))

                if pair not in merges:
                    break
                idx = merges[pair]
                chunk_ids = merge(chunk_ids, pair, idx)
            enc_ids.extend(chunk_ids)
        return enc_ids

    def save(self, file_prefix):

        merges_file = file_prefix + '.model'
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("BPE_v1 \n")
            f.write(f'{self.pattern}\n')
            for idx0, idx1 in self.merges:
                f.write(f'{idx0} {idx1}\n')

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



# %%
model = BPETokenizer(pattern)
model.train(1024, text, True)

# %%
result = model.decode(sample_ids)
result

# %%
result = model.encode(sample_text)
result

# %%
model.vocab

# %%
model.merges

# %%
model.save('bpe')

# %%



print('Script Complete')