import sys
sys.path.insert(0, 'x:\\VS_CODE\\LLM_track\\CuPy_dev')
from tokenizer.bpe import BPETokenizer, GPT4_SPLIT_PATTERN

sample_text = '''The Imperial Russian Navy (Russian: Российский императорский флот) operated as the navy of the Russian Tsardom and later the Russian Empire from 1696 to 1917.[c] Formally established in 1696, it lasted until being dissolved in the wake of the February Revolution and the declaration of the Russian Republic in 1917. It developed from a smaller force that had existed prior to Tsar Peter the Great's founding of the modern Russian navy during the Second Azov campaign in 1696[3], and expanded in the second half of the 18th century before reaching its peak strength by the early part of the 19th century, behind only the British and French fleets in terms of size.
The Imperial Navy drew its officers from the aristocracy of the Empire'''

model = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
model.load('bpe.model')

ids = model.encode(sample_text)
print(model.decode(ids))