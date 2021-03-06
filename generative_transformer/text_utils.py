# https://github.com/ceshine/finetune-transformer-lm/blob/master/inspect_lm.py

import re
import ftfy
import json
import spacy
import numpy as np

'''
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)
outfile = codecs.open(args.output.name, 'w', encoding='utf-8')

print(len(data["data"]))
print(data["data"][1].keys())
print(len(data["data"][1]["paragraphs"]))
print(data["data"][1]["paragraphs"][1].keys())
print(len(data["data"][1]["paragraphs"][1]["qas"]))
print(data["data"][1]["paragraphs"][1]["qas"][1].keys())
data["data"][1]["paragraphs"][1]["qas"][1]
'''

with open('D:/download/train-v2.0.json') as data_file:    
    data = json.load(data_file)


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return text.strip()

class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(bpe_path).read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts):
        texts_tokens = []
        '''
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        else:
        '''
        for text in texts:
            text = self.nlp(text_standardize(ftfy.fix_text(text)))
            text_tokens = []
            for token in text:
                text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
            texts_tokens.append(text_tokens)
        return texts_tokens


def transform_texts(list_of_texts):
    tokens = TEXT_ENCODER.encode(list_of_texts, verbose=False)
    n_batch = len(tokens)
    xmb = np.zeros((n_batch, N_CTX, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, N_CTX), dtype=np.float32)
    for i, x in enumerate(tokens):
        x1 = x[:N_CTX]
        l1 = len(x1)
        print(f"length: {l1}")
        xmb[i, :l1, 0] = x1
        mmb[i, :l1] = 1
    xmb[:, :, 1] = np.arange(N_VOCAB, N_VOCAB+N_CTX)
    return xmb, mmb

