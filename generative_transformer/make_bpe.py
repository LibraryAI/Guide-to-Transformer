# https://github.com/OpenNMT/OpenNMT-py/blob/master/tools/learn_bpe.py

import tensorflow as tf
import numpy as np
import random
import json
import ftfy
import spacy
import itertools
import collections
import re
import copy
import codecs
import sys

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    # 여러 출처의 텍스트의 서로 다른 포맷을 규격화
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return text.strip()

class TextDictionary(object):
    '''
    methods are divided into two tasks
    
     1) Create dictionary of words from the given text data
        : add_vocab_dictionary(), make_vocab_dictionary()
     2) Create a file of BPE pairs with the word dictionary created from task 1
        : get_stats(), merge_pairs(), update_stats(), prune_stats(), bpe()
    '''
    
    # 1) input text 데이터 내의 단어들의 딕셔너리 생성
    # 2) input text 데이터로 생성된 단어 딕셔너리를 이용해 BPE pair 생성 후 저장
    
    def __init__(self):
        '''
        creates deafultdict of defaultdict(int) to store index of words of specific bpe pairs
        '''
        self.dict_rank = collections.defaultdict(int)
        self.bpe_stat = collections.defaultdict(int)
        self.bpe_index = collections.defaultdict(lambda: collections.defaultdict(int))
        self.nlp = spacy.load("en", disable=['parser', 'tagger', 'ner', 'textcat'])
        self.big_stats = None

    def add_vocab_dictionary(self, texts):
        '''
        texts: untokenized string or string list
        standardize the data from multiple sources, tokenize and create dictionary of tokens
        '''
        # texts: 토큰화가 되지 않은 스트링 혹은 스트링 리스트
        
        # 여러 출처의 데이터를 표준화하고 토큰화한 후, 해당 토큰들의 딕셔너리 생성
        # defaultdict(int)가 아닌 Count()로 진행해도 됨
        
        doc = self.nlp(text_standardize(ftfy.fix_text(texts)))
        for token in doc:
            self.dict_rank[token.text.lower()] += 1

    def make_vocab_dictionary(self, batch):
        for i in range(len(batch)):
            self.add_vocab_dictionary(batch[i])

    def get_stats(self, sorted_vocab):
        '''
        sorted_vocab: sorted (vocab, frequency) list by ascendant frequency; the vocabs are from make_vocab_dictionary()

        from sorted_vocab word,freq list, create bigram pair dictionary of adjacent tokens and create dictionary of dictionary
        
        *key of bpe_index : bigram tuple
        *value of bpe_index : dictionary with key, value as index, frequency
        '''

        # sorted_vocab: (vocab, frequency) 리스트. frequency의 역순으로 오름차순으로 정렬되어있음. 

        # sorted_vocab 리스트로부터 bigram pair 딕셔너리 생성, 토큰 인덱스 저장을 위한 bpe_index 생성
        # key of bpe_index : bigram tuple
        # value of bpe_index : dictionary with key, value as index, frequency

        for idx, (word, freq) in enumerate(sorted_vocab):
            prev_char = word[0]
            for char in word[1:]:
                self.bpe_stat[prev_char, char] += freq
                self.bpe_index[prev_char, char][idx] += 1
                prev_char = char
        self.big_stats = copy.deepcopy(self.bpe_stat)


    def merge_pairs(self, pair, vocab):
        # pair : tuple of two strings
        # vocab : list of (word, freq) tuple
        
        # 1) self.bpe_index에서 pair에 매칭되는 defaultdict를 찾아서, vocab list에서의 word index와 frequency를 찾는다
        # 2) vocab list에서 pair가 속해 있는 word를 색인 후, pair를 merge한 새로운 word의 튜플을 구한다
        # 3) 기존 word, 새로운 new_word, index, frequency를 반환한다

        first, second = pair
        pair_joined = ''.join(pair)
        changes = []
        pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')

        for idx, freq in self.bpe_index[pair].items():
            if freq < 1:
                continue
            word, freq = vocab[idx]
            new_word = ' '.join(word)
            new_word = pattern.sub(pair_joined, new_word)
            new_word = tuple(new_word.split())

            vocab[idx] = (new_word, freq)
            changes.append((word, new_word, idx, freq))

        return changes

    def update_stats(self, pair, changes, vocab):
        # pair : tuple of tow strings
        # change : list of tuples (word, new_word, index, frequency)
        # vocab : sorted word list

        # 1) merge되기 전 pair의 frequency는 0 으로 한다 (해당 pair가 사라졌기 때문)
        # 2) merge가 된 새로운 pair를 self.bpe_index에 추가한다
        # 3) merge가 된 새로운 pair를 self.bpe_stat에 추가한다
        # 4) index에 해당하는 vocab 리스트의 각 단어들에 다음을 수행한다
        #1) merge + 전, 후에 해당하는 캐릭터의 pair를 구하고, self.bpe_index, self.bpe_stat 에 해당 조합을 추가한다
        #1) A (BC) 와 같은 경우: A BC 의 freq를 늘린다
        #2) (BC) B 와 같은 경우: BC B 의 freq를 늘린다
        #3) (BC) (BC) 와 같은 경우: pass한다 
        #2) merge되기 전의 + 전, 후에 해당하는 캐릭터의 pair를 찾고, self.bpe_index, self.bpe_stat 에 해당 조합의 freq를 내린다
        #1) A (B C) 와 같은 경우: A, B를 줄인다
        #2) A (B C) (B C) 와 같은 경우: C, B 를 줄이지 않는다.  1)에 해당하기 때문
        #3) (B C) B 와 같은 경우: C, B 를 줄인다 2)는 이 경우의 확장         

        self.bpe_stat[pair] = 0
        self.bpe_index[pair] = collections.defaultdict(int)
        first, second = pair
        new_pair = first + second

        for word, new_word, idx, freq in changes:
            i = 0
            while True:
                try:
                    i = new_word.index(new_pair, i)
                except ValueError:
                    break
                if i:
                    prev = new_word[i-1:i+1]
                    self.bpe_stat[prev] += freq
                    self.bpe_index[prev][idx] += 1
                if i < len(new_word) - 1 and new_word[i+1] != new_pair:
                    nex = word[i:i+2]
                    self.bpe_stat[nex] += freq
                    self.bpe_index[nex][idx] += 1
                i += 1

            i = 0
            while True:
                try:
                    i = word.index(first, i)
                except ValueError:
                    break
                if i < len(word) - 1 and word[i+1] == second:
                    if i:
                        prev = word[i-1:i+1]
                        self.bpe_stat[prev] -= freq
                        self.bpe_index[prev][idx] -= 1
                    if i < len(word) -2:
                        if word[i + 2] != first or i >= len(word) - 3 or word[i + 3] != second:
                            nex = word[i + 1:i + 3]
                            self.bpe_stat[nex] -= freq
                            self.bpe_index[nex][idx] -= 1
                    i += 2
                else:
                    i += 1

    def prune_stats(self, threshold):
        #
        for item, freq in list(self.bpe_stat.items()):
            if freq < threshold:
                del self.bpe_stat[item]
                if freq < 0:
                    self.big_stats[item] += freq
                else:
                    self.big_stats[item] = freq

    def bpe(self, outfile, num_symbols=40000, min_frequency=2):
        '''
        1) 딕셔너리 내에 있는 단어들을 돌아가며 merging
            1) 전체에서 가장 frequency가 높은 pair를 선별
            2) 해당 pair를 paring화 하고, merging 다시하기
            3) 
        '''

        vocab = dict([(tuple(x[:-1]) + (x[-1] + '</w>',), y) for (x,y) in self.dict_rank.items()])
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        
        self.get_stats(sorted_vocab)
        threshold = max(self.bpe_stat.values()) / 10
        for i in range(num_symbols):
            if self.bpe_stat:
                most_frequent = max(self.bpe_stat, key=lambda x: (self.bpe_stat[x], x))

            if not self.bpe_stat or (self.bpe_stat[most_frequent] < threshold):
                self.prune_stats(threshold)
                self.bpe_stat = copy.deepcopy(self.big_stats)
                most_frequent = max(self.bpe_stat, key=lambda x: (self.bpe_stat[x], x))
                # threshold is inspired by Zipfian assumption, but should only affect speed
                threshold = self.bpe_stat[most_frequent] * i / (i + 10000.0)
                self.prune_stats(threshold)

            if self.bpe_stat[most_frequent] < min_frequency:
                sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
                break

            if verbose:
                sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], self.bpe_stat[most_frequent]))
            outfile.write('{0} {1}\n'.format(*most_frequent))
            changes = self.merge_pairs(most_frequent, sorted_vocab)
            self.update_stats(most_frequent, changes, sorted_vocab)
            self.bpe_stat[most_frequent] = 0
            if not i % 100:
                self.prune_stats(threshold)

'''
print(len(data["data"]))
print(data["data"][1].keys())
print(len(data["data"][1]["paragraphs"]))
print(data["data"][1]["paragraphs"][1].keys())
print(len(data["data"][1]["paragraphs"][1]["qas"]))
print(data["data"][1]["paragraphs"][1]["qas"][1].keys())
data["data"][1]["paragraphs"][1]["qas"][1]

'''

sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
outfile = codecs.open('D:/attention_is_all/bpe_40000.txt', 'w', encoding='utf-8')

with open('D:/download/train-v2.0.json') as data_file:    
    data = json.load(data_file)

for datas in data["data"]:
    for paragraphs in datas['paragraphs']:
        textdict.add_vocab_dictionary(paragraphs['context'])

textdict.bpe(outfile=outfile)
