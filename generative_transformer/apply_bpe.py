
# -*- coding: utf-8 -*-
# Author: Rico Sennrich
# flake8: noqa

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.
Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""
# This file is retrieved from https://github.com/rsennrich/subword-nmt

from __future__ import unicode_literals, division

import sys
import codecs
import io
import argparse
import json
import re
from collections import defaultdict

# hack for python2/3 compatibility
from io import open
argparse.open = open

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. The strings provided in glossaries will not be affected" +
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords")

    return parser


class BPE(object):

    def __init__(self, codes, separator='@@', vocab=None, glossaries=None):

        self.bpe_codes = [tuple(item.split()) for item in codes]

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict(
            [(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict(
            [(pair[0] + pair[1], pair) for pair, i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []

        self.cache = {}

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""

        # sentence: 표준화되고 wihtespace-토큰화 된 스트링

        # segment method는 BPE를 수행하는 전체 과정을 포함
        # 1) _isolate_glossaries와 isolate_glossaries 함수를 통해 glossary가 주어졌다면 glossary들을 기준으로 단어를 잘게 쪼갬
        # 2) glossary를 기준으로 잘게 쪼개진 단어의 segment를 encode 함수를 통해 캐릭터 단위로 분해한 후, Byte pair encoding을 수행하며 가장 빈번한 BPE 기준으로 merge를 반복
        # 3) vocab가 주어졌다면, 이렇게 구해진 새 토큰 리스트를 check_vocab_and_split을 통해 vocabulary와 그렇지 않은 OOV들을 다시 분해

        # ex. input sentence= "긴 하루였다", output= "긴 하루@@ 였@ 다"
        # (예시에서 볼 수 있듯이 원 문장에서 떨어져 있던 단어 사이에는 새로운 구분자가 없다.)


        output = []
        for word in sentence.split():
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return ' '.join(output)

    def _isolate_glossaries(self, word):
        # word: 단어 스트링

        # glossaries 리스트에 있는 모든 glossary 들을 기준으로 split된 segment 리스트를 반환
        # ex. input word = "비행기땅콩먹는비행기땅콩비행기?", glossaries = [땅콩, 비행], 
        #     output = ["비행", "기", "땅콩", "먹는", "비행", "기", "땅콩", "비행", "기?"]


        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                             for out_segments in isolate_glossary(segment, gloss)]
        return word_segments

def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.
    Returns a list of subwords. In which all 'glossary' glossaries are isolated 
    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """

    # segment, glossary; 스트링, 스트링
    
    # glossary 스트링이 segment 스트링에 포함된 경우, glossary를 기준으로 split 하기
    # ex. glossary = 땅콩, segment = "땅콩버터먹고땅콩먹자", 
    #     output = ["땅콩", "버터먹고", "땅콩", "먹자"]

    if word == glossary or glossary not in word:
        return [word]
    else:
        splits = word.split(glossary)
        segments = [segment.strip() for split in splits[:-1]
                    for segment in [split, glossary] if segment != '']
        return segments + [splits[-1].strip()] if splits[-1] != '' else segments




def get_pairs(word):
    """Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """

    # word; 스트링 리스트

    # 인풋 리스트 내의 인접한 두 스트링의 pair를 원소로 하는 집합 구하기
    # ex. word = ["20, "018", "년", "가", "즈"], 
    #     output = {("20", "018"), ("018", "년"), ("년", "가"), ("가", "즈")}

    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries=None):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    # encode는 BPE를 적용하는 함수다. glossary가 주어진 경우, glossary를 기준으로 분해된 segment를 인풋으로 받아 다음의 과정을 수행한다.
    # 1) segment 스트링을 튜플로 분해한 후, get_pairs 함수를 통해 해당 segment 캐릭터들의 pair집합을 구한다
    # 2) pair집합과 bpe_codes 간에 교집합이 없을 때 까지 다음을 수행한다
    # 2-1) pair 묶음들과 bpe_codes의 겹치는 원소 중 가장 처음 원소(빈도수가 가장 많은)를 bigram에 할당
    # 2-2) 1의 튜플을 bigram에 해당하는 스트링의 원소가 있는 인덱스 까지의 원소들을 하나의 스트링으로 만들어 new_word 리스트에 넣고, bigram을 append한 후,  다음 인덱스 부터 2-2를 다시 반복한다
    # 2-3) 2-2에서 구해진 new_word 리스트를 튜플로 변환하고 get_pairs를 통해 pair 집합을 구한다.
    # 3) Vocab가 주어졌다면, 2의 과정을 통해 Byte-pair로 뭉쳐진 토큰들을 check_vocab_and_split을 통해 해당 토큰들이 vocab 집합에 속해있는지 확인 후 OOV들은 잘게 쪼갠다
    # 4) Vocabulary와 잘게 쪼개진 OOV 스트링으로 이루어진 리스트를 반환한다

    if orig in cache:
        return cache[orig]

    if orig in glossaries:
        cache[orig] = (orig,)
        return (orig,)

    word = tuple(orig[:-1]) + (orig[-1] + '</w>',)
    
    pairs = get_pairs(word)

    if not pairs:
        return orig

    while True:
        bigram = min(pairs, key=lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
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

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
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

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>', ''),)

    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word


def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    # segment; 스트링

    # 조건부에 일치하는 스트링을 yield로 반환
    # ex. bpe_codes= {'는다</w>': ('는', '다</w>')}, vocab= {'는', '다'}, segment= '먹는다', 
    #     output = '먹는다'
    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item


def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    # orig; 튜플

    # orig 튜플 속의 원소들이 vocab 집합에 속했는지 확인 후 vocab에 속할 때 까지 recursive_split을 통해 잘게 쪼개 OOV(Out of Vcoabulary)라면 잘게 쪼갠다.
    
    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary(vocab_file, threshold):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.split()
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary


def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.
    Returns a list of subwords. In which all 'glossary' glossaries are isolated 
    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    if word == glossary or glossary not in word:
        return [word]
    else:
        splits = word.split(glossary)
        segments = [segment.strip() for split in splits[:-1]
                    for segment in [split, glossary] if segment != '']
        return segments + [splits[-1].strip()] if splits[-1] != '' else segments


if __name__ == '__main__':

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    args.codes = codecs.open(args.codes.name, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.vocabulary:
        args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

    if args.vocabulary:
        vocabulary = read_vocabulary(
            args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    bpe = BPE(args.codes, args.separator, vocabulary, args.glossaries)

    for line in args.input:
        args.output.write(bpe.segment(line).strip())
        args.output.write('\n')     