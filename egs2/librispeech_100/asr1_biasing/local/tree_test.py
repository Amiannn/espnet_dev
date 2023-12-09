import json
import random
import numpy as np
import sentencepiece as spm

from local.utils import read_file

def get_key(elements):
    return " ".join([str(e) for e in elements])

def build_trie(elements_list):
    tree     = {}
    ele2tree = {}
    for elements in elements_list:
        now    = tree
        childs = []
        for element in elements:
            if element not in now:
                now[element] = {}
            now = now[element]
            childs.append(now)
        key = get_key(elements)
        ele2tree[key] = childs

    for key in ele2tree:
        ele2tree[key] = [
            sorted(
                child.keys()
            ) for child in ele2tree[key]
        ]
    return tree, ele2tree

def search_tree(tree, word2tree, words):
    masks  = []
    tokens = []

    first_level  = list(tree.keys())
    max_mask_len = len(first_level)
    for word in words:
        now = tree
        key = get_key(word)
        subword_length = len(word)
        # cache
        if key in word2tree:
            mask     = word2tree[key]
            mask[-1] = first_level
            masks.extend(mask)
            continue
        for i in range(subword_length):
            char = word[i]
            if (i + 1) == subword_length:
                masks.append(first_level)
            elif char in now:
                now = now[char]
                masks.append(list(now.keys()))
            elif now != tree:
                masks.extend([
                    [] for _ in range(subword_length - i - 1)
                ] + [first_level])
                break
            else:
                masks.append(first_level)
            if max_mask_len < len(masks[-1]):
                len(masks[-1])
    return masks, max_mask_len

if __name__ == '__main__':
    biasing_words_path = "./local/rareword_f15.txt"
    spm_model_path     = "./data/en_token_list/bpe_unigram600suffix/bpe.model"
    blist = [b[0] for b in read_file(biasing_words_path, sp=" ")]

    tokenizer = spm.SentencePieceProcessor(model_file=spm_model_path)
    # blist_subword = [tokenizer.encode(b, out_type=str) for b in blist]
    blist_subword = [tokenizer.encode(b) for b in blist]


    ref = "'''K HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE"

    # ref_word_tokens    = [tokenizer.encode(t, out_type=str) for t in ref.split(' ')]
    ref_word_tokens    = [tokenizer.encode(t) for t in ref.split(' ')]
    ref_subword_tokens = []
    for word in ref_word_tokens:
        ref_subword_tokens.extend(word)

    for _ in range(50):
        random.shuffle(blist_subword)
        tree, ele2tree = build_trie(blist_subword[:500])
        masks, max_mask_len = search_tree(tree, ele2tree, ref_word_tokens)
        print(f'max_len: {max_mask_len}')
        # for i in range(len(ref_subword_tokens)):
        #     print(f'token: {ref_subword_tokens[i]}, length: {len(masks[i])}, mask: {masks[i]}')
        #     print('_' * 30)