"""
Created on 2022/09/12
@ref https://github.com/catSirup/KorEDA/blob/master/eda.py
"""
import os
import random
import re
from typing import Dict, List

import joblib


def _get_only_hangul(line: str) -> str:
    parsed_text = re.sub(r"^[ㄱ-ㅎㅏ-ㅣ가-힣]*$", "", line)
    return parsed_text


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################
def _synonym_replacement(
    wordnet: Dict[str, List[str]], words: List[str], n: int
) -> List[str]:
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = _get_synonyms(wordnet, random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    if len(new_words) != 0:
        sentence = " ".join(new_words)
        new_words = sentence.split(" ")

    else:
        new_words = ""

    return new_words


def _get_synonyms(wordnet: Dict[str, List[str]], word: str) -> List[str]:
    synomyms = []

    try:
        for syn in wordnet[word]:
            for s in syn:
                synomyms.append(s)
    except:
        pass

    return synomyms


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def _random_deletion(words: List[str], p: float) -> List[str]:
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def _random_swap(words: List[str], n: int) -> List[str]:
    new_words = words.copy()
    for _ in range(n):
        new_words = _swap_word(new_words)

    return new_words


def _swap_word(new_words: List[str]) -> List[str]:
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################
def _random_insertion(
    wordnet: Dict[str, List[str]], words: List[str], n: int
) -> List[str]:
    new_words = words.copy()
    for _ in range(n):
        _add_word(wordnet, new_words)

    return new_words


def _add_word(wordnet: Dict[str, List[str]], new_words: List[str]) -> None:
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        if len(new_words) >= 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = _get_synonyms(wordnet, random_word)
            counter += 1
        else:
            random_word = ""

        if counter >= 10:
            return

    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


class EDA:
    def __init__(
        self,
        root_data_dir: str = "./data",
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        alpha_rd: float = 0.1,
        num_aug: int = 9,
    ) -> None:
        self._wordnet = joblib.load(os.path.join(root_data_dir, "wordnet"))
        self._alpha_sr = alpha_sr
        self._alpha_ri = alpha_ri
        self._alpha_rs = alpha_rs
        self._alpha_rd = alpha_rd
        self._num_aug = num_aug

    def __call__(
        self,
        sentence: str,
    ) -> List[str]:
        sentence = _get_only_hangul(sentence)
        words = sentence.split(" ")
        words = [word for word in words if word is not ""]
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(self._num_aug / 4) + 1

        n_sr = max(1, int(self._alpha_sr * num_words))
        n_ri = max(1, int(self._alpha_ri * num_words))
        n_rs = max(1, int(self._alpha_rs * num_words))

        # sr
        for _ in range(num_new_per_technique):
            a_words = _synonym_replacement(self._wordnet, words, n_sr)
            augmented_sentences.append(" ".join(a_words))

        # ri
        for _ in range(num_new_per_technique):
            a_words = _random_insertion(self._wordnet, words, n_ri)
            augmented_sentences.append(" ".join(a_words))

        # rs
        for _ in range(num_new_per_technique):
            a_words = _random_swap(words, n_rs)
            augmented_sentences.append(" ".join(a_words))

        # rd
        for _ in range(num_new_per_technique):
            a_words = _random_deletion(words, self._alpha_rd)
            augmented_sentences.append(" ".join(a_words))

        augmented_sentences = [
            _get_only_hangul(sentence) for sentence in augmented_sentences
        ]
        random.shuffle(augmented_sentences)

        if self._num_aug >= 1:
            augmented_sentences = augmented_sentences[: self._num_aug]
        else:
            keep_prob = self._num_aug / len(augmented_sentences)
            augmented_sentences = [
                s for s in augmented_sentences if random.uniform(0, 1) < keep_prob
            ]

        return augmented_sentences
