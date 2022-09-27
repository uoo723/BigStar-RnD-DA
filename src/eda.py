"""
Created on 2022/09/12
@ref https://github.com/catSirup/KorEDA/blob/master/eda.py
"""
import os
import random
import re
from typing import Dict, List, Optional

import joblib
import numpy as np


def _get_only_hangul(line: str) -> str:
    parsed_text = re.sub(r"^[ㄱ-ㅎㅏ-ㅣ가-힣]*$", "", line)
    return parsed_text


def _get_synonyms(word: str, wordnet: Dict[str, List[str]]) -> List[str]:
    return wordnet[word] if word in wordnet else []


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################
def _synonym_replacement(
    sentence: str, wordnet: Dict[str, List[str]], n: int
) -> Optional[str]:
    words = sentence.strip().split(" ")

    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = _get_synonyms(random_word, wordnet)

        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            words = [synonym if word == random_word else word for word in words]
            num_replaced += 1

        if num_replaced >= n:
            break

    return " ".join(words) if num_replaced == 0 else None


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def _random_deletion(sentence: str, p: float) -> Optional[str]:
    words = sentence.strip().split(" ")

    new_words = []
    for word in words:
        if np.random.binomial(1, p) == 0:
            new_words.append(word)

    if len(new_words) == 0:
        return None

    return " ".join(new_words)


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def _random_swap(sentence: str, n: int) -> Optional[str]:
    new_words = sentence.strip().split(" ")

    if len(new_words) == 1:
        return None

    for _ in range(n):
        new_words = _swap_word(new_words)

    return " ".join(new_words)


def _swap_word(words: List[str]) -> List[str]:
    new_words = words.copy()

    ridx_1 = np.random.randint(len(new_words))
    ridx_2 = ridx_1

    counter = 0
    while ridx_2 == ridx_1:
        ridx_2 = np.random.randint(len(new_words))
        counter += 1
        if counter > 3:
            return words

    new_words[ridx_1], new_words[ridx_2] = (
        new_words[ridx_2],
        new_words[ridx_1],
    )
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################
def _random_insertion(
    sentence: str, wordnet: Dict[str, List[str]], n: int
) -> Optional[str]:
    new_words = sentence.strip().split(" ")
    added = []
    for _ in range(n):
        added.append(_add_word(new_words, wordnet))

    return " ".join(new_words) if any(added) else None


def _add_word(words: List[str], wordnet: Dict[str, List[str]]) -> bool:
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = words[np.random.randint(len(words))]
        synonyms = _get_synonyms(random_word, wordnet)
        counter += 1
        if counter > 3:
            return False

    random_synonym = synonyms[np.random.randint(len(synonyms))]
    ridx = np.random.randint(len(words))
    words.insert(ridx, random_synonym)

    return True


class EDA:
    def __init__(
        self,
        root_data_dir: str = "./data",
        p_sr: float = 0.25,
        p_ri: float = 0.25,
        p_rs: float = 0.25,
        p_rd: float = 0.25,
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        alpha_rd: float = 0.1,
        num_aug: int = 9,
    ) -> None:
        self._wordnet = joblib.load(os.path.join(root_data_dir, "wordnet.joblib"))
        self._p_sr = p_sr
        self._p_ri = p_ri
        self._p_rs = p_rs
        self._p_rd = p_rd
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
        num_words = len(sentence.strip().split(" "))

        augmented_sentences = []
        num_new_per_technique = np.random.multinomial(
            self._num_aug,
            [self._p_sr, self._p_ri, self._p_rs, self._p_rd],
        )

        n_sr = max(1, int(self._alpha_sr * num_words))
        n_ri = max(1, int(self._alpha_ri * num_words))
        n_rs = max(1, int(self._alpha_rs * num_words))

        # sr
        for _ in range(num_new_per_technique[0]):
            a_sent = _synonym_replacement(sentence, self._wordnet, n_sr)
            if a_sent is not None:
                augmented_sentences.append(a_sent)

        # ri
        for _ in range(num_new_per_technique[1]):
            a_sent = _random_insertion(sentence, self._wordnet, n_ri)
            if a_sent is not None:
                augmented_sentences.append(a_sent)

        # rs
        for _ in range(num_new_per_technique[2]):
            a_sent = _random_swap(sentence, n_rs)
            if a_sent is not None:
                augmented_sentences.append(a_sent)

        # rd
        for _ in range(num_new_per_technique[3]):
            a_sent = _random_deletion(sentence, self._alpha_rd)
            if a_sent is not None:
                augmented_sentences.append(a_sent)

        if len(augmented_sentences) == 0:
            augmented_sentences.append(sentence)

        augmented_sentences = list(set(augmented_sentences))
        random.shuffle(augmented_sentences)

        return augmented_sentences
