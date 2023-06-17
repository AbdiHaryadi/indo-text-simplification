import nltk
import re

def word_list_to_sentence(word_list: list[str]) -> str:
    sentence = ""
    expected_close_symbol = []
    next_no_space = True
    for word in word_list:
        if next_no_space:
            sentence += word
            next_no_space = False
        elif word in [".", ","]:
            sentence += word
        elif len(expected_close_symbol) > 0 and expected_close_symbol[-1] == ")" == word:
            sentence += word
            expected_close_symbol.pop()
        elif word == "(":
            sentence += " " + word
            next_no_space = True
            expected_close_symbol.append(")")
        elif len(sentence) >= 2 and sentence[-2] in "0123456789" and sentence[-1] in [".", ","]:
            sentence += word
        elif word == "Rp":
            sentence += " " + word
            next_no_space = True
        else:
            sentence += " " + word

    sentence = re.sub(r"\[.+\]", "", sentence)

    return sentence

def sentence_to_word_list(sentence):
    return nltk.word_tokenize(sentence)
