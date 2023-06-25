import logging
from nltk.tree import Tree
from .simplification.agreement import extract_agreements_from_head_noun
from .simplification.clause_boundary import extract_boundaries
from .simplification.grammatical_function import extract_grammatical_function
from .simplification.noun_chunk import init_noun_chunk_pipeline

from .simplification.stanza_pipeline import init_stanza_pipeline
from .simplification.relative_clause_attachment import relative_clause_attachment
from .simplification.resolve_third_person_pronouns import init_third_person_pronouns_pipeline
from .simplification.transform import transform
from .utils import word_list_to_sentence

class TextSimplifier:
    def __init__(self, tokenize_no_ssplit=False, strategy=1):
        self._stanza_pipeline = init_stanza_pipeline(
            tokenize_no_ssplit=tokenize_no_ssplit,
            strategy=strategy
        )
        self._noun_chunk_pipeline = init_noun_chunk_pipeline(strategy=strategy)
        self._third_person_pronouns_pipeline = init_third_person_pronouns_pipeline(strategy=strategy)
        self._strategy = strategy
        print("Strategy:", strategy)
    
    def simplify(self, document: str) -> list[list[str]]:
        result = self._stanza_pipeline(document)
        result = self._noun_chunk_pipeline(result)
        Tree("D", result).draw()
        result = extract_grammatical_function(result)
        result = extract_agreements_from_head_noun(result)
        result = self._third_person_pronouns_pipeline(result)
        result = relative_clause_attachment(result)
        result = extract_boundaries(result, strategy=self._strategy)
        result = transform(result)
        result = tree_list_to_simplified_sentences_list(result)
        return result
    
def tree_list_to_simplified_sentences_list(tree_list: list[Tree]) -> list[list[str]]:
    simplified_sentences_list: list[list[str]] = []
    for tree in tree_list:
        _, id_attribute = tree.label().split(";")
        _, id_value = id_attribute.split("=")
        sentence_id, *_ = id_value.split(".")
        sentence_id = int(sentence_id)

        while sentence_id >= len(simplified_sentences_list):
            simplified_sentences_list.append([])

        word_list = tree.leaves()
        simplified_sentence = word_list_to_sentence(word_list)
        simplified_sentence = simplified_sentence[0].upper() + simplified_sentence[1:]
        simplified_sentences_list[sentence_id].append(simplified_sentence)

    return simplified_sentences_list

if __name__ == "__main__":
    logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
    simplifier = TextSimplifier(strategy=4)

    interrupt = False
    while not interrupt:
        try:
            document = input("Document: ")
            result = simplifier.simplify(document)
            print("Result:")
            print(result)
            print()
        except KeyboardInterrupt:
            interrupt = True
    
