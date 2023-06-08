import logging
from nltk.tree import Tree
from simplification.agreement import extract_agreements_from_head_noun
from simplification.clause_boundary import extract_boundaries
from simplification.grammatical_function import extract_grammatical_function
from simplification.noun_chunk import noun_chunk

from simplification.stanza_pipeline import init_stanza_pipeline, stanza_pipeline_document_process
from simplification.relative_clause_attachment import relative_clause_attachment
from simplification.resolve_third_person_pronouns import resolve_third_person_pronouns
from simplification.transform import transform
from utils import word_list_to_sentence

class TextSimplifier:
    def __init__(self, tokenize_no_ssplit=False):
        self._stanza_pipeline = init_stanza_pipeline(tokenize_no_ssplit=tokenize_no_ssplit)
    
    def simplify(self, document: str) -> list[str]:
        result = stanza_pipeline_document_process(self._stanza_pipeline, document)
        result = noun_chunk(result)
        result = extract_grammatical_function(result)
        result = extract_agreements_from_head_noun(result)
        result = resolve_third_person_pronouns(result)
        result = relative_clause_attachment(result)
        result = extract_boundaries(result)
        result = transform(result)
        result = tree_list_to_sentences(result)
        return result
    
def tree_list_to_sentences(tree_list: list[Tree]) -> list[str]:
    sentence_list: list[str] = []
    for tree in tree_list:
        word_list = tree.leaves()
        sentence = word_list_to_sentence(word_list)
        sentence_list.append(sentence)

    return sentence_list

if __name__ == "__main__":
    logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
    simplifier = TextSimplifier()

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
    
