import nltk
import stanza

def init_stanza_pipeline(tokenize_no_ssplit=False):
    return stanza.Pipeline(lang="id", processors="tokenize,mwt,pos", package="gsd", tokenize_no_ssplit=tokenize_no_ssplit)

def stanza_pipeline_document_process(stanza_pipeline: stanza.Pipeline, document: str) -> list[nltk.Tree]:
    result: list[nltk.Tree] = []
    for index, stanza_sentence in enumerate(stanza_pipeline(document).sentences):
        subtree_list: list[nltk.Tree] = []

        for stanza_token in stanza_sentence.tokens:
            first_word = stanza_token.words[0]
            subtree = nltk.Tree(
                f'{first_word.upos};{"" if first_word.feats is None else first_word.feats}',
                [stanza_token.text]
            )
            subtree_list.append(subtree)
    
        tree = nltk.Tree(f"S;id={index}", subtree_list)
        result.append(tree)

    return result
