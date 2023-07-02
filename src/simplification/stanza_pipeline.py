import re
import nltk
import stanza

def init_stanza_pipeline(tokenize_no_ssplit=False, strategy=1):
    nlp = stanza.Pipeline(lang="id", processors="tokenize,mwt,pos", package="gsd", tokenize_no_ssplit=tokenize_no_ssplit)

    def stanza_pipeline_document_process(document: str) -> list[nltk.Tree]:
        result: list[nltk.Tree] = []

        for index, stanza_sentence in enumerate(nlp(document).sentences):
            subtree_list: list[nltk.Tree] = []
            for stanza_token in stanza_sentence.tokens:
                first_word = stanza_token.words[0]
                upos = first_word.upos
                text = stanza_token.text

                if strategy == 5:
                    if upos == "SCONJ" and text.lower() == "hingga":
                        upos = "ADP"
                    elif upos == "ADP" and text.lower() == "setelah":
                        upos ="SCONJ"

                feats_string = extract_feats_string(first_word)
                if strategy == 2 or strategy == 3 or strategy == 5:
                    if upos == "NUM" and re.match(r"^[0-9]{4}$", text):
                        if feats_string != "":
                            feats_string += "|"
                        
                        feats_string += "Year=Yes"

                node = f'{upos};{feats_string}'

                subtree = nltk.Tree(
                    node,
                    [text]
                )
                subtree_list.append(subtree)
        
            tree = nltk.Tree(f"S;id={index}", subtree_list)
            result.append(tree)

        return result
    
    return stanza_pipeline_document_process

def extract_feats_string(first_word):
    return "" if first_word.feats is None else first_word.feats
