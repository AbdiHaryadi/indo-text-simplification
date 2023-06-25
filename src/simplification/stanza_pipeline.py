import re
import nltk
import stanza

def init_stanza_pipeline(tokenize_no_ssplit=False, strategy=1):
    nlp = stanza.Pipeline(lang="id", processors="tokenize,mwt,pos", package="gsd", tokenize_no_ssplit=tokenize_no_ssplit)

    if strategy == 2 or strategy == 3:
        def stanza_pipeline_document_process_strategy_2(document: str) -> list[nltk.Tree]:
            result: list[nltk.Tree] = []

            for index, stanza_sentence in enumerate(nlp(document).sentences):
                subtree_list: list[nltk.Tree] = []
                for stanza_token in stanza_sentence.tokens:
                    first_word = stanza_token.words[0]
                    upos = first_word.upos
                    text = stanza_token.text

                    feats_string = extract_feats_string(first_word)
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

        return stanza_pipeline_document_process_strategy_2
    
    else:
        def stanza_pipeline_document_process_strategy_1(document: str) -> list[nltk.Tree]:
            result: list[nltk.Tree] = []
            for index, stanza_sentence in enumerate(nlp(document).sentences):
                subtree_list: list[nltk.Tree] = []

                for stanza_token in stanza_sentence.tokens:
                    first_word = stanza_token.words[0]
                    node = f'{first_word.upos};{extract_feats_string(first_word)}'
                    # if include_depparse:
                    #     node += ";" + f"head={first_word.head};deprel={first_word.case}"

                    subtree = nltk.Tree(
                        node,
                        [stanza_token.text]
                    )
                    subtree_list.append(subtree)
            
                tree = nltk.Tree(f"S;id={index}", subtree_list)
                result.append(tree)

            return result
        
        return stanza_pipeline_document_process_strategy_1

def extract_feats_string(first_word):
    return "" if first_word.feats is None else first_word.feats
