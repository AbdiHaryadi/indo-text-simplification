import stanza
import logging

from nltk import trigrams
from nltk.chunk import RegexpChunkParser
from nltk.chunk.regexp import ChunkRule, StripRule, ExpandLeftRule, MergeRule, ExpandRightRule, SplitRule
from nltk.tree import Tree

import re 

def init_stanza_pipeline():
    return stanza.Pipeline(lang="id", processors="tokenize,mwt,pos", package="gsd")

def stanza_pipeline_document_process(stanza_pipeline: stanza.Pipeline, document: str) -> list[Tree]:
    return [
        Tree(
            "S",
            [
                Tree(
                    f'{stanza_word.upos};{"" if stanza_word.feats is None else stanza_word.feats}',
                    [stanza_word.text]
                )
                for stanza_word in stanza_sentence.words
            ]
        )
        for stanza_sentence in stanza_pipeline(document).sentences
    ]

def noun_chunk(tree_list: list[Tree]):
    parser = RegexpChunkParser(
        [
            ChunkRule(
                r"<(NOUN|PROPN|PRON);.*>+",
                "Chunk common noun, proper noun, and pronoun"
            ),
            StripRule(
                r"<.*PronType=Rel.*>",
                "Ignore relative pronoun"
            ),
            SplitRule(
                r"<PRON;.*>",
                r"<NOUN;.*>",
                "Split PRON and NOUN"
            ),
            ExpandLeftRule(
                r"<NUM;.*>",
                r"<NOUN;.*>",
                "Add number modifier"
            ),
            MergeRule(
                r"<NOUN;.*>",
                r"<NUM;.*>",
                "Merge NOUN and NUM"
            ),
            ExpandRightRule(
                r"<NOUN;.*>",
                r"<ADJ;.*>",
                "Add adjective modifier"
            ),
            ExpandRightRule(
                r"",
                r"<DET;.*PronType=Dem.*>",
                "Add demonstrative determiner modifier"
            ),
            ExpandLeftRule(
                r"<DET;.*>",
                r"<NOUN;.*>",
                "Add predeterminer modifier"
            ),
            ExpandRightRule( # Contoh kasus PROPN: Sultan Hamengkubuwono IX
                r"<(NOUN|PROPN);.*>",
                r"<NUM;.*>",
                "Add number modifier for noun"
            )
            # ExpandRightRule(
            #     r"<NOUN;.*><PROPN;.*>",
            #     r"<ADJ;.*>",
            #     "Add adjective modifier possibly because of entity"
            # ),
            # MergeRule(
            #     r"<NOUN;.*><NOUN;.*><ADJ;.*>",
            #     r"<NOUN;.*>",
            #     "Merge NOUN-NOUN-ADJ and NOUN, possibly because of organization"
            # )
        ],
        chunk_label="NP"
    )
    chunked_sentence_tree_list = [
        parser.parse(tree)
        for tree in tree_list
    ]

    np_id = 0
    for tree in chunked_sentence_tree_list:
        subtrees = [subtree for subtree in tree]
        for subtree in subtrees:
            if subtree.label() == "NP":
                subtree.set_label(f"NP;id={np_id}")
                np_id += 1

    return chunked_sentence_tree_list

def extract_grammatical_function(tree_list: list[Tree]):
    extracted_tree_list = tree_list

    # Extract pattern 1
    new_extracted_tree_list: list[Tree] = []
    for tree in extracted_tree_list:
        start_index_list = []
        subtrees = [subtree for subtree in tree]

        for index, subtree in enumerate(subtrees):
            if subtree.label().startswith("ADP"):
                start_index_list.append(index)

        for start_index in start_index_list:
            index = start_index + 1
            if index >= len(subtrees):
                continue

            subtree = subtrees[index]
            old_subtree_label = subtree.label()
            if (not old_subtree_label.startswith("NP")) or "gfunc" in old_subtree_label: # Including occupied NP
                continue
            
            subtree.set_label(f"{old_subtree_label};gfunc=OBLIQ")

        new_extracted_tree_list.append(Tree(
            "S",
            subtrees
        ))

    extracted_tree_list = new_extracted_tree_list

    # Extract pattern 2
    new_extracted_tree_list = []
    for tree in extracted_tree_list:
        start_index_list = []
        subtrees = [subtree for subtree in tree]

        for index, subtree in enumerate(subtrees):
            if subtree.label().startswith("NP"):
                start_index_list.append(index)

        for start_index in start_index_list:
            index = start_index + 1

            mismatch = False
            while index < len(subtrees) - 1:
                subtree = subtrees[index]
                if subtree.label().startswith("VERB"):
                    break

                elif subtree.label().startswith("PUNCT"):
                    index += 1
                    if index == len(subtrees):
                        mismatch = True
                        break

                    subtree = subtrees[index]
                    if subtree.label().startswith("PUNCT"):
                        mismatch = True
                        break

                    index += 1
                    while index < len(subtrees):
                        subtree = subtrees[index]
                        if subtree.label().startswith("VERB"):
                            mismatch = True
                            break
                        elif subtree.label().startswith("PUNCT"):
                            break
                        
                        index += 1

                    if index == len(subtrees):
                        mismatch = True
                        break

                    # else: good to go

                elif subtree.label().startswith("ADP"):
                    index += 1
                    if index == len(subtrees):
                        mismatch = True
                        break

                    subtree = subtrees[index]
                    if not subtree.label().startswith("NP"):
                        mismatch = True
                        break

                    # else: good to go

                else:
                    mismatch = True
                    break

                index += 1

            if index >= len(subtrees):
                mismatch = True

            else:
                subtree = subtrees[index]
                if not subtree.label().startswith("VERB"):
                    mismatch = True

            if mismatch:
                continue

            old_subtree_label = subtrees[start_index].label()
            if "gfunc" not in old_subtree_label: 
                subtrees[start_index].set_label(f"{old_subtree_label};gfunc=SUBJ")

        new_extracted_tree_list.append(Tree(
            "S",
            subtrees
        ))

    extracted_tree_list = new_extracted_tree_list

    # Extract pattern 3 and 4
    new_extracted_tree_list = []
    for tree in extracted_tree_list:
        start_index_list = []
        subtrees = [subtree for subtree in tree]

        for index, subtree in enumerate(subtrees):
            if subtree.label().startswith("VERB"):
                start_index_list.append(index)

        for start_index in start_index_list:
            index = start_index + 1
            if index == len(subtrees):
                continue

            subtree = subtrees[index]
            old_subtree_label = subtree.label()

            if not subtree.label().startswith("NP"):
                continue
            
            if "gfunc" not in old_subtree_label:
                subtree.set_label(f"{old_subtree_label};gfunc=DOBJ")

            index += 1
            while index < len(subtrees):
                subtree = subtrees[index]
                old_subtree_label = subtree.label()

                if not old_subtree_label.startswith("NP"):
                    break

                if "gfunc" not in old_subtree_label: 
                    subtree.set_label(f"{old_subtree_label};gfunc=IOBJ")

                index += 1

        new_extracted_tree_list.append(Tree(
            "S",
            subtrees
        ))

    extracted_tree_list = new_extracted_tree_list

    # Extract pattern 5
    new_extracted_tree_list = []
    for tree in extracted_tree_list:
        start_index_list = [-1]
        subtrees = [subtree for subtree in tree]

        for index, subtree in enumerate(subtrees):
            if subtree.label().startswith("PUNCT"):
                start_index_list.append(index)

        for start_index in start_index_list:
            index = start_index + 1
            while index < len(subtrees):
                subtree = subtrees[index]
                old_subtree_label = subtree.label()
                if old_subtree_label.startswith("NP"):
                    if "gfunc" not in old_subtree_label:
                        subtree.set_label(f"{old_subtree_label};gfunc=SUBJ")

                    break

                index += 1

        new_extracted_tree_list.append(Tree(
            "S",
            subtrees
        ))

    # Set default
    new_extracted_tree_list = []
    for tree in extracted_tree_list:
        start_index_list = [-1]
        subtrees = list([subtree for subtree in tree])

        for subtree in subtrees:
            old_subtree_label = subtree.label()
            if old_subtree_label.startswith("NP") and "gfunc" not in old_subtree_label:
                subtree.set_label(f"{old_subtree_label};") # Blank gfunc

        new_extracted_tree_list.append(Tree(
            "S",
            subtrees
        ))

    extracted_tree_list = new_extracted_tree_list

    return extracted_tree_list

def extract_restrictive_relative_clause_structure(tree: Tree, start_index: int) -> tuple[Tree, int]:
    new_tree, boundary_index = detect_restrictive_relative_clause_boundary(tree, start_index, keep_structure=False)
    if boundary_index == -1:
        return (tree, -1)
    
    clause_tree = Tree(
        node="SIMP-REST-CL",
        children=new_tree[start_index:boundary_index]
    )

    extracted_tree = Tree(
        node=new_tree.label(),
        children=new_tree[:start_index] + [clause_tree] + new_tree[boundary_index:]
    )

    return (extracted_tree, start_index + 1)

def extract_nonrestrictive_relative_clause_structure(tree: Tree, start_index: int) -> tuple[Tree, int]:
    new_tree, boundary_index = detect_nonrestrictive_relative_clause_boundary(tree, start_index, keep_structure=False)
    if boundary_index == -1:
        return (tree, -1)
    
    clause_tree = Tree(
        node="SIMP-NONREST-CL",
        children=new_tree[start_index:boundary_index]
    )

    extracted_tree = Tree(
        node=new_tree.label(),
        children=new_tree[:start_index] + [clause_tree] + new_tree[boundary_index:]
    )

    return (extracted_tree, start_index + 1)

def extract_appositive_structure(tree: Tree, start_index: int) -> tuple[Tree, int]:
    new_tree, boundary_index = detect_appositive_boundary(tree, start_index, keep_structure=False)
    if boundary_index == -1:
        return (tree, -1)
    
    clause_tree = Tree(
        node="SIMP-APPOS",
        children=new_tree[start_index:boundary_index]
    )

    # Do not simplify for one-or-two words appositive
    if len(clause_tree.leaves()) <= 2:
        return (new_tree, boundary_index)
    
    # Do not simplify for appositive that following "and" before verb or EOS.
    check_index = start_index + 1

    while check_index < len(new_tree) and not (new_tree[check_index].label().startswith("VERB") or new_tree[check_index, 0] == "." or new_tree[check_index, 0] in ["dan", "atau"]):
        check_index += 1

    # EOS, or new_tree[check_index] is verb, period, or and.

    if check_index < len(new_tree) and new_tree[check_index, 0] in ["dan", "atau"]:
        return (new_tree, boundary_index)
    
    # Find the attachment
    referred_candidates: list[Tree] = []
    check_index = start_index - 2

    # Find trailing prep phrase, if any
    while check_index >= 2 and new_tree[check_index].label().startswith("NP") and new_tree[check_index - 1].label().startswith("ADP"):
        referred_candidates.append(new_tree[check_index])
        check_index -= 2

    # check_index < 2 or not (new_tree[check_index].label().startswith("NP") and new_tree[check_index - 1].label().startswith("ADP"))
    if new_tree[check_index].label().startswith("NP"):
        referred_candidates.append(new_tree[check_index])

    appositive_np = new_tree[start_index]
    appositive_np_label = appositive_np.label()
    referred: None | Tree = None

    referred_candidates.sort(key=lambda x: get_salience(x.label()), reverse=True)
    for np in referred_candidates:
        unification_possible = True
        _, np_id_string, _, np_agreement_list_string = np.label().split(";")
        np_id = np_id_string[3:]

        logging.debug(f"Agreement filtering: {np_agreement_list_string=}")
        if np_agreement_list_string != "":
            np_agreement_list = np_agreement_list_string.split("|")
            unification_possible = is_unification_possible(appositive_np_label, np_agreement_list)

        if unification_possible:
            # NP;id=0;gfunc=SUBJ;<agreements>;coref
            splitted_appositive_np_labels = appositive_np_label.split(";")
            if len(splitted_appositive_np_labels) != 5:
                splitted_appositive_np_labels.append(f"coref={np_id}")
            elif splitted_appositive_np_labels[4] != f"coref={np_id}":
                splitted_appositive_np_labels[4] = "coref=-2" # Multi referring
            # else: ignore
            appositive_np.set_label(";".join(splitted_appositive_np_labels))

            referred = np
            break

    if referred is None:
        splitted_appositive_np_labels = appositive_np_label.split(";")
        if len(splitted_appositive_np_labels) != 5:
            splitted_appositive_np_labels.append(f"coref=-1")
        # else: ignore

        appositive_np.set_label(";".join(splitted_appositive_np_labels))

    extracted_tree = Tree(
        node=new_tree.label(),
        children=new_tree[:start_index] + [clause_tree] + new_tree[boundary_index:]
    )

    return (extracted_tree, start_index + 1)

def detect_restrictive_relative_clause_boundary(tree: Tree, start_index: int, predefined_stop_index: int | None = None, keep_structure=True) -> tuple[Tree, int]:
    """Detect relative clause boundary in restrictive case. Return -1 if it is not appositive.
    
    """
    
    logging.info(f"Start detect_restrictive_relative_clause_boundary")
    logging.info(f"Tree: {tree}")
    logging.info(f"Start pointer: index={start_index}, node={tree[start_index]}")
    logging.info(f"Stop pointer: {'-' if predefined_stop_index is None else f'index={predefined_stop_index}, node={tree[predefined_stop_index]}'}")


    if not (tree[start_index].label().startswith("PRON") and "PronType=Rel" in tree[start_index].label()):
        logging.info(f"End detect_restrictive_relative_clause_boundary (not starts with relative pronoun)")
        return (tree, -1)

    index = start_index + 1
    if predefined_stop_index is None:
        stop_index = len(tree)
    else:
        stop_index = predefined_stop_index

    # Step 1
    if index < len(tree) and tree[index, 0] == ",":
        index += 1
        while index < len(tree) and (not tree[index, 0] == ","):
            index += 1

        if index == len(tree):
            logging.info(f"End detect_restrictive_relative_clause_boundary (fail to satisfy step 1)")
            return (tree, -1)

    logging.info(f"Pointer after step 1: {index=}, node{tree[index]}")

    # Step 2
    verb_group_contains_saying_verb = False
    SAYING_VERB_WORDS = ["katakan", "kata", "mengatakan", "berkata", "dikatakan", "sampaikan", "menyampaikan", "disampaikan", "ucapkan", "ucap", "mengucapkan", "diucapkan"]

    # Find start of verb group
    while index < len(tree) and (not tree[index].label().startswith("VERB")):
        index += 1

    if index == len(tree):
        logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find verb group in step 2)")
        return (tree, -1)

    # Find end of verb group
    while index < len(tree) and (tree[index].label().startswith("VERB")):
        if tree[index].label().startswith("VERB") and tree[index, 0].lower() in SAYING_VERB_WORDS:
            verb_group_contains_saying_verb = True
        index += 1

    complementizer_encountered = False
    COMPLEMENTIZER_WORDS = ["bahwa", "agar", "sebab", "karena", "meskipun", "meski"]

    # Find noun group
    while index < len(tree) and (not tree[index].label().startswith("NP")):
        if tree[index, 0].lower() in COMPLEMENTIZER_WORDS:
            complementizer_encountered = True
        index += 1

    if index == len(tree):
        logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find noun group [NP])")
        return (tree, -1)

    # Pass noun group
    index += 1

    if index < len(tree):
        logging.info(f"Pointer after step 2: {index=}, node{tree[index]}")
    else:
        logging.info(f"Pointer after step 2: {index=}, EOS")

    # Step 3
    if complementizer_encountered or verb_group_contains_saying_verb:
        # Find start of verb group
        while index < len(tree) and (not tree[index].label().startswith("VERB")):
            index += 1

        # Find end of verb group
        while index < len(tree) and (tree[index].label().startswith("VERB")):
            index += 1

    if index < len(tree):
        logging.info(f"Pointer after step 3: {index=}, node{tree[index]}")
    else:
        logging.info(f"Pointer after step 3: {index=}, EOS")

    # Step 4
    logging.info(f"Step 4.c will be skipped because there is no equivalent of VBG and VBN in Indonesia language.")

    end_clause = False
    mismatch = False
    while index < len(tree) and (not end_clause) and (not mismatch):
        # Specific to step 4
        # Find next_index
        prev_index = index
        while index < len(tree) and not (tree[index, 0] in [",", ":", ";"] or tree[index].label().startswith("VERB") or (tree[index].label().startswith("PRON") and "PronType=Rel" in tree[index].label())):
            index += 1

        # Saya pergi ke tempat yang diminati orang dan yang diinginkan kakak saya pada hari ini.
        # Saya pergi ke tempat yang diminati orang dan diinginkan kakak saya pada hari ini.

        step_4_occured = False
        if index < len(tree):
            logging.debug(f"{index=}, node={tree[index]}")

            # Step 4.a
            if tree[index, 0] in [":", ";"] or index == stop_index:
                end_clause = True
                logging.info(f"End clause because of step 4.a.")
                step_4_occured = True

            if not step_4_occured:
                # Step 4.b
                if tree[index, 0] == "," and index < len(tree) - 1:
                    if keep_structure:
                        new_tree, boundary_index = detect_appositive_boundary(tree, index + 1, keep_structure=keep_structure)
                        if boundary_index != -1:
                            index = boundary_index
                            logging.info(f"Pointer after step 4.b: {index=}, node{tree[index]}")
                            step_4_occured = True

                    else:
                        new_tree, boundary_index = extract_appositive_structure(tree, index + 1)
                        if boundary_index != -1:
                            logging.info(f"Tree changed because of 4.b:\n{tree}")
                            tree = new_tree

                            index = boundary_index
                            logging.info(f"Pointer after step 4.b: {index=}, node{tree[index]}")
                            step_4_occured = True

            # Step 4.c (skipped, because there is no equivalent of VBG and VBN)
            
            if not step_4_occured:
                # Step 4.d (ADV is not tested yet.)
                if is_comma_for_implicit_conjunction_of_adjectives_or_adverbs(tree, index):
                    index += 2
                    logging.info(f"Pointer after step 4.d: {index=}, node{tree[index]}")
                    step_4_occured = True

            if not step_4_occured:
                if tree[index, 0] == ",":
                    delete_relpron_token = True
                    if index < len(tree) - 1 and tree[index + 1].label().startswith("PRON") and "PronType=Rel" in tree[index + 1].label():
                        delete_relpron_token = False
                    elif index < len(tree) - 2 and ("CCONJ" in tree[index + 1].label() or "SCONJ" in tree[index + 1].label()) and tree[index + 2].label().startswith("PRON") and "PronType=Rel" in tree[index + 2].label():
                        delete_relpron_token = True
                    else:
                        delete_relpron_token = None

                    if delete_relpron_token is not None:
                        logging.info(f"Step 4.e.i and 4.e.ii started.")
                        if keep_structure:
                            if delete_relpron_token:
                                new_tree, boundary_index = detect_nonrestrictive_relative_clause_boundary(tree, index + 2, keep_structure=keep_structure)
                            else:
                                new_tree, boundary_index = detect_nonrestrictive_relative_clause_boundary(tree, index + 1, keep_structure=keep_structure)

                            if boundary_index == -1:
                                logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find end of relative clause in step 4.e)")
                                mismatch = True

                            else:
                                index = boundary_index
                        
                        elif delete_relpron_token:
                            new_tree, boundary_index = detect_nonrestrictive_relative_clause_boundary(tree, index + 2, keep_structure=keep_structure)

                            if boundary_index == -1:
                                logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find end of relative clause in step 4.e)")
                                mismatch = True

                            else:
                                tree = new_tree
                                tree = Tree(tree.label(), tree[:index + 2] + tree[index + 3:])
                                logging.info(f"Tree changed because of 4.e.i:\n{tree}")
                                index = boundary_index - 1
                                stop_index -= 1

                        else:
                            new_tree, boundary_index = extract_nonrestrictive_relative_clause_structure(tree, index + 1)

                            if boundary_index == -1:
                                logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find end of relative clause in step 4.e)")
                                mismatch = True

                            else:
                                tree = new_tree
                                index = boundary_index
                        
                        if not mismatch:
                            logging.info(f"Pointer after step 4.e: {index=}, node{tree[index]}")
                            step_4_occured = True
                
                elif tree[index].label().startswith("PRON") and "PronType=Rel" in tree[index].label():
                    logging.info(f"Step 4.e.i and 4.e.iii started.")
                    delete_relpron_token = False
                    if index > 0 and ("CCONJ" in tree[index - 1].label() or "SCONJ" in tree[index - 1].label()):
                        delete_relpron_token = True

                    if keep_structure:
                        new_tree, boundary_index = detect_restrictive_relative_clause_boundary(tree, index, keep_structure=keep_structure)

                        if boundary_index == -1:
                            logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find end of relative clause in step 4.e)")
                            mismatch = True

                        else:
                            index = boundary_index

                    elif delete_relpron_token:
                        new_tree, boundary_index = detect_restrictive_relative_clause_boundary(tree, index)

                        if boundary_index == -1:
                            logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find end of relative clause in step 4.e)")
                            mismatch = True

                        else:
                            tree = new_tree
                            tree = Tree(tree.label(), tree[:index] + tree[index + 1:])
                            logging.info(f"Tree changed because of 4.e.i:\n{tree}")

                            index = boundary_index - 1
                            stop_index -= 1

                    else:
                        new_tree, boundary_index = extract_restrictive_relative_clause_structure(tree, index)

                        if boundary_index == -1:
                            logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find end of relative clause in step 4.e)")
                            mismatch = True

                        else:
                            tree = new_tree
                            logging.info(f"Tree changed:\n{tree}")

                            index = boundary_index

                    if not mismatch:
                        logging.info(f"Pointer after step 4.e: {index=}, node{tree[index]}")
                        step_4_occured = True

        if not step_4_occured:
            index = prev_index
            logging.debug(f"Step 4 not occured.")
            logging.debug(f"{index=}, node={tree[index]}")
            # Step 5
            if "CCONJ" in tree[index - 1].label() or "SCONJ" in tree[index - 1].label() or (tree[index - 1].label().startswith("NP") and tree[index - 1, 0].label().startswith("PRON") and "gfunc=SUBJ" in tree[index - 1].label()):
                # # Internal comma
                while index < len(tree) and tree[index, 0] != ",":
                    index += 1

                if index == len(tree):
                    logging.info(f"End detect_restrictive_relative_clause_boundary (fail to find comma in step 5)")
                    mismatch = True
                
                else:
                    index += 1
                    logging.info(f"Pointer after step 5: {index=}, node{tree[index]}")

            # Step 6
            else:
                end_clause = True
                logging.info(f"End clause because of step 6.")

    if mismatch:
        return (tree, -1)

    logging.info(f"End detect_restrictive_relative_clause_boundary")
    
    return (tree, index)

def detect_nonrestrictive_relative_clause_boundary(tree: Tree, start_index: int, keep_structure=True):
    """Detect relative clause boundary in nonrestrictive case. Return -1 if it is not relative clause.
    
    """
    
    logging.info(f"Start detect_restrictive_relative_clause_boundary")
    logging.info(f"Tree: {tree}")
    logging.info(f"Start pointer: index={start_index}, node={tree[start_index]}")

    if not (tree[start_index].label().startswith("PRON") and "PronType=Rel" in tree[start_index].label()):
        logging.info(f"End detect_nonrestrictive_relative_clause_boundary (not starts with relative pronoun)")
        return (tree, -1)
    
    # Check comma existence
    index = start_index + 1
    while index < len(tree) and tree[index, 0] != "." and tree[index, 0] != ",":
        index += 1

    if index != len(tree) and tree[index, 0] != ".":
        if index == start_index + 1:
            # Jump to next comma
            index += 1
            while index < len(tree) and tree[index, 0] == ",":
                index += 1

            if index == len(tree):
                logging.info(f"End detect_nonrestrictive_relative_clause_boundary: fail to find next comma in step 4.")
                return (tree, -1)
            
            index += 1
        
        logging.info(f"Pointer after step 4: {index=}, node{tree[index]}")

        # Step 5
        logging.info(f"Step 5.b will be skipped because there is no equivalent of VBG and VBN in Indonesia language.")

        end_clause = False
        while index < len(tree) and (not end_clause):
            logging.debug(f"{index=}")

            # Specific to step 5
            prev_index = index
            while index < len(tree) and tree[index, 0] != ",":
                index += 1

            step_5_occured = False
            if index < len(tree):
                # Step 5.a
                if tree[index, 0] == "," and index < len(tree) - 1:
                    if keep_structure:
                        _, boundary_index = detect_appositive_boundary(tree, index + 1, keep_structure=keep_structure)
                        if boundary_index != -1:
                            index = boundary_index
                            logging.info(f"Pointer after step 5.a: {index=}, node{tree[index]}")
                            step_5_occured = True

                    else:
                        new_tree, boundary_index = extract_appositive_structure(tree, index + 1)
                        if boundary_index != -1:
                            tree = new_tree
                            logging.info(f"Tree changed because of 5.a:\n{tree}")

                            index = boundary_index
                            logging.info(f"Pointer after step 5.a: {index=}, node{tree[index]}")
                            step_5_occured = True

                # Step 5.b (skipped, because there is no equivalent of VBG and VBN)

                # Step 5.c (ADV is not tested yet.)
                if not step_5_occured:
                    if is_comma_for_implicit_conjunction_of_adjectives_or_adverbs(tree, index):
                        index += 2
                        logging.info(f"Pointer after step 5.c: {index=}, node{tree[index]}")
                        step_5_occured = True

                # Step 5.d
                if not step_5_occured:
                    if tree[index, 0] == ",":
                        delete_current_token = True
                        if index < len(tree) - 1 and tree[index + 1].label().startswith("PRON") and "PronType=Rel" in tree[index + 1].label():
                            delete_current_token = False
                        elif index < len(tree) - 2 and ("CCONJ" in tree[index + 1].label() or "SCONJ" in tree[index + 1].label()) and tree[index + 2].label().startswith("PRON") and "PronType=Rel" in tree[index + 2].label():
                            delete_current_token = True
                        else:
                            delete_current_token = None

                        if delete_current_token is not None:
                            logging.info(f"Step 5.d started.")
                            if delete_current_token and (not keep_structure):
                                tree = Tree(tree.label(), tree[:index + 2] + tree[index + 3:])
                                logging.info(f"Tree changed because of 4.e.i:\n{tree}")
                            
                            index += 1
                            logging.info(f"Pointer after step 4.e: {index=}, node{tree[index]}")
                            step_5_occured = True

            if not step_5_occured:
                # index = prev_index

                # Step 6
                end_clause = True
                logging.info(f"End clause because of step 6.")

        logging.info(f"End detect_nonrestrictive_relative_clause_boundary")
        return (tree, index)

    else:
        logging.info(f"End detect_nonrestrictive_relative_clause_boundary (no further comma) at end of sentence or dot in step 2.")
        return (tree, index)

def is_comma_for_implicit_conjunction_of_adjectives_or_adverbs(tree, index):
    return index < len(tree) - 1 and tree[index, 0] == "," and ((tree[index - 1].label().startswith("ADJ") and tree[index + 1].label().startswith("ADJ")) or (tree[index - 1].label().startswith("ADV") and tree[index + 1].label().startswith("ADV")))

def detect_appositive_boundary(tree: Tree, start_index: int, keep_structure=True):
    """Detect appostive boundary. Return -1 if it is not appositive.
    
    """
    logging.info(f"Start detect_appositive_boundary")
    logging.info(f"Tree: {tree}")
    logging.info(f"Start pointer: index={start_index}, node={tree[start_index]}")

    index = start_index
    if index == 0 or tree[index - 1, 0] != ",":
        logging.info(f"End detect_appositive_boundary (expect previous token is comma)")
        return (tree, -1)
    
    elif not tree[index].label().startswith("NP"):
        logging.info(f"End detect_appositive_boundary (expect NP as first element)")
        return (tree, -1)
    
    index += 1
    finding_repeated_prep_phrases = True
    mismatch = False
    while index < len(tree) and finding_repeated_prep_phrases:
        if tree[index].label().startswith("ADP"):
            if index < len(tree) - 1 and tree[index + 1].label().startswith("NP"):
                index += 2
            else:
                logging.info(f"End detect_appositive_boundary (expect NP after ADP)")
                mismatch = True
                finding_repeated_prep_phrases = False

        elif tree[index].label().startswith("PRON") and "PronType=Rel" in tree[index].label():
            finding_repeated_prep_phrases = False
        
        elif tree[index, 0] in [",", "."]:
            finding_repeated_prep_phrases = False

        else:
            mismatch = True
            finding_repeated_prep_phrases = False

    if mismatch:
        return (tree, -1)
    
    if tree[index].label().startswith("PRON") and "PronType=Rel" in tree[index].label():
        if keep_structure:
            new_tree, boundary_index = detect_restrictive_relative_clause_boundary(tree, index, keep_structure=keep_structure)
            if boundary_index != -1:
                index = boundary_index

        else:
            new_tree, boundary_index = extract_restrictive_relative_clause_structure(tree, index)
            if boundary_index != -1:
                tree = new_tree
                index = boundary_index

    if index == len(tree):
        logging.info(f"End detect_appositive_boundary")
        return (tree, index)
    
    elif tree[index, 0] == ".":
        logging.info(f"End detect_appositive_boundary")
        return (tree, index)
    
    elif tree[index, 0] == ",":
        new_tree, boundary_index = detect_appositive_boundary(tree, index + 1, keep_structure=keep_structure)
        logging.info(f"End detect_appositive_boundary")

        if boundary_index == -1:
            return (tree, index)
        elif keep_structure:
            return (tree, boundary_index)
        else:
            return (new_tree, boundary_index)
    
    else:
        logging.info(f"End detect_appositive_boundary")
        return (tree, -1)

MASCULINE_TITLES = ["tuan", "pangeran", "raja", "kaisar", "bapak", "pak"]
FEMININE_TITLES = ["nyonya", "nona", "ratu", "permaisuri", "ibu", "bu"]
COMMON_TITLES = ["adipati", "pendeta", "dokter", "doktor", "profesor", "menteri", "sekretaris", "presiden"]
COMPANY_KEYWORDS = ["perusahaan", "pt", "tbk"]

def extract_agreements(tree_list: list[Tree]) -> list[Tree]:
    # Ambil semua fitur dari head noun.
    result = extract_agreements_from_head_noun(tree_list)

    # # Step 1: Check gender
    # result = extract_gender_and_animacy_agreements_from_keywords(result)

    # # Step 2: Do coreference
    # result = extract_gender_and_animacy_agreements_from_coreference_with_same_head_noun(result)

    # # Step 3: Find from Wordnet
    # result = extract_gender_and_animacy_agreements_from_wordnet(result)

    # # Step 4: Check appositive and copula patterns
    # result = extract_gender_and_animacy_agreements_from_copula(result)

    return result

def extract_gender_and_animacy_agreements_from_wordnet(tree_list: list[Tree]) -> list[Tree]:
    person_matcher = re.compile(r"[0-9]+-n\tind:def\t([^\s]* \()?((se)?se)?orang")
    animal_matcher = re.compile(r"[0-9]+-n\tind:def\t(binatang|hewan)")
    company_matcher = re.compile(r"[0-9]+-n\tind:def\t(perusahaan|lembaga)")

    new_tree_list: list[Tree] = []
    for tree in tree_list:
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree in subtrees:
            logging.debug(f"Checking {subtree} for step 3 condition")
            if subtree.label().startswith("NP") and "Gender" not in subtree.label():
                head_noun = get_head_noun(subtree)
                head_noun_text_lower = head_noun[0].lower()
                logging.debug(f"{head_noun_text_lower=}")
                np_agreements_string = subtree.label().split(";")[2]
                if np_agreements_string == "":
                    np_agreements = []
                else:
                    np_agreements = np_agreements_string.split("|")

                found = False
                with open("wn-msa-tab-r24-trunk/wn-msa-all.tab", mode="r", encoding="utf-8") as f_lemma:
                    for lemma_line in f_lemma:
                        lemmas = lemma_line.strip().split("\t")[3].split(" ")
                        if head_noun_text_lower in lemmas:
                            logging.debug(f"{lemmas=}")
                            code, *_ = lemma_line.lower().split("\t")
                            with open("wn-msa-tab-r24-trunk/wn-ind-def.tab", mode="r") as f_def:
                                for def_line in f_def:
                                    def_line = def_line.lower()
                                    if def_line.startswith(code):
                                        if person_matcher.match(def_line):
                                            logging.debug(def_line)
                                            np_agreements.append("Animacy=Anim")
                                            if "binatang" not in def_line and "hewan" not in def_line:
                                                logging.debug("Marked as person")
                                                np_agreements.append("Gender=Com")
                                            else:
                                                logging.debug("Marked as animal")
                                            found = True

                                        elif animal_matcher.match(def_line):
                                            logging.debug(def_line)
                                            logging.debug("Marked as animal")
                                            np_agreements.append("Animacy=Anim")
                                            found = True

                                        elif company_matcher.match(def_line):
                                            logging.debug(def_line)
                                            logging.debug("Marked as company")
                                            np_agreements.append("Animacy=Inan")
                                            np_agreements.append("Gender=Neut")
                                            found = True

                                        # else: ignore
                                        
                                        break

                            if found:
                                break

                np_agreements_string = "|".join(np_agreements)

                subtree_labels = subtree.label().split(";")
                subtree_labels[2] = np_agreements_string
                subtree.set_label(";".join(subtree_labels))

        new_tree_list.append(Tree(tree.label(), subtrees))

    result = new_tree_list
    return result

def extract_gender_and_animacy_agreements_from_copula(tree_list: list[Tree]) -> list[Tree]:
    logging.info(f"Step 4")
    new_tree_list: list[Tree] = []
    for tree in tree_list:
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree_1, subtree_2, subtree_3 in trigrams(subtrees):
            logging.debug(f"Checking {subtree_1=}")
            if not subtree_1.label().startswith("NP"):
                logging.debug(f"subtree_1 is not NP")
                continue

            if "Gender" in subtree_1.label() or "Animacy" in subtree_1.label():
                logging.debug(f"subtree_1 has gender and/or animacy agreements")
                continue

            logging.debug(f"Checking {subtree_2=}")
            if " ".join(subtree_2.leaves()) not in ["adalah", "merupakan"]:
                logging.debug(f"subtree_2 is not in copula keywords")
                continue

            logging.debug(f"Checking {subtree_3=}")
            if not subtree_3.label().startswith("NP"):
                logging.debug(f"subtree_3 is not NP")
                continue

            if not ("Gender" in subtree_3.label() or "Animacy" in subtree_3.label()):
                logging.debug(f"subtree_3 has not gender and/or animacy agreements.")
                continue

            target_labels = subtree_1.label().split(";")
            target_agreements_string = target_labels[2]
            if target_agreements_string == "":
                np_agreements = []
            else:
                np_agreements = target_agreements_string.split("|")

            other_agreements_string = subtree_3.label().split(";")[2]
            if other_agreements_string == "":
                other_agreements = []
            else:
                other_agreements = other_agreements_string.split("|")

            for agreement in other_agreements:
                if agreement.startswith("Gender") or agreement.startswith("Animacy"):
                    np_agreements.append(agreement)

            np_agreements_string = "|".join(np_agreements)

            target_labels[2] = np_agreements_string
            subtree_1.set_label(";".join(target_labels))

        new_tree_list.append(Tree(tree.label(), subtrees))

    result = new_tree_list
    return result

def extract_agreements_from_head_noun(tree_list: list[Tree]) -> list[Tree]:
    new_tree_list: list[Tree] = []
    for tree in tree_list:
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree in subtrees:
            if subtree.label().startswith("NP"):
                head_noun = get_head_noun(subtree)
                head_noun_feats = head_noun.label().split(";")[1]
                subtree.set_label(subtree.label() + f";{head_noun_feats}")

            # else: no modification

        new_tree_list.append(Tree(tree.label(), subtrees))

    return new_tree_list

def extract_gender_and_animacy_agreements_from_keywords(tree_list: list[Tree]) -> list[Tree]:
    raise NotImplementedError("Fix the agreement location first.")
    new_tree_list: list[Tree] = []
    for tree in tree_list:
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree in subtrees:
            if subtree.label().startswith("NP"):
                np_agreements_string = subtree.label().split(";")[2]
                if np_agreements_string == "":
                    np_agreements = []
                else:
                    np_agreements = np_agreements_string.split("|")
                
                word_index = 0
                while word_index < len(subtree) and not any([x.startswith("Gender") for x in np_agreements]):
                    word_lower_text = subtree[word_index, 0].lower()
                    if word_lower_text in MASCULINE_TITLES:
                        np_agreements.append("Gender=Masc")
                        np_agreements.append("Animacy=Anim")
                    elif word_lower_text in FEMININE_TITLES:
                        np_agreements.append("Gender=Fem")
                        np_agreements.append("Animacy=Anim")
                    elif word_lower_text in COMMON_TITLES:
                        np_agreements.append("Gender=Com")
                        np_agreements.append("Animacy=Anim")
                    elif word_lower_text in COMPANY_KEYWORDS:
                        np_agreements.append("Gender=Neut")
                        np_agreements.append("Animacy=Inan")

                    word_index += 1

                np_agreements_string = "|".join(np_agreements)

                subtree_labels = subtree.label().split(";")
                subtree_labels[2] = np_agreements_string
                subtree.set_label(";".join(subtree_labels))

            # else: no modification

        new_tree_list.append(Tree(tree.label(), subtrees))

    return new_tree_list

def extract_gender_and_animacy_agreements_from_coreference_with_same_head_noun(tree_list: list[Tree]) -> list[Tree]:
    raise NotImplementedError("Fix the agreement location first.")
    logging.debug(f"Forward search for person in step 2")
    result = extract_gender_and_animacy_agreements_from_person_coreference_with_same_head_noun(tree_list)
    
    logging.debug(f"Backward search for company in step 2")
    result = extract_gender_and_animacy_agreements_from_company_coreference_with_same_head_noun(tree_list, result)
    return result

def extract_gender_and_animacy_agreements_from_company_coreference_with_same_head_noun(tree_list, result):
    new_tree_list: list[Tree] = []
    for tree_index, tree in enumerate(result):
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree_index, subtree in enumerate(subtrees):
            logging.debug(f"Checking {subtree} for step 2 condition")
            if subtree.label().startswith("NP") and "Gender" not in subtree.label():
                logging.debug(f"Entering step 2 for {subtree}")
                np_agreements_string = subtree.label().split(";")[2]
                if np_agreements_string == "":
                    np_agreements = []
                else:
                    np_agreements = np_agreements_string.split("|")

                target_head_noun = get_head_noun(subtree)
                logging.debug(f"{target_head_noun=}")
                
                # Target: company
                # Check inside sentence.
                other_subtree_index = subtree_index - 1
                while (not any([agreement.startswith("Animacy") for agreement in np_agreements])) and other_subtree_index >= 0:
                    other_subtree = tree[other_subtree_index]
                    other_subtree_label = other_subtree.label()
                    logging.debug(f"{other_subtree=}")
                    logging.debug(f"{other_subtree_label=}")
                    logging.debug(f"Condition 1: {other_subtree_label.startswith('NP')}")
                    logging.debug(f"Condition 2: {'Animacy=Inan' in other_subtree_label}")
                    if other_subtree_label.startswith("NP") and "Animacy=Inan" in other_subtree_label:
                        other_head_noun = get_head_noun(other_subtree)
                        logging.debug(f"{other_head_noun=}")
                        if other_head_noun[0].lower() == target_head_noun[0].lower():
                            other_subtree_agreements = other_subtree_label.split(";")[2]
                            for agreement in other_subtree_agreements.split("|"):
                                if agreement.startswith("Gender") or agreement.startswith("Animacy"):
                                    np_agreements.append(agreement)
                        else:
                            other_subtree_index -= 1
                    else:
                        other_subtree_index -= 1

                logging.debug(f"Check previous sentences")

                # Check previous sentences
                other_tree_index = tree_index - 1
                while (not any([agreement.startswith("Animacy") for agreement in np_agreements])) and other_tree_index >= 0:
                    other_tree = tree_list[other_tree_index]
                    other_subtree_index = len(other_tree) - 1
                    while (not any([agreement.startswith("Animacy") for agreement in np_agreements])) and other_subtree_index >= 0:
                        other_subtree = other_tree[other_subtree_index]
                        other_subtree_label = other_subtree.label()
                        logging.debug(f"{other_subtree=}")
                        logging.debug(f"{other_subtree_label=}")
                        logging.debug(f"Condition 1: {other_subtree_label.startswith('NP')}")
                        logging.debug(f"Condition 2: {'Animacy=Inan' in other_subtree_label}")
                        if other_subtree_label.startswith("NP") and "Animacy=Inan" in other_subtree_label:
                            other_head_noun = get_head_noun(other_subtree)
                            logging.debug(f"{other_head_noun=}")
                            if other_head_noun[0].lower() == target_head_noun[0].lower():
                                other_subtree_agreements = other_subtree_label.split(";")[2]
                                for agreement in other_subtree_agreements.split("|"):
                                    if agreement.startswith("Gender") or agreement.startswith("Animacy"):
                                        np_agreements.append(agreement)
                            else:
                                other_subtree_index -= 1
                        else:
                            other_subtree_index -= 1

                    if (not any([agreement.startswith("Animacy") for agreement in np_agreements])):
                        other_tree_index -= 1
                
                np_agreements_string = "|".join(np_agreements)

                subtree_labels = subtree.label().split(";")
                subtree_labels[2] = np_agreements_string
                subtree.set_label(";".join(subtree_labels))

            # else: no modification

        new_tree_list.append(Tree(tree.label(), subtrees))

    result = new_tree_list
    return result

def extract_gender_and_animacy_agreements_from_person_coreference_with_same_head_noun(tree_list: list[Tree]) -> list[Tree]:
    new_tree_list: list[Tree] = []
    for tree_index, tree in enumerate(tree_list):
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree_index, subtree in enumerate(subtrees):
            logging.debug(f"Checking {subtree} for step 2 condition")
            if subtree.label().startswith("NP") and "Gender" not in subtree.label():
                logging.debug(f"Entering step 2 for {subtree}")
                np_agreements_string = subtree.label().split(";")[2]
                if np_agreements_string == "":
                    np_agreements = []
                else:
                    np_agreements = np_agreements_string.split("|")

                target_head_noun = get_head_noun(subtree)
                logging.debug(f"{target_head_noun=}")
                
                # Target: person
                # Check inside sentence.
                other_subtree_index = subtree_index + 1
                while (not any([agreement.startswith("Animacy") for agreement in np_agreements])) and other_subtree_index < len(tree):
                    other_subtree = tree[other_subtree_index]
                    other_subtree_label = other_subtree.label()
                    logging.debug(f"{other_subtree=}")
                    logging.debug(f"{other_subtree_label=}")
                    logging.debug(f"Condition 1: {other_subtree_label.startswith('NP')}")
                    logging.debug(f"Condition 2: {'Animacy=Anim' in other_subtree_label}")
                    if other_subtree_label.startswith("NP") and "Animacy=Anim" in other_subtree_label:
                        other_head_noun = get_head_noun(other_subtree)
                        logging.debug(f"{other_head_noun=}")
                        if other_head_noun[0].lower() == target_head_noun[0].lower():
                            other_subtree_agreements = other_subtree_label.split(";")[2]
                            for agreement in other_subtree_agreements.split("|"):
                                if agreement.startswith("Gender") or agreement.startswith("Animacy"):
                                    np_agreements.append(agreement)
                        else:
                            other_subtree_index += 1
                    else:
                        other_subtree_index += 1

                logging.debug(f"Check next sentences")

                # Check next sentences
                other_tree_index = tree_index + 1
                while (not any([agreement.startswith("Animacy") for agreement in np_agreements])) and other_tree_index < len(tree_list):
                    other_tree = tree_list[other_tree_index]
                    other_subtree_index = 0
                    while (not any([agreement.startswith("Animacy") for agreement in np_agreements])) and other_subtree_index < len(other_tree):
                        other_subtree = other_tree[other_subtree_index]
                        other_subtree_label = other_subtree.label()
                        logging.debug(f"{other_subtree=}")
                        logging.debug(f"{other_subtree_label=}")
                        logging.debug(f"Condition 1: {other_subtree_label.startswith('NP')}")
                        logging.debug(f"Condition 2: {'Animacy=Anim' in other_subtree_label}")
                        if other_subtree_label.startswith("NP") and "Animacy=Anim" in other_subtree_label:
                            other_head_noun = get_head_noun(other_subtree)
                            logging.debug(f"{other_head_noun=}")
                            if other_head_noun[0].lower() == target_head_noun[0].lower():
                                other_subtree_agreements = other_subtree_label.split(";")[2]
                                for agreement in other_subtree_agreements.split("|"):
                                    if agreement.startswith("Gender") or agreement.startswith("Animacy"):
                                        np_agreements.append(agreement)
                            else:
                                other_subtree_index += 1
                        else:
                            other_subtree_index += 1

                    if (not any([agreement.startswith("Animacy") for agreement in np_agreements])):
                        other_tree_index += 1
                
                np_agreements_string = "|".join(np_agreements)

                subtree_labels = subtree.label().split(";")
                subtree_labels[2] = np_agreements_string
                subtree.set_label(";".join(subtree_labels))

            # else: no modification

        new_tree_list.append(Tree(tree.label(), subtrees))

    return new_tree_list

def get_head_noun(subtree: Tree) -> Tree:
    head_noun = None
    possibly_title_head_noun = None
    for word in subtree:
        if any([word.label().startswith(noun_tag) for noun_tag in ["NOUN", "PROPN", "PRON"]]):
            if (possibly_title_head_noun is None) and word[0].lower() in (MASCULINE_TITLES + FEMININE_TITLES + COMMON_TITLES + COMPANY_KEYWORDS):
                possibly_title_head_noun = word
            else:
                head_noun = word
                break

    if head_noun is None:
        if possibly_title_head_noun is not None:
            head_noun = possibly_title_head_noun
        else:
            raise ValueError("Invalid noun phrase: no noun exists")
    
    return head_noun

def resolve_third_person_pronouns(tree_list: list[Tree]) -> list[Tree]:
    logging.info("Resolve third person pronouns start.")

    new_tree_list: list[Tree] = []
    coref_classes = {}

    for tree in tree_list:
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree_index, subtree in enumerate(subtrees):
            logging.debug(f"Checking {subtree=}")
            subtree_label = subtree.label()
            if subtree_label.startswith("NP"):
                if "PronType=Prs" in subtree_label:
                    # Input: subtree_index
                    # Input/output: coref_classes, subtrees
                    if "Person=3" in subtree_label or "Person=" not in subtree_label:
                        found = False
                        for coref_class_id, coref_class_details in sorted(coref_classes.items(), key=lambda x: x[1]["salience"], reverse=True):
                            # Agreement filtering
                            unification_possible = True
                            np = coref_class_details["tree"]
                            _, np_id_string, np_gfunc_string, np_agreement_list_string = np.label().split(";")

                            logging.debug(f"Agreement filtering: {np_agreement_list_string=}")
                            if np_agreement_list_string != "":
                                np_agreement_list = np_agreement_list_string.split("|")
                                unification_possible = is_unification_possible(subtree_label, np_agreement_list)

                            if not unification_possible:
                                continue

                            # Syntax filtering
                            _, _, subtree_gfunc_string, _ = subtree_label.split(";")
                            _, subtree_gfunc = subtree_gfunc_string.split("=")
                            _, np_gfunc = np_gfunc_string.split("=")

                            logging.debug(f"Syntax filtering 1")
                            if not is_satisfy_reflex_syntax_filtering(subtrees, subtree_index, np):
                                continue # You can't refer to it.

                            logging.debug(f"Syntax filtering 2")
                            if not is_satisfy_iobj_obliq_syntax_filtering(subtrees, subtree_index, np):
                                continue # You can't refer to it.

                            # Continue with other syntax filtering
                            logging.debug(f"Syntax filtering 3")
                            if subtree_gfunc == "DOBJ" and np_gfunc == "SUBJ":
                                # Find subject of subtree
                                subtree_subject_index = subtree_index - 1
                                subtree_subject_label = ""
                                while subtree_subject_index >= 0:
                                    subtree_subject_label = subtrees[subtree_subject_index].label()
                                    if subtree_subject_label.startswith("NP") and "gfunc=SUBJ" in subtree_subject_label:
                                        break
                                    else:
                                        subtree_subject_index -= 1


                                if subtree_subject_index >= 0:
                                    logging.debug(f"Subtree subject: {subtrees[subtree_subject_index]}")
                                    if subtrees[subtree_subject_index] == np:
                                        continue # You can't refer to it.

                            subtree.set_label(f"{subtree_label};coref={coref_class_id}")
                            salience = get_salience(subtree_label)
                            coref_class_details["salience"] += salience
                            found = True
                            break

                        if not found:
                            logging.debug(f"Not found; set coref=-1")
                            subtree.set_label(f"{subtree_label};coref=-1")

                    # else: do nothing
                else:
                    _, np_id_string, *_ = subtree_label.split(";")
                    _, np_id = np_id_string.split("=")
                    np_id = int(np_id)

                    salience = get_salience(subtree_label)
                    # else: no addition

                    coref_classes[np_id] = {
                        "salience": salience,
                        "members": [],
                        "tree": subtree
                    }

        new_tree_list.append(Tree(tree.label(), subtrees))

        # Update saliences
        for np_id in coref_classes.keys():
            coref_classes[np_id]["salience"] /= 2

    return new_tree_list

def is_satisfy_iobj_obliq_syntax_filtering(subtrees, subtree_index, np):
    subtree_label = subtrees[subtree_index].label()
    _, _, subtree_gfunc_string, _ = subtree_label.split(";")
    _, subtree_gfunc = subtree_gfunc_string.split("=")

    _, _, np_gfunc_string, _ = np.label().split(";")
    _, np_gfunc = np_gfunc_string.split("=")

    satisfied = True
    if subtree_gfunc in ["OBLIQ", "IOBJ"] and np_gfunc == "DOBJ":
        # Find subject of subtree
        subtree_subject_index = subtree_index - 1
        subtree_subject_label = ""
        while subtree_subject_index >= 0:
            subtree_subject_label = subtrees[subtree_subject_index].label()
            if subtree_subject_label.startswith("NP") and "gfunc=SUBJ" in subtree_subject_label:
                break
            else:
                subtree_subject_index -= 1

        if subtree_subject_index >= 0:
            # Find NP index
            np_index = subtree_subject_index + 1
            while np_index < subtree_index and subtrees[np_index] != np:
                np_index += 1

            if np_index < subtree_index:
                # Find NP subject index
                np_subject_index = np_index - 1
                while np_subject_index > subtree_subject_index:
                    np_subject_label = subtrees[np_subject_index].label()
                    if np_subject_label.startswith("NP") and "gfunc=SUBJ" in np_subject_label:
                        break
                    else:
                        np_subject_index -= 1

                if np_subject_index == subtree_subject_index:
                    satisfied = False

    return satisfied

def is_satisfy_reflex_syntax_filtering(subtrees: list[Tree], subtree_index: int, np: Tree):
    subtree_label: str = subtrees[subtree_index].label()
    if "Reflex=Yes" not in subtree_label:
        return True

    satisfied = True
    # Find subject of subtree or find NP
    subtree_subject_index = subtree_index - 1
    subtree_subject_label = ""
    while subtree_subject_index >= 0 and subtrees[subtree_subject_index] != np:
        subtree_subject_label = subtrees[subtree_subject_index].label()
        if subtree_subject_label.startswith("NP") and "gfunc=SUBJ" in subtree_subject_label:
            break
        else:
            subtree_subject_index -= 1

    # subtree_subject_index < 0, subtrees[subtree_subject_index] == np, or if subtree_subject_label.startswith("NP") and "gfunc=SUBJ" in subtree_subject_label

    if subtree_subject_label.startswith("NP") and "gfunc=SUBJ" in subtree_subject_label:
        if subtrees[subtree_subject_index] != np:
            satisfied = False

    return satisfied

def is_unification_possible(subtree_label, np_agreement_list):
    unification_possible = True
    for np_agreement in np_agreement_list:
        np_agreement_type, _ = np_agreement.split("=")
        logging.debug(f"Checking {np_agreement=}")
        if f"{np_agreement_type}=" in subtree_label:
            if np_agreement not in subtree_label:
                logging.debug(f"{np_agreement=} conflicts with {subtree_label=}")
                unification_possible = False

        if not unification_possible:
            break

    return unification_possible

def relative_clause_attachment(tree_list: list[Tree]) -> list[Tree]:
    logging.info("Relative clause attachment start.")

    new_tree_list: list[Tree] = []
    coref_classes = {}

    for tree in tree_list:
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree_index, subtree in enumerate(subtrees):
            logging.debug(f"Checking {subtree=}")
            subtree_label = subtree.label()
            if subtree_label.startswith("NP"):
                if "PronType=Prs" not in subtree_label: # "PronType=Rel" in subtree_label
                    _, np_id_string, *_ = subtree_label.split(";")
                    _, np_id = np_id_string.split("=")
                    np_id = int(np_id)

                    salience = get_salience(subtree_label)
                    # else: no addition

                    coref_classes[np_id] = {
                        "salience": salience,
                        "members": [],
                        "tree": subtree
                    }

                # else: Salience: include third pronoun or not?
                # Let's assume we don't first

            elif "PronType=Rel" in subtree_label:
                found = False
                for coref_class_id, coref_class_details in sorted(coref_classes.items(), key=lambda x: x[1]["salience"], reverse=True):
                    np = coref_class_details["tree"]
                    _, np_id_string, _, _ = np.label().split(";")

                    # Agreement filtering (skipped because ... no case)

                    # Syntax filtering
                    # Find the np_index first
                    np_index = subtree_index - 1
                    while np_index >= 0 and subtrees[np_index] != np:
                        np_index -= 1

                    # np_index < 0 | subtrees[np_index] == np
                    if np_index < 0:
                        continue # Out of sentence

                    np_index += 1
                    while np_index < subtree_index:
                        logging.debug(f"{np_index=}")
                        if subtrees[np_index].label().startswith("ADP") and np_index < len(subtrees) - 1:
                            if subtrees[np_index + 1].label().startswith("NP"):
                                np_index += 2
                            else:
                                break

                        elif subtrees[np_index][0] == "," and np_index < len(subtrees) - 3:
                            _, boundary_index = detect_appositive_boundary(tree, start_index=np_index + 1)
                            logging.debug(f"For relative clause attachment: {boundary_index=}")
                            if boundary_index == -1:
                                logging.debug("Nope, it is not appostive.")
                                break

                            if boundary_index >= subtree_index:
                                logging.debug("Passing subtree_index, not an expected appositive")
                                break

                            logging.debug("Appositive detected")
                            np_index = boundary_index + 1

                        else:
                            break

                    if np_index < subtree_index:
                        continue

                    subtree.set_label(f"{subtree_label};coref={coref_class_id}")
                    # salience = get_salience(subtree_label)
                    # coref_class_details["salience"] += salience
                    found = True
                    break

                if not found:
                    for coref_class_id, coref_class_details in sorted(coref_classes.items(), key=lambda x: x[1]["salience"], reverse=True):
                        np = coref_class_details["tree"]
                        _, np_id_string, _, _ = np.label().split(";")

                        # Agreement filtering (skipped because ... no case)

                        subtree.set_label(f"{subtree_label};coref={coref_class_id}")
                        # salience = get_salience(subtree_label)
                        # coref_class_details["salience"] += salience
                        found = True
                        break

                if not found:
                    logging.debug(f"Not found; set coref=-1")
                    subtree.set_label(f"{subtree_label};coref=-1")
        
        new_tree_list.append(Tree(tree.label(), subtrees))

        # Update saliences
        for np_id in coref_classes.keys():
            coref_classes[np_id]["salience"] /= 2

    return new_tree_list

def get_salience(subtree_label):
    salience = 100.0
    if "gfunc=SUBJ" in subtree_label:
        salience += 80.0 + 80.0
    elif "gfunc=DOBJ" in subtree_label:
        salience += 70.0 + 80.0
    elif "gfunc=IOBJ" in subtree_label:
        salience += 50.0 + 80.0
    elif "gfunc=OBLIQ" in subtree_label:
        salience += 50.0
    return salience

def extract_prefix_conjunctions(tree: Tree) -> Tree:
    if not tree[0].label().startswith("SCONJ"):
        return tree
    
    check_index = 1
    # If starts with comma, go to next comma
    if tree[check_index, 0] == ",":
        check_index += 1
        while check_index < len(tree) and tree[check_index, 0] != ",":
            check_index += 1

        if check_index == len(tree):
            return tree

    # Find NP, verb, and comma
    subject_exists = False
    verb_exists = False
    
    stop_clause = False
    while check_index < len(tree) and tree[check_index, 0] != "." and not stop_clause:
        if tree[check_index, 0] == "," and subject_exists and verb_exists:
            if is_comma_for_implicit_conjunction_of_adjectives_or_adverbs(tree, check_index):
                check_index += 2
            else:
                stop_clause = True
        else:
            if is_subject(tree[check_index]):
                subject_exists = True
            elif tree[check_index].label().startswith("VERB"):
                verb_exists = True

            check_index += 1

    # check_index == len(tree) or comma or period detected

    if check_index == len(tree) or tree[check_index, 0] == ".":
        return tree
    
    first_clause_stop_index = check_index
    first_clause_tree = Tree(
        node="CONJ-CLAUSE-1",
        children=tree[1:first_clause_stop_index]
    )

    check_index += 1
    if tree[check_index].label().startswith("SCONJ"):
        check_index += 1

    second_clause_start_index = check_index
    check_index = find_stop_index_for_second_conjoined_clause(tree, check_index)
        
    if check_index == -1:
        return tree
    
    second_clause_tree = Tree(
        node="CONJ-CLAUSE-2",
        children=tree[second_clause_start_index:check_index]
    )

    return Tree(
        node=tree.label(),
        children=[tree[0], first_clause_tree, *tree[first_clause_stop_index:second_clause_start_index], second_clause_tree, *tree[check_index:]]
    )

def find_stop_index_for_second_conjoined_clause(tree: Tree, start_index: int):
    check_index = start_index
    subject_exists = False
    verb_exists = False
    
    # Find period, verb, and comma
    while check_index < len(tree) and tree[check_index, 0] != ".":
        if is_subject(tree[check_index]):
            subject_exists = True
        elif tree[check_index].label().startswith("VERB"):
            verb_exists = True

        check_index += 1
    # check_index == len(tree) or comma or period detected

    if check_index == len(tree) or not (subject_exists and verb_exists):
        check_index = -1

    return check_index

def is_subject(np_tree: Tree):
    return np_tree.label().startswith("NP") and "gfunc=SUBJ" in np_tree.label()

def extract_infix_conjunctions(tree: Tree) -> Tree:
    # Find conjunction
    check_index = 0
    stop_clause = False
    subject_exists = False
    verb_exists = False

    while check_index < len(tree) and tree[check_index, 0] != "." and not stop_clause:
        if (tree[check_index].label().startswith("SCONJ") or tree[check_index].label().startswith("CCONJ")):
            if subject_exists and verb_exists:
                stop_clause = True
            else:
                check_index += 1
        else:
            if is_subject(tree[check_index]):
                subject_exists = True
            elif tree[check_index].label().startswith("VERB"):
                verb_exists = True
            check_index += 1

    if check_index == len(tree) or tree[check_index, 0] == ".":
        return tree
    
    conjunction_index = check_index
    first_clause_stop_index = conjunction_index
    if check_index > 0 and tree[check_index - 1, 0] == ",":
        first_clause_stop_index -= 1
    
    first_clause_tree = Tree(
        node="CONJ-CLAUSE-1",
        children=tree[:first_clause_stop_index]
    )

    check_index += 1
    second_clause_start_index = check_index
    check_index = find_stop_index_for_second_conjoined_clause(tree, check_index)
        
    if check_index == -1:
        return tree
    
    second_clause_tree = Tree(
        node="CONJ-CLAUSE-2",
        children=tree[second_clause_start_index:check_index]
    )

    return Tree(
        node=tree.label(),
        children=[first_clause_tree, *tree[first_clause_stop_index:conjunction_index], second_clause_tree, *tree[check_index:]]
    )

def extract_boundaries_for_single_tree(initial_tree: Tree, start_index: int) -> Tree:
    if len(initial_tree) == 0:
        return initial_tree

    tree = initial_tree
    subtree_index = start_index
    while subtree_index < len(tree):
        logging.debug(f"Checking {tree[subtree_index]}")

        if tree[subtree_index].label().startswith("NP"):
            if subtree_index > 0 and tree[subtree_index - 1, 0] == ",":
                new_tree, stop_index = extract_appositive_structure(tree, subtree_index)
                if stop_index != -1:
                    tree = new_tree
                    subtree_index = stop_index
                else:
                    subtree_index += 1

            else:
                subtree_index += 1

        elif "PronType=Rel" in tree[subtree_index].label():
            if subtree_index > 0 and tree[subtree_index - 1, 0] != ",":
                new_tree, stop_index = extract_restrictive_relative_clause_structure(tree, subtree_index)
                if stop_index != -1:
                    tree = new_tree
                    subtree_index = stop_index
                else:
                    subtree_index += 1
            else:
                new_tree, stop_index = extract_nonrestrictive_relative_clause_structure(tree, subtree_index)
                if stop_index != -1:
                    tree = new_tree
                    subtree_index = stop_index
                else:
                    subtree_index += 1

        else:
            subtree_index += 1
    
    # TODO: Do conjoined clauses
    tree = extract_prefix_conjunctions(tree)
    tree = extract_infix_conjunctions(tree)
    return tree

def extract_boundaries(tree_list: list[Tree]):
    logging.info("Relative clause boundary starts.")

    new_tree_list: list[Tree] = []
    for tree in tree_list:
        new_tree_list.append(extract_boundaries_for_single_tree(tree, start_index=0))

    return new_tree_list

if __name__ == "__main__":
    logging.basicConfig(filename='myapp.log', level=logging.DEBUG)

    stanza_pipeline = init_stanza_pipeline()
    document = input("Document: ")
    result = stanza_pipeline_document_process(stanza_pipeline, document)
    result = noun_chunk(result)
    result = extract_grammatical_function(result)

    result = extract_agreements(result)
    result = resolve_third_person_pronouns(result)
    result = relative_clause_attachment(result)
    result = extract_boundaries(result)

    # tree = result[0]
    # for index, subtree in enumerate(tree):
    #     if "PronType=Rel" in subtree.label():
    #         if tree[index - 1, 0] != ",":
    #             used_tree, stop_index = detect_restrictive_relative_clause_boundary(tree, index, keep_structure=False)
    #             if stop_index != -1:
    #                 print("Relative clause (restrictive):")
    #                 print(used_tree[index:stop_index])
    #                 Tree("REST-CLAUSE", used_tree[index:stop_index]).draw()
    #         else:
    #             used_tree, stop_index = detect_nonrestrictive_relative_clause_boundary(tree, index, keep_structure=False)
    #             if stop_index != -1:
    #                 print("Relative clause (nonrestrictive):")
    #                 print(used_tree[index:stop_index])
    #                 Tree("NONREST-CLAUSE", used_tree[index:stop_index]).draw()
    #         break

    # tree = result[0]
    # for index, subtree in enumerate(tree):
    #     if subtree[0] == ",":
    #         _, stop_index = detect_appositive_boundary(tree, index + 1)
    #         if stop_index != -1:
    #             print("Appositive:")
    #             print(tree[index + 1:stop_index])
    #             Tree("Appositive", tree[index + 1:stop_index]).draw()

    #         break

    print(result)
    Tree("Document", result).draw()
