import logging
from nltk import Tree
from .agreement import is_unification_possible
from .grammatical_function import is_subject

from .resolve_third_person_pronouns import get_salience

def extract_boundaries(tree_list: list[Tree], strategy=1):
    logging.info("Relative clause boundary starts.")

    new_tree_list: list[Tree] = []
    for tree in tree_list:
        new_tree_list.append(extract_boundaries_for_single_tree(tree, start_index=0, strategy=strategy))

    return new_tree_list

def extract_boundaries_for_single_tree(initial_tree: Tree, start_index: int, strategy=1) -> Tree:
    if len(initial_tree) == 0:
        return initial_tree

    tree = initial_tree
    subtree_index = start_index
    while subtree_index < len(tree):
        logging.debug(f"Checking {tree[subtree_index]}")

        if tree[subtree_index].label().startswith("NP"):
            if subtree_index > 0 and tree[subtree_index - 1, 0] == ",":
                new_tree, stop_index = extract_appositive_structure(tree, subtree_index, strategy=strategy)
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

    tree = extract_prefix_conjunctions(tree)
    tree = extract_infix_conjunctions(tree)
    return tree

def extract_appositive_structure(tree: Tree, start_index: int, strategy=1) -> tuple[Tree, int]:
    new_tree, boundary_index = detect_appositive_boundary(tree, start_index, keep_structure=False, strategy=strategy)
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
        _, np_id_string, _, np_agreement_list_string, *_ = np.label().split(";")
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

def detect_appositive_boundary(tree: Tree, start_index: int, keep_structure=True, strategy=1):
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
        
        elif strategy == 4 and tree[index].label().startswith("VERB"):
            finding_repeated_prep_phrases = False

        else:
            mismatch = True
            finding_repeated_prep_phrases = False

    if mismatch or index >= len(tree):
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
    
    elif strategy == 4 and tree[index].label().startswith("VERB"):
        logging.info(f"End detect_appositive_boundary")
        return (tree, index)
    
    elif tree[index, 0] == ",":
        new_tree, boundary_index = detect_appositive_boundary(tree, index + 1, keep_structure=keep_structure, strategy=strategy)
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

    logging.info(f"Pointer after step 1: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")

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
        logging.info(f"Pointer after step 2: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
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
        logging.info(f"Pointer after step 3: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
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
                            logging.info(f"Pointer after step 4.b: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
                            step_4_occured = True

                    else:
                        new_tree, boundary_index = extract_appositive_structure(tree, index + 1)
                        if boundary_index != -1:
                            logging.info(f"Tree changed because of 4.b:\n{tree}")
                            tree = new_tree

                            index = boundary_index
                            logging.info(f"Pointer after step 4.b: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
                            step_4_occured = True

            # Step 4.c (skipped, because there is no equivalent of VBG and VBN)
            
            if not step_4_occured:
                # Step 4.d (ADV is not tested yet.)
                if is_comma_for_implicit_conjunction_of_adjectives_or_adverbs(tree, index):
                    index += 2
                    logging.info(f"Pointer after step 4.d: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
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
                            logging.info(f"Pointer after step 4.e: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
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
                        logging.info(f"Pointer after step 4.e: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
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
                    logging.info(f"Pointer after step 5: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")

            # Step 6
            else:
                end_clause = True
                logging.info(f"End clause because of step 6.")

    if mismatch:
        return (tree, -1)

    logging.info(f"End detect_restrictive_relative_clause_boundary")
    
    return (tree, index)

def is_comma_for_implicit_conjunction_of_adjectives_or_adverbs(tree, index):
    return index < len(tree) - 1 and tree[index, 0] == "," and ((tree[index - 1].label().startswith("ADJ") and tree[index + 1].label().startswith("ADJ")) or (tree[index - 1].label().startswith("ADV") and tree[index + 1].label().startswith("ADV")))

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
        
        logging.info(f"Pointer after step 4: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")

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
                            logging.info(f"Pointer after step 5.a: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
                            step_5_occured = True

                    else:
                        new_tree, boundary_index = extract_appositive_structure(tree, index + 1)
                        if boundary_index != -1:
                            tree = new_tree
                            logging.info(f"Tree changed because of 5.a:\n{tree}")

                            index = boundary_index
                            logging.info(f"Pointer after step 5.a: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
                            step_5_occured = True

                # Step 5.b (skipped, because there is no equivalent of VBG and VBN)

                # Step 5.c (ADV is not tested yet.)
                if not step_5_occured:
                    if is_comma_for_implicit_conjunction_of_adjectives_or_adverbs(tree, index):
                        index += 2
                        logging.info(f"Pointer after step 5.c: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
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
                            logging.info(f"Pointer after step 4.e: {index=}, node{tree[index] if index < len(tree) else 'EOS'}")
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
    first_subject_exists = False
    first_verb_exists = False
    
    stop_clause = False
    while check_index < len(tree) and tree[check_index, 0] != "." and not stop_clause:
        if tree[check_index, 0] == "," and first_verb_exists and first_subject_exists: # BRUUUUUUH
            if is_comma_for_implicit_conjunction_of_adjectives_or_adverbs(tree, check_index):
                check_index += 2
            else:
                stop_clause = True
        else:
            if is_subject(tree[check_index]):
                first_subject_exists = True
            elif tree[check_index].label().startswith("VERB"):
                first_verb_exists = True

            check_index += 1

    # check_index == len(tree) or comma or period detected

    if check_index == len(tree) or tree[check_index, 0] == ".":
        return tree
    
    first_clause_stop_index = check_index

    check_index += 1
    if tree[check_index].label().startswith("SCONJ"):
        check_index += 1

    second_clause_start_index = check_index
    check_index, second_subject, second_verb = identify_second_conjoined_clause(tree, check_index)
        
    if check_index == -1 or not (second_subject is not None and second_verb is not None):
        return tree
    
    first_clause_tree_children = tree[1:first_clause_stop_index]
    if not first_subject_exists:
        first_clause_tree_children.insert(0, second_subject)

    first_clause_tree = Tree(
        node="CONJ-CLAUSE-1",
        children=first_clause_tree_children
    )
    
    second_clause_tree = Tree(
        node="CONJ-CLAUSE-2",
        children=tree[second_clause_start_index:check_index]
    )

    return Tree(
        node=tree.label(),
        children=[tree[0], first_clause_tree, *tree[first_clause_stop_index:second_clause_start_index], second_clause_tree, *tree[check_index:]]
    )

def identify_second_conjoined_clause(tree: Tree, start_index: int):
    check_index = start_index
    subject: Tree | None = None
    verb: Tree | None = None
    
    # Find period, verb, and comma
    while check_index < len(tree) and tree[check_index, 0] != ".":
        if is_subject(tree[check_index]):
            subject = tree[check_index]
        elif tree[check_index].label().startswith("VERB"):
            verb = tree[check_index]

        check_index += 1
    # check_index == len(tree) or comma or period detected

    if check_index == len(tree):
        check_index = -1

    return check_index, subject, verb

def extract_infix_conjunctions(tree: Tree) -> Tree:
    # Find conjunction
    check_index = 0
    stop_clause = False
    subject: Tree | None = None
    verb: Tree | None = None

    while check_index < len(tree) and tree[check_index, 0] != "." and not stop_clause:
        subtree = tree[check_index]
        if is_valid_conjunction(subtree):
            if subject is not None and verb is not None:
                stop_clause = True
            else:
                check_index += 1
        else:
            if is_subject(subtree):
                subject = subtree
            elif subtree.label().startswith("VERB"):
                verb = subtree
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
    check_index, subject_exists, verb_exists = identify_second_conjoined_clause(tree, check_index)
        
    if check_index == -1:
        return tree
    
    if subject_exists and (not verb_exists):
        return tree # Unhandled
    
    second_clause_tree_children = tree[second_clause_start_index:check_index]
    
    if not subject_exists:
        second_clause_tree_children.insert(0, subject)
        if not verb_exists:
            second_clause_tree_children.insert(1, verb)

    second_clause_tree = Tree(
        node="CONJ-CLAUSE-2",
        children=second_clause_tree_children
    )

    return Tree(
        node=tree.label(),
        children=[first_clause_tree, *tree[first_clause_stop_index:conjunction_index + 1], second_clause_tree, *tree[check_index:]]
    )

def is_valid_conjunction(tree: Tree):
    return (tree.label().startswith("SCONJ") or tree.label().startswith("CCONJ")) and tree[0] != "untuk"
