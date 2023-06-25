import logging
from nltk import Tree
from .agreement import is_unification_possible
from .grammatical_function import extract_gfunc_from_label

def init_third_person_pronouns_pipeline(strategy: int = 1):
    if strategy == 2:
        return resolve_third_person_pronouns_strategy_2
    else:
        return resolve_third_person_pronouns_strategy_1

def resolve_third_person_pronouns_strategy_1(tree_list: list[Tree]) -> list[Tree]:
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
                    add_to_valid_coreference_class_with_max_salience(
                        subtree_index=subtree_index,
                        coref_classes=coref_classes,
                        subtrees=subtrees
                    )
                else:
                    create_new_coreference_class(
                        subtree=subtree,
                        coref_classes=coref_classes
                    )

        new_tree_list.append(Tree(tree.label(), subtrees))

        # Update saliences
        for np_id in coref_classes.keys():
            coref_classes[np_id]["salience"] /= 2

    return new_tree_list

def add_to_valid_coreference_class_with_max_salience(subtree_index: int, coref_classes: dict, subtrees: list[Tree]):
    # Input: subtree_index
    # Input/output: coref_classes, subtrees

    subtree = subtrees[subtree_index]
    subtree_label = subtree.label()

    if "Person=3" in subtree_label or "Person=" not in subtree_label:
        found = False
        for coref_class_id, coref_class_details in sorted(coref_classes.items(), key=lambda x: x[1]["salience"], reverse=True):
            # Agreement filtering
            unification_possible = True
            np = coref_class_details["tree"]
            np_label = np.label()
            _, np_id_string, _, np_agreement_list_string = np_label.split(";")

            logging.debug(f"Agreement filtering: {np_agreement_list_string=}")
            if np_agreement_list_string != "":
                np_agreement_list = np_agreement_list_string.split("|")
                unification_possible = is_unification_possible(subtree_label, np_agreement_list)

            if not unification_possible:
                continue

            # Syntax filtering
            subtree_gfunc = extract_gfunc_from_label(subtree_label)
            np_gfunc = extract_gfunc_from_label(np_label)

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

def is_satisfy_iobj_obliq_syntax_filtering(subtrees, subtree_index, np):
    subtree_label = subtrees[subtree_index].label()
    subtree_gfunc = extract_gfunc_from_label(subtree_label)
    np_gfunc = extract_gfunc_from_label(np.label())

    if subtree_gfunc not in ["OBLIQ", "IOBJ"] or np_gfunc != "DOBJ":
        return True
    
    # Find subject of subtree
    subtree_subject_index = subtree_index - 1
    subtree_subject_label = ""
    while subtree_subject_index >= 0:
        subtree_subject_label = subtrees[subtree_subject_index].label()
        if subtree_subject_label.startswith("NP") and "gfunc=SUBJ" in subtree_subject_label:
            break
        else:
            subtree_subject_index -= 1

    if subtree_subject_index == -1:
        return True
    
    # Find NP index
    np_index = subtree_subject_index + 1
    while np_index < subtree_index and subtrees[np_index] != np:
        np_index += 1

    if np_index >= subtree_index:
        return True

    # Find NP subject index
    np_subject_index = np_index - 1
    while np_subject_index > subtree_subject_index:
        np_subject_label = subtrees[np_subject_index].label()
        if np_subject_label.startswith("NP") and "gfunc=SUBJ" in np_subject_label:
            break
        else:
            np_subject_index -= 1

    return np_subject_index != subtree_subject_index

def get_salience(subtree_label):
    salience = 100.0
    if "gfunc=SUBJ" in subtree_label:
        salience += 70.0 + 80.0
    elif "gfunc=DOBJ" in subtree_label:
        salience += 50.0 + 80.0
    elif "gfunc=IOBJ" in subtree_label:
        salience += 40.0 + 80.0
    elif "gfunc=OBLIQ" in subtree_label:
        salience += 40.0
    return salience

def create_new_coreference_class(subtree: Tree, coref_classes: dict[int, dict[str]]):
    subtree_label = subtree.label()

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

def resolve_third_person_pronouns_strategy_2(tree_list: list[Tree]) -> list[Tree]:
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
                    add_to_valid_coreference_class_with_max_salience(
                        subtree_index=subtree_index,
                        coref_classes=coref_classes,
                        subtrees=subtrees
                    )
                elif "PronType=Dem" in subtree[-1].label():
                    add_to_valid_coreference_class_with_max_salience(
                        subtree_index=subtree_index,
                        coref_classes=coref_classes,
                        subtrees=subtrees
                    )
                else:
                    create_new_coreference_class(
                        subtree=subtree,
                        coref_classes=coref_classes
                    )

        new_tree_list.append(Tree(tree.label(), subtrees))

        # Update saliences
        for np_id in coref_classes.keys():
            coref_classes[np_id]["salience"] /= 2

    return new_tree_list
