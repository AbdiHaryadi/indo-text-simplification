import logging
from nltk import Tree
from .clause_boundary import detect_appositive_boundary
from .resolve_third_person_pronouns import get_salience

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
                    _, np_id_string, *_ = np.label().split(";")

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
                        _, np_id_string, *_ = np.label().split(";")

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
