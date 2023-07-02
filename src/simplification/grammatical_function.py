from nltk import Tree

def extract_grammatical_function(tree_list: list[Tree], strategy=1):
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
            tree.label(),
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
            tree.label(),
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
            tree.label(),
            subtrees
        ))

    extracted_tree_list = new_extracted_tree_list

    # Extract pattern 5
    new_extracted_tree_list = []
    for tree in extracted_tree_list:
        start_index_list = [-1]
        subtrees = [subtree for subtree in tree]

        if strategy != 5:
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
            tree.label(),
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
            tree.label(),
            subtrees
        ))

    extracted_tree_list = new_extracted_tree_list

    return extracted_tree_list

def extract_gfunc_from_label(label: str):
    _, _, gfunc_string, _ = label.split(";")

    if gfunc_string == "":
        return ""
    else:
        _, gfunc = gfunc_string.split("=")
        return gfunc

def is_subject(np_tree: Tree):
    return np_tree.label().startswith("NP") and "gfunc=SUBJ" in np_tree.label()
