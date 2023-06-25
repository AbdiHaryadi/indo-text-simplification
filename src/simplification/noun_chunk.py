from nltk.tree import Tree
from nltk.chunk import RegexpChunkParser
from nltk.chunk.regexp import ChunkRule, StripRule, ExpandLeftRule, MergeRule, ExpandRightRule, SplitRule, UnChunkRule

def init_noun_chunk_pipeline(strategy=1):
    if strategy == 2 or strategy == 3:
        def noun_chunk_strategy_2(tree_list: list[Tree]):
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
                    StripRule(
                        r"<NUM;.*Year=Yes.*>",
                        "Exclude year from number modifier"
                    ),
                    MergeRule(
                        r"<NOUN;.*>",
                        r"<NUM;.*>",
                        "Merge NOUN and NUM"
                    ),
                    ExpandRightRule(
                        r"<NOUN;.*>",
                        r"<ADJ;.*>+",
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
                ],
                chunk_label="NP"
            )
            chunked_sentence_tree_list: list[Tree] = []

            for tree in tree_list:
                chunked_sentence_tree = parser.parse(tree)
                new_subtrees: list[Tree] = []
                subtree_index = 0
                while subtree_index < len(chunked_sentence_tree):
                    if subtree_index < len(chunked_sentence_tree) - 2 and chunked_sentence_tree[subtree_index].label().startswith("NP") and chunked_sentence_tree[subtree_index + 1].label().startswith("CCONJ") and chunked_sentence_tree[subtree_index + 2].label().startswith("NP"):
                        new_subtrees.append(Tree(
                            node=chunked_sentence_tree[subtree_index].label(),
                            children=chunked_sentence_tree[subtree_index, :] + [chunked_sentence_tree[subtree_index + 1]] + chunked_sentence_tree[subtree_index + 2, :]
                        ))
                        subtree_index += 3
                    else:
                        new_subtrees.append(chunked_sentence_tree[subtree_index])
                        subtree_index += 1

                chunked_sentence_tree_list.append(Tree(
                    node=chunked_sentence_tree.label(),
                    children=new_subtrees
                ))

            # Handle ()
            new_chunked_sentence_tree_list: list[Tree] = []
            for tree in chunked_sentence_tree_list:
                # Find (
                subtree_index = 0

                new_subtrees: list[Tree] = []
                while subtree_index < len(tree):
                    found = False
                    while (not found) and subtree_index < len(tree):
                        subtree = tree[subtree_index]
                        if subtree_index >= len(tree) - 1:
                            new_subtrees.append(subtree)
                            subtree_index += 1
                            continue

                        next_subtree = tree[subtree_index + 1]
                        if subtree.label().startswith("NP") and next_subtree.label().startswith("PUNCT") and next_subtree[0] == "(":
                            found = True
                        else:
                            new_subtrees.append(subtree)
                            subtree_index += 1

                    if not found:
                        continue

                    # else:
                    start_index = subtree_index
                    subtree_index += 1

                    # Find )
                    found = False
                    while (not found) and subtree_index < len(tree):
                        subtree = tree[subtree_index]
                        if subtree.label().startswith("PUNCT") and subtree[0] == ")":
                            found = True
                        else:
                            subtree_index += 1

                    if not found:
                        subtree_index = start_index
                        subtree = tree[subtree_index]
                        new_subtrees.append(subtree)
                        subtree_index += 1
                        continue

                    subtree_index += 1
                    if subtree_index < len(tree) and tree[subtree_index].label().startswith("NP"):
                        subtree_index += 1

                    stop_index = subtree_index

                    np_children: list[Tree] = []
                    subtree_index = start_index
                    while subtree_index < stop_index:
                        subtree = tree[subtree_index]
                        if subtree.label().startswith("NP"):
                            np_children += list(subtree)
                        else:
                            np_children.append(subtree)
                        
                        subtree_index += 1

                    new_subtrees.append(Tree("NP", np_children))

                new_chunked_sentence_tree_list.append(Tree(
                    tree.label(),
                    new_subtrees
                ))

            chunked_sentence_tree_list = new_chunked_sentence_tree_list

            # Akhiran demonstrative
            new_chunked_sentence_tree_list: list[Tree] = []
            for tree in chunked_sentence_tree_list:
                # Find noun chunk, ends with demonstrative pronoun.
                start_subtree_index = 0
                subtree_index = 0

                new_subtrees: list[Tree] = []
                while subtree_index < len(tree):
                    found = False
                    while (not found) and subtree_index < len(tree):
                        subtree = tree[subtree_index]
                        if subtree.label().startswith("NP"):
                            last_np_child_label = subtree[-1].label()
                            if last_np_child_label.startswith("DET") and "PronType=Dem" in last_np_child_label:
                                found = True
                            else:
                                new_subtrees.append(subtree)
                                subtree_index += 1
                        else:
                            new_subtrees.append(subtree)
                            subtree_index += 1

                    if found:
                        # Check first previous NP if exists
                        first_np_index = subtree_index
                        prev_subtree_index = subtree_index - 1
                        while prev_subtree_index >= start_subtree_index:
                            if tree[prev_subtree_index].label().startswith("NP"):
                                first_np_index = prev_subtree_index

                            prev_subtree_index -= 1

                        prev_subtree_index = subtree_index - 1
                        new_np_children: list[Tree] = list(tree[subtree_index])
                        while first_np_index <= prev_subtree_index:
                            child = new_subtrees.pop(-1)
                            if child.label().startswith("NP"):
                                new_np_children = list(child) + new_np_children
                            else:
                                new_np_children.insert(0, child)

                            prev_subtree_index -= 1

                        # first_np_index == prev_subtree_index + 1

                        new_subtrees.append(Tree("NP", new_np_children))
                        subtree_index += 1
                        start_subtree_index = subtree_index

                new_chunked_sentence_tree_list.append(Tree(
                    tree.label(),
                    new_subtrees
                ))

            chunked_sentence_tree_list = new_chunked_sentence_tree_list    

            np_id = 0
            for tree in chunked_sentence_tree_list:
                subtrees = [subtree for subtree in tree]
                for subtree in subtrees:
                    if subtree.label() == "NP":
                        subtree.set_label(f"NP;id={np_id}")
                        np_id += 1

            return chunked_sentence_tree_list
        
        return noun_chunk_strategy_2

    else:
        def noun_chunk_strategy_1(tree_list: list[Tree]):
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
                        r"<ADJ;.*>+",
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
                ],
                chunk_label="NP"
            )
            chunked_sentence_tree_list: list[Tree] = []

            for tree in tree_list:
                chunked_sentence_tree = parser.parse(tree)
                new_subtrees: list[Tree] = []
                subtree_index = 0
                while subtree_index < len(chunked_sentence_tree):
                    if subtree_index < len(chunked_sentence_tree) - 2 and chunked_sentence_tree[subtree_index].label().startswith("NP") and chunked_sentence_tree[subtree_index + 1].label().startswith("CCONJ") and chunked_sentence_tree[subtree_index + 2].label().startswith("NP"):
                        new_subtrees.append(Tree(
                            node=chunked_sentence_tree[subtree_index].label(),
                            children=chunked_sentence_tree[subtree_index, :] + [chunked_sentence_tree[subtree_index + 1]] + chunked_sentence_tree[subtree_index + 2, :]
                        ))
                        subtree_index += 3
                    else:
                        new_subtrees.append(chunked_sentence_tree[subtree_index])
                        subtree_index += 1

                chunked_sentence_tree_list.append(Tree(
                    node=chunked_sentence_tree.label(),
                    children=new_subtrees
                ))

            np_id = 0
            for tree in chunked_sentence_tree_list:
                subtrees = [subtree for subtree in tree]
                for subtree in subtrees:
                    if subtree.label() == "NP":
                        subtree.set_label(f"NP;id={np_id}")
                        np_id += 1

            return chunked_sentence_tree_list
        
        return noun_chunk_strategy_1
    