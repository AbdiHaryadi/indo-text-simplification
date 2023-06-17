from nltk.tree import Tree
from nltk.chunk import RegexpChunkParser
from nltk.chunk.regexp import ChunkRule, StripRule, ExpandLeftRule, MergeRule, ExpandRightRule, SplitRule, UnChunkRule

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