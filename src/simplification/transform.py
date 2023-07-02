from nltk import Tree
from .utils import get_attribute_value_from_tree_label, get_coref_id_from_tree_label, is_numeric_attribute_np

def transform(tree_list: list[Tree], strategy=1) -> list[Tree]:
    new_tree_list: list[Tree] = []
    noun_phrase_map: dict[int, Tree] = {}
    potentially_simplified_list = tree_list.copy()

    while len(potentially_simplified_list) > 0:
        tree = potentially_simplified_list.pop(0)
        changed = False

        first_new_tree, second_new_tree, changed = transform_for_conjoined_clauses(tree)
        if strategy != 5:
            if not changed:
                first_new_tree, second_new_tree, changed = transform_for_relative_clause(tree)
            
            if not changed:
                first_new_tree, second_new_tree, changed = transform_for_appositive(tree, strategy=strategy)

        else:
            
            if not changed:
                first_new_tree, second_new_tree, changed = transform_for_appositive(tree, strategy=strategy)
                
            if not changed:
                first_new_tree, second_new_tree, changed = transform_for_relative_clause(tree)

        if changed:
            potentially_simplified_list.insert(0, first_new_tree)
            if second_new_tree is not None:
                potentially_simplified_list.insert(1, second_new_tree)
        else:
            new_subtree_list: list[Tree] = []
            for subtree in tree:
                subtree_label = subtree.label()
                if not subtree_label.startswith("NP"):
                    new_subtree_list.append(subtree)
                    continue

                np_id = get_attribute_value_from_tree_label(subtree_label, "id")
                if np_id is None:
                    raise ValueError("NP should have an ID, right? Something is wrong.")
                
                np_id = int(np_id)
                noun_phrase_map[np_id] = subtree

                coref_id = get_coref_id_from_tree_label(subtree_label)
                if coref_id not in noun_phrase_map.keys():
                    new_subtree_list.append(subtree)
                    continue

                new_subtree = noun_phrase_map[coref_id]
                noun_phrase_map[np_id] = new_subtree

                # Check whether the NP exists in the same tree
                if strategy==5 and new_subtree in list(tree):
                    new_subtree_list.append(subtree)
                else:
                    new_subtree_list.append(new_subtree)

            new_tree = Tree(
                node=tree.label(),
                children=new_subtree_list
            )
            new_tree_list.append(new_tree)

    result = new_tree_list

    return result

def transform_for_conjoined_clauses(tree: Tree) -> tuple[Tree|None, Tree|None, bool]:
    # Find "CONJ-CLAUSE-1"
    first_conj_clause_index = 0
    while first_conj_clause_index < len(tree) and tree[first_conj_clause_index].label() != "CONJ-CLAUSE-1":
        first_conj_clause_index += 1

    if first_conj_clause_index == len(tree):
        return (None, None, False)
        
    # Find "CONJ-CLAUSE-2"
    second_conj_clause_index = first_conj_clause_index + 1
    while second_conj_clause_index < len(tree) and tree[second_conj_clause_index].label() != "CONJ-CLAUSE-2":
        second_conj_clause_index += 1

    if second_conj_clause_index == len(tree):
        new_tree = demark_clause(tree, [first_conj_clause_index])
        return (new_tree, None, True)

    dot_index = second_conj_clause_index + 1
    if dot_index == len(tree) or tree[dot_index, 0] != ".":
        new_tree = demark_clause(tree, [first_conj_clause_index, second_conj_clause_index])
        return (new_tree, None, True)
    
    dot_node = tree[dot_index]
    first_new_tree = Tree(
        node=new_id_because_of_split(tree, 0),
        children=[subtree for subtree in tree[first_conj_clause_index]] + [dot_node]
    )
    second_new_tree = Tree(
        node=new_id_because_of_split(tree, 1),
        children=[subtree for subtree in tree[second_conj_clause_index]] + [dot_node]
    )
    return (first_new_tree, second_new_tree, True)

def demark_clause(tree: Tree, clause_indices: list[int]):
    # Precondition: clause_indices is sorted

    prev_clause_index = -1
    children = []

    for clause_index in clause_indices:
        children += tree[prev_clause_index + 1:clause_index] + tree[clause_index, :]
        prev_clause_index = clause_index

    children += tree[prev_clause_index + 1:]

    return Tree(
        node=tree.label(),
        children=children
    )

def new_id_because_of_split(tree: Tree, split_id: int) -> str:
    tree_label = tree.label()
    return f"{tree_label}.{split_id}"

def transform_for_relative_clause(tree: Tree) -> tuple[Tree|None, Tree|None, bool]:
    rel_clause_index = 0
    while rel_clause_index < len(tree) and tree[rel_clause_index].label() not in ["SIMP-REST-CL", "SIMP-NONREST-CL"]:
        rel_clause_index += 1

    if rel_clause_index == len(tree):
        return (None, None, False)
    
    np_id = get_coref_id_from_tree_label(tree[rel_clause_index, 0].label()) # Lanjut resolve ini
    if np_id == -1:
        new_tree = demark_clause(tree, [rel_clause_index])
        return (new_tree, None, True)

    end_index = rel_clause_index + 1

    while end_index < len(tree) and tree[end_index, 0] != ".":
        end_index += 1

    if end_index == len(tree):
        new_tree = demark_clause(tree, [rel_clause_index])
        return (new_tree, None, True)

    end_index += 1

    referred_np_index = get_np_index_by_id(tree, np_id)
    if referred_np_index == -1 or referred_np_index >= rel_clause_index:
        new_tree = demark_clause(tree, [rel_clause_index])
        return (new_tree, None, True)

    if is_numeric_attribute_np(tree[referred_np_index]):
        new_tree = demark_clause(tree, [rel_clause_index])
        return (new_tree, None, True)
    
    first_part = tree[:rel_clause_index]
    if len(first_part) > 0 and " ".join(first_part[-1].leaves()) == ",":
        first_part = first_part[:-1]

    last_part = tree[rel_clause_index + 1:]
    if len(last_part) > 0 and " ".join(last_part[0].leaves()) == ",":
        last_part = last_part[1:]

    first_new_tree = Tree(
        node=new_id_because_of_split(tree, 0),
        children=first_part + last_part
    )

    second_new_tree = Tree(
        node=new_id_because_of_split(tree, 1),
        children=[tree[referred_np_index]] + tree[rel_clause_index, 1:] + [last_part[-1]]
    )

    return (first_new_tree, second_new_tree, True)

def get_np_index_by_id(tree: Tree, np_id: int | str):
    np_index = 0
    found = False
    while np_index < len(tree) and (not found):
        other_subtree_label = tree[np_index].label()
        if other_subtree_label.startswith("NP") and f"id={np_id}" in other_subtree_label:
            found = True
        else:
            np_index += 1

    if found:
        return np_index
    else:
        return -1
    
def transform_for_appositive(tree: Tree, strategy=1) -> tuple[Tree|None, Tree|None, bool]:
    appos_index = 0
    while appos_index < len(tree) and tree[appos_index].label() != "SIMP-APPOS":
        appos_index += 1

    if appos_index == len(tree):
        return (None, None, False)
    
    np_id = get_coref_id_from_tree_label(tree[appos_index, 0].label())
    if np_id == -1:
        new_tree = demark_clause(tree, [appos_index])
        return (new_tree, None, True)
    
    end_index = appos_index + 1
    while end_index < len(tree) and tree[end_index, 0] != ".":
        end_index += 1

    if end_index == len(tree):
        new_tree = demark_clause(tree, [appos_index])
        return (new_tree, None, True)

    end_index += 1

    referred_np_index = get_np_index_by_id(tree, np_id)
    if referred_np_index == -1 or referred_np_index >= appos_index:
        new_tree = demark_clause(tree, [appos_index])
        return (new_tree, None, True)
    
    first_part = tree[:appos_index]
    if len(first_part) > 0 and " ".join(first_part[-1].leaves()) == ",":
        first_part = first_part[:-1]

    last_part = tree[appos_index + 1:]
    if len(last_part) > 0 and " ".join(last_part[0].leaves()) == ",":
        last_part = last_part[1:]

    first_new_tree = Tree(
        node=new_id_because_of_split(tree, 0),
        children=first_part + last_part
    )

    appos_subtrees = tree[appos_index, :]
    if strategy != 5:
        appos_subtrees[0] = removed_coref(appos_subtrees[0])
        second_new_tree = Tree(
            node=new_id_because_of_split(tree, 1),
            children=[tree[referred_np_index]] + [Tree(node="AUX", children=["adalah"])] + list(appos_subtrees) + [last_part[-1]]
        )
    else:
        second_new_tree = Tree(
            node=new_id_because_of_split(tree, 1),
            children=[tree[referred_np_index]] + [Tree(node="AUX", children=["adalah"])] + list(appos_subtrees) + [last_part[-1]]
        )

    return (first_new_tree, second_new_tree, True)

def removed_coref(tree: Tree):
    label = tree.label()
    attributes = label.split(";")
    new_attributes: list[str] = []
    for attribute in attributes:
        if not attribute.startswith("coref"):
            new_attributes.append(attribute)

    return Tree(
        node=";".join(new_attributes),
        children=list(tree)
    )
