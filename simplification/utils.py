def get_coref_id_from_tree_label(tree_label: str) -> int:
    result = get_attribute_value_from_tree_label(tree_label, "coref")
    if result is None:
        return -1
    else:
        return int(result)

def get_attribute_value_from_tree_label(tree_label: str, key: str) -> str | None:
    result: str | None = None
    attributes = tree_label.split(";")
    for attribute in attributes:
        if attribute.startswith(key):
            _, result = attribute.split("=")
            break

    return result

def is_numeric_attribute_np(tree: str) -> bool:
    target_np_text = " ".join(tree.leaves()).lower()
    return "jumlah" in target_np_text and "sejumlah" not in target_np_text
