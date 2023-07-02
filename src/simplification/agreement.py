import logging
from nltk import Tree
import re

MASCULINE_TITLES = ["tuan", "pangeran", "raja", "kaisar", "bapak", "pak"]
FEMININE_TITLES = ["nyonya", "nona", "ratu", "permaisuri", "ibu", "bu"]
COMMON_TITLES = ["adipati", "pendeta", "dokter", "doktor", "profesor", "menteri", "sekretaris", "presiden"]
COMPANY_KEYWORDS = ["perusahaan", "pt", "tbk"]

PLACE_TIME_KEYWORD_LIST = [
    "rukun tetangga",
    "rt",
    "rukun warga",
    "rw",
    "desa",
    "dusun",
    "kelurahan",
    "kecamatan",
    "kabupaten",
    "kota",
    "kotamadya",
    "provinsi",
    "negara",
    "aceh",
    "bangka",
    "belitung",
    "banten",
    "bengkulu",
    "jawa",
    "kalimantan",
    "sulawesi",
    "nusa tenggara",
    "ntt",
    "ntb",
    "gorontalo",
    "jakarta",
    "jambi",
    "lampung",
    "maluku",
    "sumatera",
    "sumatra",
    "papua",
    "riau",
    "yogyakarta",
    "senin",
    "selasa",
    "rabu",
    "kamis",
    "jumat",
    "sabtu",
    "minggu",
    "pagi",
    "siang",
    "sore",
    "malam",
    "januari",
    "februari",
    "maret",
    "april",
    "mei",
    "juni`",
    "juli",
    "agustus",
    "september",
    "oktober",
    "november",
    "desember",
    "hari",
    "bulan",
    "minggu",
    "tahun",
    "kemarin",
    "besok",
    "lusa"
]

def extract_agreements_from_head_noun(tree_list: list[Tree]) -> list[Tree]:
    new_tree_list: list[Tree] = []
    for tree in tree_list:
        subtrees: list[Tree] = [subtree for subtree in tree]
        for subtree in subtrees:
            if subtree.label().startswith("NP"):
                head_noun = get_head_noun(subtree)
                head_noun_feats = head_noun.label().split(";")[1]

                for place_time_keyword in PLACE_TIME_KEYWORD_LIST:
                    if re.search(rf"\b{place_time_keyword}\b", " ".join(subtree.leaves()), re.IGNORECASE):
                        if head_noun_feats != "":
                            head_noun_feats += "|"
                        head_noun_feats += "LocationTime=Yes"

                        break

                subtree.set_label(subtree.label() + f";{head_noun_feats}")

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
