import csv
from . import indo_ts
import json
import time
from .max_bleu import bleu_score_for_simplified_sentences_pair

from .utils import word_list_to_sentence
analyzer = indo_ts.TextSimplifier(tokenize_no_ssplit=True)

def load_dataset(file_path: str) -> dict:
    with open(file_path, mode="r") as file:
        csv_reader = csv.reader(file, delimiter=";")

        first_row = True
        selected_text_id = None
        selected_sentence_index = None
        # references = []
        references = {}
        dataset = {}
        for row in csv_reader:
            if first_row:
                first_row = False
                continue

            text_id, sentence_index, result, *_ = row
            if result == "":
                continue
            
            if selected_text_id is None:
                selected_text_id = text_id
                selected_sentence_index = int(sentence_index)
                references = {selected_sentence_index: []}
            elif selected_text_id != text_id and text_id != "":
                dataset[selected_text_id] = references
                selected_sentence_index = int(sentence_index)
                selected_text_id = text_id
                references = {selected_sentence_index: []}
            elif selected_sentence_index != sentence_index and sentence_index != "":
                selected_sentence_index = int(sentence_index)
                references[selected_sentence_index] = []
            # else: lanjut

            word_list = result.split(" ")
            sentence = word_list_to_sentence(word_list)

            references[selected_sentence_index].append(sentence)

        if len(references) > 0:
            dataset[selected_text_id] = references

    return dataset

result = load_dataset("private/simplification_dataset_v2.csv")

latest_analyzed_document_id = 100007
skipping = False

table_tuple_list = [
    ["ID Dokumen", "Kalimat", "Hasil Penyederhanaan", "Simplicity", "Grammaticality", "Meaning Preservation"]
]

output_path = f"private/test_result_{int(round(time.time()))}.txt"

total_score = 0.0
max_score = 0.0
for document_id, simplified_sentences in result.items():
    if skipping:
        if int(document_id) == latest_analyzed_document_id:
            skipping = False
        else:
            continue

    file_path = f"private/liputan6_data/canonical/train/{document_id}.json"
    with open(file_path, mode="r") as file_io:
        content: dict = json.load(file_io)

    sentences = []
    for sentence_index, word_list in enumerate(content["clean_article"]):
        if sentence_index == 0:
            colon_index = word_list.index(":")
            word_list = word_list[colon_index + 1:]
        elif sentence_index == len(content["clean_article"]) - 1:
            open_bracket_index = len(word_list) - 1
            while open_bracket_index >= 0 and word_list[open_bracket_index] != "(":
                open_bracket_index -= 1

            if open_bracket_index >= 0:
                word_list = word_list[:open_bracket_index]

        sentence = word_list_to_sentence(word_list)
        sentences.append(sentence)

    document = "\n\n".join(sentences)

    result: list[list[str]] = analyzer.simplify(document)

    # Baseline 1:
    # result: list[list[str]] = [[sentence] * 4 for sentence in document.split("\n\n")]

    for sentence_index in simplified_sentences.keys():
        input_sentence = word_list_to_sentence(content["clean_article"][sentence_index])
        output_sentences = result[sentence_index]
        target_sentences = simplified_sentences[sentence_index]

        average_bleu = bleu_score_for_simplified_sentences_pair(output_sentences, target_sentences) / len(target_sentences)
        total_score += average_bleu
        max_score += bleu_score_for_simplified_sentences_pair(target_sentences, target_sentences) / len(target_sentences)

        with open(output_path, mode="a") as f:
            print("Document ID:", document_id, file=f)
            print("Sentence index:", sentence_index, file=f)
            print("Input:", input_sentence, file=f)
            print("Output:", *output_sentences, sep="\n", file=f)
            print("Target:", *target_sentences, sep="\n", file=f)
            print("Average BLEU:", average_bleu, file=f)
            print("-" * 21, file=f)

        print(".", end="")

print("\nDone!")
print(f"Score: {total_score}/{max_score} ({total_score/max_score})")
