import nltk
from nltk.translate.bleu_score import SmoothingFunction

def bleu_score(reference, hypothesis):
    if len(hypothesis) >= 4:
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction(epsilon=0.00000000000000001).method1)
    else:
        n = len(hypothesis)
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction(epsilon=0.00000000000000001).method1, weights=[1/n for _ in range(n)])

def bleu_score_for_text_simplification(simplified_sentences_list_output: list[str], simplified_sentences_list_target: list[str]):
    assert len(simplified_sentences_list_target) == len(simplified_sentences_list_output)

    total_score = 0.0
    for simplified_sentences_output, simplified_sentences_target in zip(simplified_sentences_list_output, simplified_sentences_list_target):
        final_score = bleu_score_for_simplified_sentences_pair(simplified_sentences_output, simplified_sentences_target)
        len_expected_simplification = len(simplified_sentences_target)
        total_score += final_score / len_expected_simplification

    return total_score / len(simplified_sentences_list_output)

def bleu_score_for_simplified_sentences_pair(simplified_sentences_output, simplified_sentences_target):
    scores = {}
    for reference_index, reference_sentence in enumerate(simplified_sentences_target):
        scores[reference_index] = {}
        for hypothesis_index, hypothesis_sentence in enumerate(simplified_sentences_output):
            reference_word_list = nltk.word_tokenize(reference_sentence.lower())
            hypothesis_word_list = nltk.word_tokenize(hypothesis_sentence.lower())
            score_from_bleu = bleu_score(reference=reference_word_list, hypothesis=hypothesis_word_list)
            scores[reference_index][hypothesis_index] = score_from_bleu

    max_score = 0.0
    len_expected_simplification = len(simplified_sentences_target)
    stack = [([], 0.0)]
        
    while len(stack) > 0:
            #selected_index = max(list(range(len(stack))), key=lambda index: stack[index][1] + (len_references - len(stack[index][0])))
        chosen_hypothesis_indices, score = stack.pop(0)

        if score + len_expected_simplification - len(chosen_hypothesis_indices) <= max_score:
            break # No need to check any nodes again.

        if len(chosen_hypothesis_indices) == len_expected_simplification:
            max_score = max(max_score, score)
            continue

        reference_index = len(chosen_hypothesis_indices)

            # Expand
        for hypothesis_index, score_from_bleu in scores[reference_index].items():
            if hypothesis_index not in chosen_hypothesis_indices:
                next_score = score + score_from_bleu
                next_chosen_hypothesis_indices = chosen_hypothesis_indices + [hypothesis_index]
                if len(stack) == 0:
                    stack.append((next_chosen_hypothesis_indices, next_score))
                else:
                    insert_index = 0
                    while insert_index < len(stack) - 1 and next_score - len(next_chosen_hypothesis_indices) < stack[insert_index][1] - len(stack[insert_index][0]):
                        insert_index += 1

                    if next_score - len(next_chosen_hypothesis_indices) < stack[insert_index][1] - len(stack[insert_index][0]):
                        stack.append((next_chosen_hypothesis_indices, next_score))
                    else:
                        stack.insert(insert_index, (next_chosen_hypothesis_indices, next_score))

                        # insert_index == len(stack) - 1


            # Skipping
        stack.append((chosen_hypothesis_indices + [-1], score))

    final_score = max_score
    return final_score

if __name__ == "__main__":
    reference = ["abdi", "ke", "pasar", "sekarang"]
    hypothesis = ["abdi"]
    print("Reference:", reference)
    print("Hypothesis:", hypothesis)
    result = bleu_score(reference, hypothesis)
    print("Score:", result)
