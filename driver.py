import tagger


def main():
    tagged_sentences = tagger.load_annotated_corpus("en-ud-train.upos.tsv")
    (
        allTagCounts,
        perWordTagCounts,
        transitionCounts,
        emissionCounts,
        A,
        B,
    ) = tagger.learn_params(tagged_sentences)

    sentence = tagger.load_annotated_corpus("en-ud-dev.upos.tsv")
    sentence = [[tup[0] for tup in sent] for sent in sentence]
    model = {"baseline": [perWordTagCounts, allTagCounts]}
    model = {"hmm": [A, B]}

    gold_sentence = tagger.load_annotated_corpus("en-ud-dev.upos.tsv")
    total = 0
    for sent in gold_sentence:
        for tup in sent:
            total += 1
    t_correct, t_correctOOV, t_OOV = 0, 0, 0
    for i, sent in enumerate(gold_sentence):
        correct, correctOOV, OOV = tagger.count_correct(
            sent, tagger.tag_sentence(sentence[i], model)
        )
        t_correct += correct
        t_correctOOV += correctOOV
        t_OOV += OOV

    print(
        f"Accuracy: {round(t_correct / total,3)}\nOOV Accuracy: {round(t_correctOOV / t_OOV,3)}"
    )


if __name__ == "__main__":
    main()
