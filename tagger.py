"""

In this assignment I will implement a Hidden Markov model
to predict the part of speech sequence for a given sentence.

"""

from math import log, isfinite
from collections import Counter


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, "r", encoding="utf-8") as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because missing Counter entries default to 0, not log(0)
A = {}  # transisions probabilities
B = {}  # emissions probabilities


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
      and emissionCounts data-structures.
     allTagCounts and perWordTagCounts should be used for baseline tagging and
     should not include pseudocounts, dummy tags and unknowns.
     The transitionCounts and emissionCounts
     should be computed with pseudo tags and should be smoothed.
     A and B should be the log-probability of the normalized counts, based on
     transisionCounts and  emmisionCounts

     Args:
       tagged_sentences: a list of tagged sentences, each tagged sentence is a
        list of pairs (w,t), as retunred by load_annotated_corpus().

    Return:
       [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """

    # init transitionCounts
    transitionCounts[START] = Counter()

    for line in tagged_sentences:
        lst_tag = START

        for tup in line:
            word, tag = tup

            allTagCounts[tag] += 1

            # add word tag
            if word not in perWordTagCounts:
                perWordTagCounts[word] = Counter()
            perWordTagCounts[word][tag] += 1

            # add punctuation tag
            if tag not in emissionCounts:
                emissionCounts[tag] = Counter()
            emissionCounts[tag][word] += 1

            # add transition
            if lst_tag not in transitionCounts:
                transitionCounts[lst_tag] = Counter()
            transitionCounts[lst_tag][tag] += 1
            lst_tag = tag

        # last transition
        if lst_tag not in transitionCounts:
            transitionCounts[lst_tag] = Counter()
        transitionCounts[lst_tag][END] += 1

    # transitionCounts to prob -> A
    for key, val in transitionCounts.items():
        A[key] = {}
        total = sum(val.values())
        for k, v in val.items():
            A[key][k] = log(v / total)

    # emissionCounts to prob -> B
    for key, val in emissionCounts.items():
        B[key] = {}
        total = sum(val.values())
        for k, v in val.items():
            B[key][k] = log(v / total)

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """

    tagged_sentence = []
    top_tag = max(allTagCounts, key=allTagCounts.get)

    for word in sentence:
        if word in perWordTagCounts:
            tup = (word, max(perWordTagCounts[word], key=perWordTagCounts[word].get))
        else:
            tup = (word, top_tag)
        tagged_sentence.append(tup)

    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): The HMM emmission probabilities.

    Return:
        list: list of pairs
    """

    tagged_sentence = []
    v_last = viterbi(sentence, A, B)
    lst = retrace(v_last)

    for w, t in zip(sentence, lst):
        tagged_sentence.append((w, t))

    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a triple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtracking.

    """

    # Viterbi matrix with the start word
    matrix = [[(START, None, 0)]]

    # calculate each column in the Viterbi matrix
    for i, word in enumerate(sentence):
        column = []
        temp = {}
        for triple in matrix[i]:
            lst_tag = triple[0]
            for next_tag in A[lst_tag]:
                if next_tag == END:
                    continue
                if word in B[next_tag]:
                    prob = triple[2] + A[lst_tag][next_tag] + B[next_tag][word]
                    if next_tag not in temp or prob > temp[next_tag][2]:
                        temp[next_tag] = (next_tag, triple, prob)

        if len(temp) == 0:  # OOV
            temp = {}
            tag = "NOUN"  # tried all combinations of tags, but got lower accuracy
            for triple in matrix[i]:
                if tag not in temp or triple[2] > temp[tag][2]:
                    temp[tag] = (tag, triple, triple[2])

        for val in temp.values():
            column.append(val)

        matrix.append(column)

    # add the end tag
    next_tag = END
    temp = {}
    for triple in matrix[-1]:
        lst_tag = triple[0]
        if next_tag in A[lst_tag]:
            prob = triple[2]
            if next_tag not in temp:
                temp[next_tag] = (next_tag, triple, prob)
            elif prob > temp[next_tag][2]:
                temp[next_tag] = (next_tag, triple, prob)
    matrix.append([temp[next_tag]])

    return matrix[-1][0]


def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
    reversing it and returning the list). The list should correspond to the
    list of words in the sentence (same indices).
    """

    nodes = []
    end_item = end_item[1]

    while end_item[1] is not None:
        nodes.insert(0, end_item[0])
        end_item = end_item[1]

    return nodes


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
    the HMM model.

    Args:
        sentence (pair): a sequence of pairs (w,t) to compute.
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emission probabilities.
    """

    p = 0  # joint log prob. of words and tags
    lst_tag = START

    for i, tup in enumerate(sentence):
        next_tag = tup[1]
        p += A[lst_tag][next_tag]
        if tup[0] in B[next_tag]:  # not OOV
            p += B[next_tag][tup[0]]
        lst_tag = next_tag
    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================


def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}


        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emission probabilities.


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == "baseline":
        return baseline_tag_sentence(
            sentence, model["baseline"][0], model["baseline"][1]
        )
    if list(model.keys())[0] == "hmm":
        return hmm_tag_sentence(sentence, model["hmm"][0], model["hmm"][1])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correctly predicted tags for oov words, and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)

    correct, correctOOV, OOV = 0, 0, 0

    for i, tup in enumerate(gold_sentence):
        if tup[1] == pred_sentence[i][1]:
            correct += 1
            if tup[0] not in perWordTagCounts:
                correctOOV += 1
        if tup[0] not in perWordTagCounts:
            OOV += 1

    return correct, correctOOV, OOV
