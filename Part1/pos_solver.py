###################################
# CS B551 Fall 2023, Assignment #3
#
# ISHIKA THAKUR(isthakur) PRITHVI AMIN(aminpri) RADHIKA GANESH(rganesh)
#



import random
import math
from collections import defaultdict, Counter
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence

    def posterior(self, model, sentence, label):
        # Calculate posterior probability based on the specified model
        if model == "Simple":
            product = sum(math.log(self.word_label_emi_prob[self.pos_index_mapping[label[i]]].get(sentence[i], 1e-10)) +
                          math.log(self.label_pos_prob[self.pos_index_mapping[label[i]]]) for i in range(len(sentence)))
        elif model == "HMM":
            product = math.log(self.word_label_emi_prob[self.pos_index_mapping[label[0]]].get(sentence[0], 1e-10)) + \
                      math.log(self.label_pos_prob[self.pos_index_mapping[label[0]]])

            product = product+sum(math.log(self.pos_transition_prob[self.pos_index_mapping[label[i - 1]]][self.pos_index_mapping[label[i]]]) +
                           math.log(self.word_label_emi_prob[self.pos_index_mapping[label[i]]].get(sentence[i], 1e-10)) +
                           math.log(self.label_pos_prob[self.pos_index_mapping[label[i - 1]]]) for i in range(1, len(sentence)))
        else:
            print("Unknown algo!")
            return 0
        return product

    # Do the training!
    #
    def train(self, data):
        # Initialize parameters and tables for training
        self.label_poss = ['conj', '.', 'adj', 'noun', 'prt', 'adv', 'pron', 'det', 'num', 'x', 'adp', 'verb']
        self.pos_index_mapping = {label: idx for idx, label in enumerate(self.label_poss)}
        self.pos_transition_prob = np.zeros((len(self.label_poss), len(self.label_poss)))
        self.label_pos_prob = np.zeros(len(self.label_poss))
        self.word_label_emi_prob = [{} for _ in range(len(self.label_poss))]
        self.start_prob = np.zeros(len(self.label_poss))
        self.transition_table_word, self.transition_table_label = {}, {}
        word_counter = 0
        for instance in data:
            secondlast_poslabel,last_poslabel = None,None
            for i in range(len(instance[0])):
                # Update transition and emission probabilities
                self.pos_transition_prob[self.pos_index_mapping[instance[1][i - 1]]][self.pos_index_mapping[instance[1][i]]] += 1 if i != 0 else 0
                word_counter += 1
                curr_word = instance[0][i].strip()
                if curr_word.startswith("'") and curr_word.endswith("'"):
                    curr_word = curr_word
                else:
                    curr_word = curr_word.strip("'")
                curr_pos = instance[1][i]
                self.label_pos_prob[self.pos_index_mapping[curr_pos]] += 1
                self.word_label_emi_prob[self.pos_index_mapping[curr_pos]].setdefault(curr_word, 0)
                self.word_label_emi_prob[self.pos_index_mapping[curr_pos]][curr_word] += 1
                self.transition_table_word = defaultdict(list)
                self.transition_table_word[curr_word].append((last_poslabel, curr_pos))
                self.transition_table_label.setdefault((secondlast_poslabel, last_poslabel), {})
                self.transition_table_label.setdefault((secondlast_poslabel, last_poslabel), {}).setdefault(curr_pos, 0)
                label_pos_pair = (secondlast_poslabel, last_poslabel)
                inner_dict = self.transition_table_label.setdefault(label_pos_pair, {})
                if curr_pos not in inner_dict:
                    inner_dict[curr_pos] = 0
                inner_dict[curr_pos] += 1
                secondlast_poslabel, last_poslabel = last_poslabel, curr_pos
            curr_pos = instance[1][0]
            self.start_prob[self.pos_index_mapping[curr_pos]] += 1

        # Normalize transition and emission probabilities
        for curr_pos in range(self.pos_transition_prob.shape[0]):
            overall_cnt = sum(self.pos_transition_prob[curr_pos])
            for target_label_idx in range(self.pos_transition_prob.shape[1]):
                self.pos_transition_prob[curr_pos][target_label_idx] /= max(overall_cnt, 1e-10)
                self.pos_transition_prob[curr_pos][target_label_idx] = max(self.pos_transition_prob[curr_pos][target_label_idx], 1e-10)
        for curr_pos in range(len(self.word_label_emi_prob)):
            overall_cnt = sum([cnt for (w, cnt) in self.word_label_emi_prob[curr_pos].items()])
            for curr_word in self.word_label_emi_prob[curr_pos]:
                self.word_label_emi_prob[curr_pos][curr_word] /= max(overall_cnt, 1e-10)
                self.word_label_emi_prob[curr_pos][curr_word] = max(self.word_label_emi_prob[curr_pos][curr_word], 1e-10)
        self.start_prob[curr_pos] /= len(data)
        self.transition_table_word[curr_word] = {pair: count / len(self.transition_table_word[curr_word]) for pair, count in Counter(
            self.transition_table_word[curr_word]).items()}
        for pair in self.transition_table_label:
            prob_cumulative = sum(self.transition_table_label[pair].values())
            self.transition_table_label[pair] = {trgt: count / prob_cumulative for trgt, count in
                                                             self.transition_table_label[pair].items()}
        self.label_pos_prob /= word_counter

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        # Predict POS tags using the simplified model
        predictions = []
        i = 0
        while i < len(sentence):
            word = sentence[i].strip()
            if word.startswith("'") and word.endswith("'"):
                word = word
            else:
                word = word.strip("'")
            top_label = None
            for label_idx in range(len(self.word_label_emi_prob)):
                if word in self.word_label_emi_prob[label_idx]:
                    current_value = self.word_label_emi_prob[label_idx][word]
                    if top_label is None or current_value > top_label[1]:
                        top_label = (self.label_poss[label_idx], current_value)
            predictions.append("x" if top_label is None else top_label[0])

            i += 1

        return predictions

    def hmm_viterbi(self, sentence):
        # Implement the Viterbi algorithm for HMM
        viterbi_matrix = []
        traceback = []

        for _ in range(self.pos_transition_prob.shape[0]):
            likelihoods = [0] * len(sentence)
            traceback_row = [""] * len(sentence)
            viterbi_matrix.append(likelihoods)
            traceback.append(traceback_row)

        viterbi_matrix = [
            [self.start_prob[label_idx] * self.word_label_emi_prob[label_idx].get(sentence[0], 1e-10) if matrix_col == 0 else 0 for matrix_col in
             range(len(sentence))]
            for label_idx in range(len(self.label_poss))
        ]
        i = 0
        for i in range(1, len(sentence)):
            for label_idx in range(len(self.label_poss)):
                max_value = float('-inf')
                max_label_pos = None
                for ct_label_indx in range(len(self.label_poss)):
                    current_value = viterbi_matrix[ct_label_indx][i - 1] * self.pos_transition_prob[ct_label_indx][label_idx]
                    if current_value > max_value:
                        max_value = current_value
                        max_label_pos = self.label_poss[ct_label_indx]
                traceback[label_idx][i] = max_label_pos
                viterbi_matrix[label_idx][i] = max_value * self.word_label_emi_prob[label_idx].get(sentence[i], 1e-10)

        predicted_string = ["" for _ in range(len(sentence))]
        max_pos_index_mapping = max(range(len(self.label_poss)), key=lambda label_idx: viterbi_matrix[label_idx][i])
        predicted_string[len(sentence) - 1] = self.label_poss[max_pos_index_mapping]
        for i in reversed(range(len(sentence) - 1)):
            prev_label_pos = predicted_string[i + 1]
            predicted_string[i] = traceback[self.pos_index_mapping[prev_label_pos]][i + 1]

        return predicted_string



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

