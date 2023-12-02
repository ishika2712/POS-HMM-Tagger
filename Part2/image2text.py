from PIL import Image, ImageDraw, ImageFont
import sys
import operator
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    return [
        [
            "".join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH)])
            for y in range(0, CHARACTER_HEIGHT)
        ]
        for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH)
    ]

def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    return {TRAIN_LETTERS[i]: load_letters(fname)[i] for i in range(len(TRAIN_LETTERS))}

def load_text_lines(training_text_path):
    with open(training_text_path, 'r') as file:
        return [[tuple(w for w in line.split())] for line in file]

def calculate_transition_probabilities(text_lines):
    transition_probabilities = {}
    character_counts = {}

    for line in text_lines:
        line_text = " ".join(line[0])
        for i in range(len(line_text)-1):
            if line_text[i] in TRAIN_LETTERS and line_text[i+1] in TRAIN_LETTERS:
                transition_key = line_text[i] + "#" + line_text[i + 1]
                transition_probabilities[transition_key] = transition_probabilities.get(transition_key, 0) + 1

    character_counts = {char: sum(transition_probabilities.get(char + "#", 0) for char in TRAIN_LETTERS) for char in TRAIN_LETTERS}

    for transition_key in transition_probabilities:
        transition_probabilities[transition_key] /= float(character_counts.get(transition_key.split("#")[0], 1))

    char_first_occurrences = {char: char_first_occurrences[char] / float(sum(character_counts.values())) for char in TRAIN_LETTERS if char in char_first_occurrences}

    char_count = 0
    char_occurrence_probs = {}
    for line in text_lines:
        line_text = " ".join(line[0])
        for char in line_text:
            if char in TRAIN_LETTERS:
                char_count += 1
                if char in char_occurrence_probs:
                    char_occurrence_probs[char] += 1
                else:
                    char_occurrence_probs[char] = 1

    char_occurrence_probs = {char: (char_occurrence_probs[char] + math.pow(10, 10)) / (float(char_count) + math.pow(10, 10)) for char in char_occurrence_probs}
    char_occurrence_probs = {char: char_occurrence_probs[char] / float(sum(char_occurrence_probs.values())) for char in char_occurrence_probs}

    return [char_occurrence_probs, transition_probabilities, char_first_occurrences]


def letter_comparison(test_letter, train_letters):
    comparison_result = {' ': 0.2}

    for i in TRAIN_LETTERS:
        match_count, mismatch_count, space_count = 0, 1, 0

        for k in range(len(test_letter)):
            for char_pos in range(len(test_letter[k])):
                if test_letter[k][char_pos] == ' ' and train_letters[i][k][char_pos] == ' ':
                    space_count += 1
                elif test_letter[k][char_pos] == train_letters[i][k][char_pos]:
                    match_count += 1
                else:
                    mismatch_count += 1

        comparison_result[i] = space_count if space_count > 340 else match_count / float(mismatch_count)

    total = sum(comparison_result.values())
    for key in comparison_result:
        comparison_result[key] /= float(total) if key != " " else 0.2

    return comparison_result

def hidden_markov_model(test_letters, train_letters, transition_probabilities):
    result = ['0'] * len(test_letters)
    probability_matrix = [[0 for _ in range(len(test_letters))] for _ in range(len(TRAIN_LETTERS))]

    # Initialize the first column of the matrix
    for r in range(len(TRAIN_LETTERS)):
        if TRAIN_LETTERS[r] in char_first_occurrences:
            letter_result = letter_comparison(test_letters[0], train_letters, 0)
            if TRAIN_LETTERS[r] in letter_result and letter_result[TRAIN_LETTERS[r]] != 0:
                probability_matrix[r][0] = -math.log10(letter_result[TRAIN_LETTERS[r]])

    # Fill the rest of the matrix using dynamic programming
    for c in range(1, len(test_letters)):
        letter_result = letter_comparison(test_letters[c], train_letters, 0)

        for r in range(len(TRAIN_LETTERS)):
            if TRAIN_LETTERS[r] in letter_result:
                min_cost = min(probability_matrix[i][c - 1] - math.log10(transition_probabilities[TRAIN_LETTERS[i] + "#" + TRAIN_LETTERS[r]]) - 10 * math.log10(letter_result[TRAIN_LETTERS[r]])
                               for i in range(len(TRAIN_LETTERS)))

                probability_matrix[r][c] = min_cost

    # Backtrack to find the optimal path
    result[0] = TRAIN_LETTERS[probability_matrix.index(min(probability_matrix[i][0] for i in range(len(TRAIN_LETTERS))))]

    for c in range(1, len(test_letters)):
        result[c] = TRAIN_LETTERS[probability_matrix[result[c - 1].index(min(probability_matrix[i][c] for i in range(len(TRAIN_LETTERS))))]][c]

    return "".join(result)

def simple_bayes_net(test_letters, train_letters):
    return ''.join(max(letter_comparison(letter, train_letters).items(), key=operator.itemgetter(1))[0] for letter in test_letters)

if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_path, train_txt_path, test_img_path) = sys.argv[1:]
TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
train_letters = load_training_letters(train_img_path)
test_letters = load_letters(test_img_path)

text_lines = load_text_lines(train_txt_path)
[char_first_occurrences, transition_probabilities, char_occurrence_probs] = calculate_transition_probabilities(text_lines)

print("Simple:", simple_bayes_net(test_letters, train_letters))
print("HMM:", hidden_markov_model(test_letters, train_letters, transition_probabilities))
