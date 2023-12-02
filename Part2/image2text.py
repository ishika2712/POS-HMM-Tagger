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
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):   
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def train_decoder(training_text_path):
    text_lines = []
    with open(training_text_path, 'r') as file:
        for line in file:
            words = tuple([w for w in line.split()])
            text_lines += [[words]]

    transition_probabilities = {}

    for line in text_lines:
        line_text = (" ").join(line[0])
        for i in range(0, len(line_text)-1):
            if (line_text[i] in TRAIN_LETTERS) and (line_text[i+1] in TRAIN_LETTERS):
                transition_key = line_text[i] + "#" + line_text[i + 1]
                if transition_key in transition_probabilities:
                    transition_probabilities[transition_key] = transition_probabilities[transition_key] + 1
                else:
                    transition_probabilities[transition_key] = 1

    character_counts = {}
    i = 0
    while i < len(TRAIN_LETTERS):
        count = 0
        for transition_key in transition_probabilities.keys():
            if TRAIN_LETTERS[i] == transition_key.split('#')[0]:
                count = count + transition_probabilities[transition_key]
        if count != 0:
            character_counts[TRAIN_LETTERS[i]] = count
        i += 1

    for transition_key in transition_probabilities.keys():
        transition_probabilities[transition_key] = (transition_probabilities[transition_key]) / (float(character_counts[transition_key.split("#")[0]]))

    initial_character_probabilities = {}
    for transition_key in transition_probabilities.keys():
        transition_probabilities[transition_key] = transition_probabilities[transition_key] / float(sum(character_counts.values()))

    char_first_occurrences = {}
    total = sum(character_counts.values())
    for line in text_lines:
        for word in line[0]:
            if word[0] in TRAIN_LETTERS:
                if word[0] in char_first_occurrences:
                    char_first_occurrences[word[0]] = char_first_occurrences[word[0]] + 1
                else:
                    char_first_occurrences[word[0]] = 1

    total = sum(char_first_occurrences.values())
    for char in char_first_occurrences.keys():
        char_first_occurrences[char] = char_first_occurrences[char] / float(total)

    char_count = 0
    char_occurrence_probs = {}
    for line in text_lines:
        line_text = (" ").join(line[0])
        for char in line_text:
            if char in TRAIN_LETTERS:
                char_count = char_count + 1
                if char in char_occurrence_probs:
                    char_occurrence_probs[char] = char_occurrence_probs[char] + 1
                else:
                    char_occurrence_probs[char] = 1

    for char in char_occurrence_probs.keys():
        char_occurrence_probs[char] = (char_occurrence_probs[char] + math.pow(10, 10)) / (float(char_count) + math.pow(10, 10))

    total = sum(char_occurrence_probs.values())
    for char in char_occurrence_probs.keys():
        char_occurrence_probs[char] = char_occurrence_probs[char] / float(total)

    return [char_occurrence_probs, transition_probabilities, char_first_occurrences]

def letter_comparison(test_letter, train_letters, flag):
    comparison_result = {}
    for i in TRAIN_LETTERS:
        match_count = 0
        mismatch_count = 1
        space_count = 0
        for k in range(0, len(test_letter)):
            for char_pos in range(0, len(test_letter[k])):
                if test_letter[k][char_pos] == ' ' and train_letters[i][k][char_pos] == ' ':
                    space_count = space_count + 1
                else:
                    if test_letter[k][char_pos] == train_letters[i][k][char_pos]:
                        match_count = match_count + 1
                    else:
                        mismatch_count = mismatch_count + 1
            comparison_result[' '] = 0.2
            if space_count > 340:
                comparison_result[i] = space_count
            else:
                comparison_result[i] = match_count / float(mismatch_count)

    total = 0
    for key in comparison_result.keys():
        if key != " ":
            total = total + comparison_result[key]
        else:
            total = total + 2
    for key in comparison_result.keys():
        if key != " ":
            if comparison_result[key] != 0:
                comparison_result[key] = comparison_result[key] / float(total)
            else:
                comparison_result[key] = 0.00001

    if flag == 0:
        return_result = dict(sorted(comparison_result.items(), key=operator.itemgetter(1), reverse=True)[:3])
    if flag == 1:
        return_result = comparison_result

    return return_result

def hmm_finder(test_letters):
    result = ['0'] * len(test_letters)
    A2D = []
    for i in range(0, len(TRAIN_LETTERS)):
        row = []
        for j in range(0, len(test_letters)):
            row.append([0, ''])
        A2D.append(row)

    letter_1 = letter_comparison(test_letters[0], train_letters, 0)
    for r in range(0, len(TRAIN_LETTERS)):
        if (TRAIN_LETTERS[r]) in char_first_occurrences and (TRAIN_LETTERS[r]) in letter_1 and letter_1[TRAIN_LETTERS[r]] != 0:
            A2D[r][0] = [- math.log10(letter_1[TRAIN_LETTERS[r]]), 'q1']

    for c in range(1, len(test_letters)):
        letter_result = letter_comparison(test_letters[c], train_letters, 0)
        if (' ') in letter_result:
            result[c] = " "

        for key in letter_result.keys():
            string = {}
            for r in range(0, len(TRAIN_LETTERS)):
                if (TRAIN_LETTERS[r] + "#" + key) in transition_probabilities and key in letter_result:
                    string[TRAIN_LETTERS[r]] = 0.1 * A2D[r][c - 1][0] - math.log10(transition_probabilities[TRAIN_LETTERS[r] + "#" + key]) - 10 * math.log10(letter_result[key])

            max_key = ''
            max_value = 0
            for i in string.keys():
                if max_value < string[i]:
                    max_value = string[i]
                    max_key = i
            if max_key != '':
                A2D[TRAIN_LETTERS.index(key)][c] = [string[max_key], max_key]

    max_value = math.pow(9, 99)
    for r in range(0, len(TRAIN_LETTERS)):
        if max_value > A2D[r][0][0] and A2D[r][0][0] != 0:
            max_value = A2D[r][0][0]
            result[0] = TRAIN_LETTERS[r]

    for c in range(1, len(test_letters)):
        min_value = math.pow(9, 96)
        for r in range(0, len(TRAIN_LETTERS)):
            if A2D[r][c][0] != 0 and A2D[r][c][0] < min_value and r != len(TRAIN_LETTERS) - 1 and result[c] != ' ':
                min_value = A2D[r][c][0]
                result[c] = TRAIN_LETTERS[r]

    i = 1
    while i < len(test_letters):
        min_value = math.pow(9, 96)
        for r in range(0, len(TRAIN_LETTERS)):
            if A2D[r][i][0] != 0 and A2D[r][i][0] < min_value and r != len(TRAIN_LETTERS) - 1 and result[i] != ' ':
                min_value = A2D[r][i][0]
                result[i] = TRAIN_LETTERS[r]
        i += 1

    max_value = math.pow(9, 99)
    for r in range(0, len(TRAIN_LETTERS)):
        if max_value > A2D[r][0][0] and A2D[r][0][0] != 0:
            max_value = A2D[r][0][0]
            result[0] = TRAIN_LETTERS[r]

    c = len(test_letters) - 2
    while c > 0:
        large_str = ''
        min_value = math.pow(10, 100)
        for row in range(0, len(TRAIN_LETTERS)):
            for r in range(0, len(TRAIN_LETTERS)):
                if large_str == '':
                    if min_value > A2D[r][c][0] and A2D[r][c][0] != 0:
                        min_value = A2D[r][c][0]
                        large_str = TRAIN_LETTERS[r]
            if (TRAIN_LETTERS[row] + "#" + large_str) in transition_probabilities:
                A2D[row][c][0] = A2D[row][c][0] - math.log10(transition_probabilities[TRAIN_LETTERS[row] + "#" + large_str])
        c -= 1

    return "".join(result)

def simple_bayes_net(test_letters, train_letters):
    result = ''
    for letter in test_letters:
        comparison_result = letter_comparison(letter, train_letters, 1)
        result += max(comparison_result.items(), key=operator.itemgetter(1))[0]
    return result

if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_path, train_txt_path, test_img_path) = sys.argv[1:]
TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
train_letters = load_training_letters(train_img_path)
test_letters = load_letters(test_img_path)

[char_first_occurrences, transition_probabilities, char_occurrence_probs] = train_decoder(train_txt_path)
print("Simple: " + simple_bayes_net(test_letters, train_letters))
print("HMM: " + hmm_finder(test_letters))
