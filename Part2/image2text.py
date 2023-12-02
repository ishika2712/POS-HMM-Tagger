#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: ISHIKA THAKUR(isthakur) PRITHVI AMIN(aminpri) RADHIKA GANESH(rganesh)

#

from PIL import Image, ImageDraw, ImageFont
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def simple_model(test_letters):
    # probality of Occurance of any letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"'"
    #26 uppercase latin characters, the 26 lowercase characters, the 10 digits, spaces, and 7 punctuation symbols, (),.-!?’"
    predicted_text = ""
    num_test_letters = len(test_letters)
    emission_matrix = emissionmatrix(test_letters, num_test_letters)
    
    for i in range(num_test_letters):
        letters_list = emission_matrix[i]
        predicted_letter = max(letters_list, key = lambda k : k[0])[1]
        predicted_text += predicted_letter 
    return predicted_text

def emissionmatrix(test_letters, num_test_letters):
    emission_matrix = []
    for k in range(num_test_letters):
        row_ele = []
        prob_li = 1/72 
        num_pixel = CHARACTER_HEIGHT * CHARACTER_WIDTH
        test_letter =  test_letters[k] 
        for train_letter in train_letters:
            pixel_match_count = pixel_unmatch_count= no_pixel_count = 0
            letter = train_letters[train_letter] 
            for i in range(CHARACTER_HEIGHT):
                for j in range(CHARACTER_WIDTH):
                    if test_letter[i][j] != letter[i][j]:
                        pixel_unmatch_count = pixel_unmatch_count + 1
                    else:    
                        if test_letter[i][j] == '*':
                            pixel_match_count = pixel_match_count + 1
                        else:
                            no_pixel_count = no_pixel_count + 1
                            
            prob_oi_li = (((0.87 *pixel_match_count)+ (0.05 *pixel_unmatch_count)+ (0.15 *no_pixel_count))/num_pixel) 
            prob_li_oi = prob_oi_li * prob_li
            row_ele.append((prob_li_oi, train_letter))
        emission_matrix.append(row_ele)
    return emission_matrix

def get_emisision_probability(test_letter):
    row_ele = []
    prob_li = 1/72 
    num_pixel = CHARACTER_HEIGHT * CHARACTER_WIDTH
    for train_letter in train_letters:
        pixel_match_count = pixel_unmatch_count= no_pixel_count = 0
        letter = train_letters[train_letter] 
        for i in range(CHARACTER_HEIGHT):
            for j in range(CHARACTER_WIDTH):
                if test_letter[i][j] != letter[i][j]:
                        pixel_unmatch_count = pixel_unmatch_count + 1
                else:    
                    if test_letter[i][j] == '*':
                        pixel_match_count = pixel_match_count + 1
                    else:
                        no_pixel_count = no_pixel_count + 1
                        
        prob_oi_li = (((0.87 *pixel_match_count)+ (0.05 *pixel_unmatch_count)+ (0.15 *no_pixel_count))/num_pixel) * prob_li
        row_ele.append((prob_oi_li, train_letter))
    return  row_ele 

def transitionprobmatrix(train_file_data):
    len_train_file_data = len(train_file_data)
    letter_count = {}
    combination_count = {}
    transition_matrix = []
    
    TRAIN_LETTERS1="$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' #"
    TRAIN_LETTERS2="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    
    for string in train_file_data:
        # finding the charector count
        for letter in string:
            if letter not in letter_count:
                letter_count[letter] = 1
            else:
                letter_count[letter] = letter_count[letter] + 1
        # finding the occurence of current letter and next letter sequence
        for letter in range(len(string)):
            first_letter = string[letter] 
            if first_letter != "#":
                secound_letter = string[letter+1]
                combination = first_letter + secound_letter
                if combination not in combination_count:
                            combination_count[combination] = 1
                else:
                            combination_count[combination] = combination_count[combination] + 1               
            else:
                pass
    
    #building transition matrix
    for i in range(len(TRAIN_LETTERS1)):
        transition_row = []
        train_letter = TRAIN_LETTERS1[i]
        for j in range(len(TRAIN_LETTERS2)):
            sequence = TRAIN_LETTERS1[i]+TRAIN_LETTERS2[j]
            if sequence in combination_count:
                if train_letter in letter_count:
                    transition_row.append(combination_count[sequence]/letter_count[train_letter])
            else:
                transition_row.append(0.00000001)
                  
        transition_matrix.append(transition_row)

    return transition_matrix

def get_vertibiprobability(vertibi_probability, i, probility_matrix):
    TRAIN_LETTERS1="$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' #"
    TRAIN_LETTERS2="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    
    for l in range(len(TRAIN_LETTERS1)):
        for m in range(len(TRAIN_LETTERS2)):
            transition_probability = transition_matrix[l][m]
            vertibi_probability.append(probility_matrix[i-1][m]+transition_probability)
     
    return vertibi_probability

# Viterbi Algorithm reference: https://www.cs.utexas.edu/~gdurrett/courses/sp2020/viterbi.pdf
def hmm_model(test_letters):
    TRAIN_LETTERS1="$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' #"
    TRAIN_LETTERS2="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    predicted_text = ""
    num_train_letters = len(TRAIN_LETTERS2)
    num_test_letters = len(test_letters)
#    probility_matrix = [[0]*num_train_letters]*num_test_letters
    probility_matrix = []
    path = []
    emission_matrix = emissionmatrix(test_letters, num_test_letters)
    
#    emisision_lst_first_letter = get_emisision_probability(emission_matrix, test_letters[0])
    #intialization:
    emission_probality_list = []

    for i in range(len(emission_matrix[0])):
        emission_probality_list.append(emission_matrix[0][i][0])
        
    probility_matrix.append(emission_probality_list)
#    probility_matrix[0] = emission_probality_list
    
    for i in range(1, num_test_letters):
        vertibi_probability_list = []
        for j in range(num_train_letters):
            emission_probality = emission_matrix[i][j] 
            vertibi_probability = []
            vertibi_probability = get_vertibiprobability(vertibi_probability, i, probility_matrix)
#            print(vertibi_probability)
#            print(emission_probality[0])
            vertibi_probability_val = 0
            vertibi_probability_val = emission_probality[0] * max(vertibi_probability)
#            print(vertibi_probability_val)
            vertibi_probability_list.append(vertibi_probability_val)
        probility_matrix.append(vertibi_probability_list)
        
#    print(probility_matrix)
#            print("here", vertibi_probability_val)
            
    predicted_text = ""
    for i in range(num_test_letters):
        current_max = 0
        max_index = 0
        for j in range(num_train_letters):
            if probility_matrix[i][j] > current_max:
                current_max = probility_matrix[i][j]
                max_index = j
        predicted_letter = TRAIN_LETTERS2[max_index]
        predicted_text += predicted_letter
        
    return predicted_text
        
#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

train_file_data = []
train_file = open("train-text.txt", "r")
train_file_line = train_file.readlines()
for string in train_file_line:
    string = '$' + string.rstrip() + '#'
    train_file_data.append(string)
    
transition_matrix = transitionprobmatrix(train_file_data)

predicted_text_simple = simple_model(test_letters)
predicted_text_hmm = hmm_model(test_letters)

# The final two lines of your output should look something like this:
print("Simple: " + predicted_text_simple)
print("   HMM: " + predicted_text_hmm)
