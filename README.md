
## Part1: Part-of-speech tagging

### Problem Formulation

1. Simplified Model:
The goal was to implement part-of-speech tagging using a simplified Bayes net. For this, we had to estimate the most-probable part-of-speech tag for each word in a sentence.

2. HMM:
The goal was to implement the Viterbi algorithm for part-of-speech tagging using a richer Bayes net. This model incorporated dependencies between words, making the tagging process more accurate.

3. Evaluation:
Evaluation involved loading training and testing data, deriving parameter estimates from the training corpus, and generating part-of-speech tagging results for each sentence in the testing file. The program was required to assess performance using evaluation metrics, including word and sentence accuracy.

### Program Functionality:

Class Structure (Solver): The code defines a class called Solver that encapsulates the functionality for training the model and predicting part-of-speech tags.

Training (train method): The train method initializes and populates various probability tables and matrices based on the provided training data. It calculates transition probabilities, emission probabilities, and other parameters required for both the simplified and HMM models.

Simplified Model:
Simplified Part-of-Speech Tagging: The simplified method predicts part-of-speech tags for a given sentence using a simplified model. It iterates through each word in the sentence and selects the part-of-speech label with the highest emission probability for that word.
Posterior Probability: The posterior method calculates the log posterior probability of a sentence given a part-of-speech label sequence. It uses emission probabilities and label probabilities to compute the overall probability.

Hidden Markov Model (HMM):
HMM Part-of-Speech Tagging (hmm_viterbi method): The hmm_viterbi method implements the Viterbi algorithm for Hidden Markov Models. It uses dynamic programming to find the most likely sequence of part-of-speech tags for a given sentence, considering both emission and transition probabilities.


Posterior Probability (posterior method with "HMM" model): Similar to the simplified model, the posterior method calculates the log posterior probability, but this time using the HMM-specific probabilities such as emission, transition, and initial probabilities.

### Accuracy
![image](https://media.github.iu.edu/user/24716/files/94faaf4e-f710-443b-abc6-66c4f5ce07e0?token=AAAGCCUN2RO4LO3BL3XGT7TH4OJCO)

### Challenges faced
Developing the Viterbi algorithm for HMM involved handling multiple probabilities and transitions. Understanding the algorithm's complexity and ensuring its correct implementation was challenging. Further, it was challenging to optimize the code to ensure that it is efficient and runs within the given time limit.

## Part2 Reading text in an image

### Description of problem

The problem is to extract data from an image which contains text and noise too.This was implemeted using HMM and simple bayes net classifier using the training data and training image  provided 
The code's main objective is to recognize and convert the characters in the test image into a machine-readable text format. It does this using two different methods:

Simple Bayes Net Method: This method calculates probabilities of the test characters being similar to the training characters based on their visual appearance. It then selects the most likely characters for each test character based on these probabilities.

Hidden Markov Model (HMM) Method: This method models the problem as an HMM, where it considers the transitions between characters and the probabilities of each character given the observed image. It calculates the most likely sequence of characters in the test image based on the HMM model.

### Code working
he provided code is an Optical Character Recognition (OCR) system that recognizes characters in a test image by comparing them to a set of known characters in a training image and uses probabilistic models to determine the most likely sequence of characters in the test image. Let's break down the working of the code in detail:

Loading Training Data:

The code begins by loading the training data, which consists of two parts:
Training Image: This image contains a set of known characters (letters, numbers, and punctuation marks) in a fixed format.Here we use courier-train.png as our training image as it contains a perfect version of each character. The image is loaded using the Python Imaging Library (PIL), and character images are extracted from it.
Training Text: This text file contains the actual text corresponding to the characters in the training image.For training we use bc.train file form Part1.

Training the Decoder:

The code then processes the training text data to train the decoder. The goal is to learn the probabilities of character transitions and character occurrences.
Transition Probabilities: It calculates transition probabilities between characters based on their co-occurrences in the training text.
Character Occurrence Probabilities: It calculates the probabilities of characters occurring as the first character in a word or sequence.
Character Occurrence Probabilities (smoothing): It smoothens character occurrence probabilities by adding a small constant to handle cases where a character may not occur in the training text.

Character Comparison Function:

The code defines a function called letter_comparison that compares a test character image with each of the known training characters. It computes a similarity score for each character based on visual similarity.

HMM (Hidden Markov Model) Method:

The HMM method models the problem as a Hidden Markov Model, where it considers the transitions between characters and the probabilities of each character given the observed image.
It initializes a dynamic programming matrix (A2D) to store intermediate results.
For each character in the test image, it calculates the most likely character based on the current observation and previous state probabilities.
It uses transition probabilities and emission probabilities (character visual similarity) to compute the probabilities.
The final result is a sequence of characters that maximizes the likelihood of the observed test image.

Simple Bayes Net Method:

The simple Bayes net method calculates probabilities of the test characters being similar to the training characters based on their visual appearance.
It selects the most likely characters for each test character based on these probabilities.

### Assumptions and challenges faced
No Text Layout Analysis: It does not perform text layout analysis to identify words, paragraphs, or other text structures. It treats the entire image as a sequence of characters.

Simplistic Hidden Markov Model (HMM): The HMM model used in the code is relatively simplistic and does not account for more complex patterns or higher-order dependencies often found in text.

Challenges were faced while formulating logic for the HMM algorithm as we have used dynamic programming approach which gave a lot of errors in the beginning.


![image](https://media.github.iu.edu/user/24842/files/24a2bd44-82c8-409d-b492-44ae0858c829?token=AAAGCCWIGMF4MVSH6PNQ2QDH4OJCO)
