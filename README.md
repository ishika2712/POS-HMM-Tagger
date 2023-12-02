# aminpri-isthakur-rganesh-a3

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
![image](https://media.github.iu.edu/user/24716/files/94faaf4e-f710-443b-abc6-66c4f5ce07e0)

### Challenges faced

## Part2 Reading text in an image
