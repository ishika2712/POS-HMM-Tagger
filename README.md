# aminpri-isthakur-rganesh-a3

## Part1: Part-of-speech tagging

The problem at hand is Part-of-Speech (POS) tagging, a crucial task in natural language processing. The objective is to assign grammatical categories, such as nouns, verbs, and adjectives, to each word in a given sentence. This is framed as a classification problem where the models predict the most probable POS tag for each word. Two models are implemented in this program:

Simple Model: Based on word emission probabilities given POS tags.
Hidden Markov Model (HMM): Integrates both transition probabilities between POS tags and word emission probabilities.

### Training (train method):
Initialization: Sets up parameters and tables for training, including transition probabilities, emission probabilities, and start probabilities.
Data Processing: Handles variations in word representations (with or without quotes) for accurate training.
Training Algorithm: Iterates through the provided data to update transition and emission probabilities.
Normalization: Normalizes probabilities to prevent numerical instability issues.
Model Choices: Implements both a Simple Model and an HMM.

### Prediction (solve method):
#### Simplified Model (simplified method):

Predicts POS tags using emission probabilities.
Selects the most likely tag for each word.

###HMM Model (hmm_viterbi method):

Implements the Viterbi algorithm for efficient prediction.
Considers both transition and emission probabilities.
Utilizes dynamic programming to find the most likely sequence of POS tags.

### Accuracy
![image](https://media.github.iu.edu/user/24716/files/94faaf4e-f710-443b-abc6-66c4f5ce07e0)

### Challenges faced
