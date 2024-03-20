import os
MAX_NUMBER_OF_ALPHABET_CUTS = 26
MIN_NUMBER_OF_ALPHABET_CUTS = 1
MIN_NUMBER_OF_WORD_CUTS = 1
MAX_NUMBER_OF_WORD_CUTS = 0.9
PERCENTAGE_SPLIT = 0.6
LETTERS = ['','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
DISTANCES = {
    'KNN':['dtw', 'gak', 'euclidean'],
    'RBF':['dtw', 'gak', 'euclidean'],
    'SVR':['dtw', 'gak', 'euclidean'],
    'RBFNN':['dtw', 'gak', 'euclidean']
}
