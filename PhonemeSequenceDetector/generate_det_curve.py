import numpy as np
from metrics import generate_det_curve

num_phonemes = 6

y_eval = np.load('data/' + str(num_phonemes) + '/eval/y.npy')
p_eval = np.load('data/' + str(num_phonemes) + '/eval/p.npy')
generate_det_curve(y_eval, p_eval)
