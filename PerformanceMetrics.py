import numpy as np


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def smape_batch(A,F, samples_count):
    return smape(A,F) / samples_count