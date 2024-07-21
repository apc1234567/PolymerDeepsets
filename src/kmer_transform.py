import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import math

def generate_kmer_from_ID(distribution, ID):
    '''
    takes distribution: N_monomers dimensional vector of monomer distribution
    ID: str that identifies a given kmer, e.g. "001", "123"
    '''
    counts = np.zeros((len(distribution)), dtype = int)
    for char in ID:
        counts[int(char)] += 1
    multinomial = math.factorial(np.sum(counts))
    for num in counts:
        multinomial *= 1 / math.factorial(num)
    return np.prod(distribution**counts) * multinomial

desired_kmers = ["0", "1", "2", "00", "01", "02", "11", "12", "22"]
def transform_2mers(M):
    '''
    takes M: matrix of parent polymers represented by monomer distributions
    returns k-mer transform applied to each row
    '''
    n, _ = M.shape
    K = np.zeros((n, len(desired_kmers)))
    for i, ID in enumerate(desired_kmers):
        for j in range(n):
            K[j, i] = generate_kmer_from_ID(M[j], ID)
    return K

def augment_library(library, k:int = 2):
    '''
    returns library augmented with kmer features up to specified k
    '''
    poly = PolynomialFeatures(k, include_bias=False)
    return poly.fit_transform(library)




