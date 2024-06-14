import numpy as np
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

desired_kmers = ["00", "01", "02", "11", "12", "22"]
def transform(M):
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
