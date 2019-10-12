from sklearn.decomposition import NMF
from gcnetwork.gcmc.model import RecommenderGAE, RecommenderSideInfoGAE
from dataUtil import loadData
import argparse

def generate_masks(X):
    X = X.toarray()
    zero_masks = X != 0
if __name__ == '__main__':
    pass
