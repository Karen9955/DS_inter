from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD

class RecommenderSVDKNN:
    def __init__(self, emb_dim=500, num_neighb=30, metric='cosine'):
        self.params = {'embedding_dim': emb_dim,
                       'num_neighbours': num_neighb,
                       'metric': metric}
        self.svd = TruncatedSVD(self.params['embedding_dim'])


    def fit(self, x_train_sparse):
        self.X_train_svd = self.svd.fit_transform(x_train_sparse)
    
    def argpartition(self, row, num):
        row_sim, row_y = row
        mask = np.zeros(len(row_sim), bool)
        mask[row_y] = 1
        sim_temp = row_sim.copy()
        sim_temp[mask] = -float('inf')

        return np.argpartition(sim_temp, -num, axis=0)[-num:]
    
    def predict_sample(self, x_test_row, num=3):
        r = np.array(x_test_row)
        sim = cosine_similarity(self.X_train_svd, 
                                self.X_train_svd[r].sum(0).reshape((1, -1)))
        
        return self.argpartition((sim, r), num)
    
def precision(true, pred):
    if true in pred:
        return 1
    else:
        return 0

def mean_precision(true, pred):
    sum_ = 0
    for t, p in zip(true, pred):
        sum_ += precision(t, p)
    return sum_ / len(true)