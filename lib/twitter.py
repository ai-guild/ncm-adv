import data_utils
import numpy as np


PATH = '../data/twitter/'
        

class Twitter(object):

    def __init__(self, path=PATH):
        # data
        metadata, idx_q, idx_a = data_utils.load_data('../data/')

        # get dictionaries
        i2w = metadata['idx2w']
        w2i = metadata['w2idx']
                    
        # num of examples
        n = len(idx_q)

        def split_data():
            data = {}

            # sample indices from range(0,n)
            n_train = int(n*0.8)
            n_test = n - n_train
            indices = list(range(n))
            np.random.shuffle(indices)
            data['train'] = ( idx_q[indices][:n_train], idx_a[indices][:n_train] )
            data['test']  = ( idx_q[indices][n_train:], idx_a[indices][n_train:] )

            self.data = data

    def batch(self, batch_size, idx, data_key='train'):
        # get indices of batch size
        indices = list(range(batch_size))
        # shuffle indices within batch
        np.random.shuffle(indices)
        # return shuffled batch
        x,y = self.data[data_key]
        start, end = idx*batch_size, (idx+1)*batch_size
        return x[indices][start, end], y[indices][start, end]
