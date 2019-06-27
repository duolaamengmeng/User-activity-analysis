from dataloader import DataLoader
import add_feature
import os
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
from itertools import product
import tqdm
from functools import partial

if __name__ == '__main__':
    data = DataLoader(filename='C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\data.txt',
                      time_col='created',
                      new_buyers=[],
                      onehot_features=['appId', 'action'],
                      features=['instanceId', 'num', 'memberId', 'created', 'appId', 'action'],
                      ten_col='instanceId',
                      error=['>"\'>alert&#40;5424&#41;',
                             '>"\'>alert&#40;5449&#41;',
                             '>"\'>alert&#40;5451&#41;',
                             '>"\'>alert&#40;5492&#41;',
                             '>"\'>alert&#40;5493&#41;',
                             '>"\'>alert&#40;5494&#41;',
                             '>"\'>alert&#40;5495&#41;',
                             '>"\'>alert&#40;589&#41;',
                             '>"\'>alert&#40;590&#41;',
                             '>"\'>alert&#40;591&#41;',
                             '>"\'>alert&#40;619&#41;',
                             '>"\'>alert&#40;620&#41;',
                             '>"\'>alert&#40;621&#41;',
                             '>"\'>alert&#40;637&#41;',
                             '>"\'>alert&#40;638&#41;',
                             '>"\'>alert&#40;639&#41;',
                             '>"\'>alert&#40;640&#41;',
                             '>"\'>alert&#40;663&#41;',
                             '>"\'>alert&#40;664&#41;',
                             '>"\'>alert&#40;665&#41;',
                             '>"\'>alert&#40;666&#41;'],
                      error_col='error', no_processor=800, data_split=True
                      )
    if data.datasplit:
        data.df = data.df.sample(n=self.sample_size)
