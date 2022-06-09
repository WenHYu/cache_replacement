from tqdm import tqdm 
import numpy as np
from collections import deque, defaultdict
import os
import pandas as pd
import csv
tqdm.pandas()

def generate_train_samples(traces_pd, cache_size):
    x = []
    y = np.empty((0,3), int)
    cache = set()
    lru_recency = deque()
    lfu_frequency = defaultdict()
    fifo_order = deque()
    belady = defaultdict(deque)
    for i, row in tqdm(traces_pd.iterrows(), total=traces_pd.shape[0], desc="Belady: building index", leave = False):
        trace = row['Address']
        belady[trace].append(i)   


    for i, row in tqdm(traces_pd.iterrows(), total=traces_pd.shape[0], leave = False):
        trace = row['Address']
         # Pop the visit position
        belady[trace].popleft()
        if trace in cache:
            # update the lfu frequency
            lfu_frequency[trace] += 1
            # update the lfu 
            lru_recency.remove(trace)
            lru_recency.append(trace)
        elif len(cache) < cache_size:
            cache.add(trace)
            # update the lfu
            lfu_frequency[trace] = 1
            # update the lru
            lru_recency.append(trace)
            # update the fifo
            fifo_order.append(trace)
        else:
            # lfu propse a candidate
            lfu_candidate, f = min(lfu_frequency.items(), key=lambda a: a[1])
            # lru propose a candidate
            lru_candidate = lru_recency[0]
            # fifo propose a candidate
            fifo_candidate = fifo_order[0]
            candidates = [lfu_candidate, lru_candidate, fifo_candidate]
            # determine which one has the longest trace distance
            maxAccessPosition = -1
            valid_strategies = np.zeros((3,), dtype=int)
            
            is_non_recurrent_detected = 0 
            for j, candidate in enumerate(candidates):
                # this is the last time a block will ever be used
                # test = belady[candidate][0]
                if len(belady[candidate]) == 0:
                    maxAccessPosition = 0
                    evicted_candidate = candidate
                    valid_strategies[j] = 1

            if maxAccessPosition == -1:
                for j, candidate in enumerate(candidates):
                    if belady[candidate][0] > maxAccessPosition:
                        maxAccessPosition = belady[candidate][0]
                        evicted_candidate = candidate
                        valid_strategies[j] = 1
                    elif belady[candidate][0] == maxAccessPosition:
                        valid_strategies[j] = 1
                    else:
                        valid_strategies[j] = 0


            x.append(row)
            y = np.vstack((y, valid_strategies))

            # update the cache
            cache.remove(evicted_candidate)
            cache.add(trace)

            # update the meta data
            # update the lru meta data
            lru_recency.remove(evicted_candidate)
            lru_recency.append(trace)
            # update the lfu meta data
            lfu_frequency.pop(evicted_candidate)
            lfu_frequency[trace] = 1
            # 
            fifo_order.remove(evicted_candidate)
            fifo_order.append(trace)
    return x, y


print(os.getcwd())
df = pd.read_csv("./ChampSim/llc_access_trace.csv", sep=',')
df.columns = ['PC','Address']
df.head()

x, y = generate_train_samples(df,  2048)
np.save('x', x)
np.save('y', y)

#with open('xtrain.csv', 'w', encoding='UTF8', newline='') as f:
#    writer = csv.writer(f)
#    write multiple rows
#    writer.writerows(x)

#with open('ytrain.csv', 'w', encoding='UTF8', newline='') as f:
#    writer = csv.writer(f)
#    writer.writerows(y)