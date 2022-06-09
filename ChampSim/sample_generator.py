from tqdm import tqdm_notebook as tqdm 
import numpy as np
from collections import deque, defaultdict
import timeit
import pandas as pd

def generate_train_samples(traces_pd, cache_size):
    x = []
    y = []
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
            picked_candidate = -1
            for j, candidate in enumerate(candidates):
                # this is the last time a block will ever be used
                # test = belady[candidate][0]
                if len(belady[candidate]) == 0:
                    picked_candidate = candidate
                    picked_policy = j
                    break
                elif belady[candidate][0] > maxAccessPosition:
                    maxAccessPosition = belady[candidate][0]
                    picked_candidate = candidate
                    picked_policy = j

            x.append(row)
            y.append(picked_policy)

            # update the cache
            cache.remove(picked_candidate)
            cache.add(trace)

            # update the meta data
            # update the lru meta data
            lru_recency.remove(picked_candidate)
            lru_recency.append(trace)
            # update the lfu meta data
            lfu_frequency.pop(picked_candidate)
            lfu_frequency[trace] = 1
            # 
            fifo_order.remove(picked_candidate)
            fifo_order.append(trace)
    return x, y



df = pd.read_csv('./llc_access_trace.csv', sep=',')
df.columns = ['PC','Address']
df.head()

x, y = generate_train_samples(df,  2048)
y
