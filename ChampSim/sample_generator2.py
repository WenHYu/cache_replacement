from tqdm import tqdm 
import numpy as np
from collections import deque, defaultdict
import pandas as pd
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

    MissDistance = 0
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
            MissDistance += 1
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

            s1 = pd.Series([MissDistance], index = ['MissDistance'])
            s2 = pd.concat([row, s1])
            x.append(s2)
            y = np.vstack((y, valid_strategies))
            MissDistance = 0 

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


df = pd.read_csv("./ChampSim/llc_access_trace.csv", sep=',')
df.columns = ['PC','Address']
df.head()

x, y = generate_train_samples(df,  2048)
print(x[:5])
np.save('x', x)
np.save('y', y)
