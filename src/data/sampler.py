
import hashlib 

import torch
import numpy as np
from overrides import overrides
import time

def unique_random_index(length):
    worker_info = torch.utils.data.get_worker_info()

    r = np.random.choice(length)

    if worker_info is None:
        return r
    else:
        offset = int(
            hashlib.md5(
                str(worker_info.id).encode()
            ).hexdigest(),
            16
        )           
        return (r + offset) % length

def epoch_unique_random_seed():
    r = torch.initial_seed() + int(time.time() * 1e6)
    return r

def worker_unique_random_seed():
    worker_info = torch.utils.data.get_worker_info()

    r = torch.initial_seed()

    if worker_info is None:
        return r
    else:
        return r + worker_info.id

def _hash_int(i):
    return int(
        hashlib.md5(
            str(i).encode()
        ).hexdigest(),
        16
    )           

class BaseLazySampler(torch.utils.data.Sampler):

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def load(self, data_source):
        raise NotImplementedError


class UniqueSequentialSampler(torch.utils.data.SequentialSampler):
    '''
    Sequential sampler which produces unique indexes even when
    duplicated across torch dataloader worker threads.
    
    Use this instead of UniqueRandomSampler if you want to classify
    the entire pointclouds (e.g. for evalulation as opposed to training)
    '''

    def __init__(self, dataset, num_workers, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

        self.num_workers = num_workers
        self.i = 0
        self._range = None

    def _get_range(self, num_patches):

        if self._range is not None:
            return self._range

        worker_info = torch.utils.data.get_worker_info()

        # print('worker info: {}, num workers {}'.format(worker_info, self.num_workers))

        if worker_info is None:
            self._range = 0, num_patches
            return self._range

        wid = worker_info.id 

        patchesPerWorker = num_patches // self.num_workers

        start = wid * patchesPerWorker

        if wid == self.num_workers - 1:
            end = num_patches
        else:
            end = (wid+1) * patchesPerWorker

        self._range = start, end
        return self._range


    @overrides
    def __iter__(self):
        return self

    def __next__(self):
        
        start, end = self._get_range(len(self.data_source))
        if self.i + start < end:
            ret = self.i + start 
            self.i += 1
            return ret
        else:
            raise StopIteration

class LazyUniqueRandomSampler(BaseLazySampler, torch.utils.data.RandomSampler):

    def __init__(self, num_samples, replacement=False, worker_unique=True, epoch_unique=True):
        self._num_samples = num_samples
        self.replacement = replacement
        self.data_source = None
        self.worker_unique = worker_unique
        self.epoch_unique = epoch_unique



    @property
    def num_samples(self):
        return self._num_samples

    @overrides
    def __iter__(self):
        if self.data_source is None:
            raise Exception("This is a lazy sampler, you need to call sampler.load(<data_source>) before calling iter")

        n = len(self.data_source)

        return iter(np.random.choice(n, self.num_samples, self.replacement))

    @overrides
    def __len__(self):
        return self.num_samples

    def load(self, data_source):
        self.data_source = data_source

        seed = torch.initial_seed()
        if self.worker_unique:
            seed += worker_unique_random_seed()
        if self.epoch_unique:
            seed += epoch_unique_random_seed()
        np.random.seed(seed % 2**31)


class UniqueRandomSampler(torch.utils.data.RandomSampler):
    '''
    Random sampler which producess unique indexes even when
    duplicated across torch dataloader worker threads. 

    torch.utils.data.RandomSampler will produce the same sequence
    of indexes in each worker thread.

    '''

    def __init__(self, dataset, replacement=False, num_samples=None, worker_unique=True, epoch_unique=False):

        self.data_source = dataset
        self.replacement = replacement
        self._num_samples = num_samples
        self.worker_unique = worker_unique
        self.epoch_unique = epoch_unique

        self.index_sequence = None

    @overrides
    def __iter__(self):
        return self

    def __next__(self):
        if self.index_sequence is None:
            r = torch.initial_seed()
            if self.worker_unique:
                r += worker_unique_random_seed()
            if self.epoch_unique:
                r += epoch_unique_random_seed()
            np.random.seed(r % 2**31)
            self.index_sequence = iter(np.random.choice(len(self.data_source), self.num_samples, self.replacement))

        return next(self.index_sequence)


        # return (
        #     (idx + self._get_offset()) % len(self)
        #     for idx in super().__iter__()
        # )

