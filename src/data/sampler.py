
import hashlib 

import torch
import numpy as np
from overrides import overrides

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

class UniqueSequentialSampler(torch.utils.data.SequentialSampler):
    '''
    Sequential sampler which produces unique indexes even when
    duplicated across torch dataloader worker threads.
    
    Use this instead of UniqueRandomSampler if you want to classify
    the entire pointclouds (e.g. for evalulation as opposed to training)
    '''

    def __init__(self, num_workers, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

class UniqueRandomSampler(torch.utils.data.RandomSampler):
    '''
    Random sampler which producess unique indexes even when
    duplicated across torch dataloader worker threads. 

    torch.utils.data.RandomSampler will produce the same sequence
    of indexes in each worker thread.

    '''

    def __init__(self, dataset, replacement=False, num_samples=None):
        super().__init__(dataset, replacement, num_samples)

        self._offset = None

    def _get_offset(self):

        if self._offset is not None:
            return self._offset

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            self._offset = 0
        else:      
            self._offset = int(
                hashlib.md5(
                    str(worker_info.id).encode()
                ).hexdigest(),
                16
            )     

        return self._offset

    @overrides
    def __iter__(self):
        return (
            (idx + self._get_offset()) % len(self)
            for idx in super().__iter__()
        )