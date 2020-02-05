
import torch
from overrides import overrides

class InvalidIndexError(ValueError):
    pass

class FalibleIterDatasetWrapper(torch.utils.data.IterableDataset):

    def __init__(self, dataset: torch.utils.data.IterableDataset):
        self._dataset = dataset
        self._max_retries = 10

    def __iter__(self):
        return self

    def __next__(self):

        for _ in range(self._max_retries):
            try:
                return next(self._dataset)
            except InvalidIndexError as e:
                continue

        raise InvalidIndexError("Dataset returned InvalidIndexError more times than _max_retries")

    #forward all attribute calls to the underlying dataset
    #(e.g. num_features)
    def __getattr__(self, name):
        return getattr(self._dataset, name)


class FalibleDatasetWrapper(torch.utils.data.IterableDataset):
    '''Creates an IterableDataset around an ordinary map-style dataset
    where some indicies point to bad data. For example a patch dataset
    where some patches are empty. 

    The underlying dataset should throw BadDataException to indicate
    that the index is bad. FalibleDatasetWrapper will try retry 
    up to max_retries times to fetch an item from the dataset. 

    '''

    def __init__(self, dataset: torch.utils.data.Dataset, sampler: torch.utils.data.Sampler):
        self._sampler = iter(sampler)
        self._dataset = dataset
        self._max_retries = 10

    def __iter__(self):
        return self

    def __next__(self):

        for _ in range(self._max_retries):

            idx = next(self._sampler)

            try:
                return self._dataset[idx]
            except InvalidIndexError as e:
                # print('Skipping bad data sample', print(str(e)))
                continue
        
        raise InvalidIndexError("Dataset returned InvalidIndexError more times than _max_retries")

    #forward all attribute calls to the underlying dataset
    #(e.g. num_features)
    def __getattr__(self, name):
        return getattr(self._dataset, name)

    #this is just an approximation
    def __len__(self):
        return len(self._dataset)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            (idx + self._get_offset()) % len(self.data_source)
            for idx in super().__iter__()
        )