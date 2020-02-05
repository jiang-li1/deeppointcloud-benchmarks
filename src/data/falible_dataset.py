
import torch
from overrides import overrides

class InvalidIndexError(ValueError):
    pass

class FalibleIterDatasetWrapper(torch.utils.data.IterableDataset):

    def __init__(self, dataset: torch.utils.data.IterableDataset, length):
        self._dataset = dataset
        self._length = length

        self._num_returned = 0
        self._max_retries = 10

    def __iter__(self):
        return self

    def __next__(self):

        if self._num_returned == self._length:
            raise StopIteration()

        for _ in range(self._max_retries):
            try:
                d = next(self._dataset)
                self._num_returned += 1
                return d
            except InvalidIndexError as e:
                continue

        raise InvalidIndexError("Dataset returned InvalidIndexError more times than _max_retries")

    def __len__(self):
        return self._length

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


