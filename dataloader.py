import math
import sys
import os.path
sys.path.insert(0, os.path.abspath('axolotl/src'))

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
import accelerate

from axolotl.utils.collators import DataCollatorForSeq2Seq

def split_batch(batch, pieces):
    example_tuple, labels = batch
    split_size = example_tuple[0].size(0) // pieces
    split_examples = zip(*(torch.split(tensor, split_size) for tensor in example_tuple))
    return [(ex, None) for ex in split_examples]

# A distributed batch sampler that supports grouping by length
class DistributedBatchSamper(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True, group_by_length=False, seed=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.seed = seed

    def __iter__(self):
        if self.group_by_length:
            index_and_length = ((i, len(item['input_ids'])) for i, item in enumerate(self.dataset))
            indices = list(sorted(index_and_length, key=lambda t: t[1]))
        elif self.shuffle:
            # deterministically shuffle based on seed
            g = torch.Generator()
            g.manual_seed(self.seed)
            shuffle_idx = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = [(i, len(self.dataset[i]['input_ids'])) for i in shuffle_idx]
        else:
            indices = [(i, len(item['input_ids'])) for i, item in enumerate(self.dataset)]
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        global_batch_size = self.batch_size * self.num_replicas
        global_batches = [indices[i:i+global_batch_size] for i in range(0, len(indices), global_batch_size)]
        if self.drop_last:
            global_batches = [b for b in global_batches if len(b) == global_batch_size]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed+1)
            shuffle_idx = torch.randperm(len(global_batches), generator=g)
            global_batches = [global_batches[i] for i in shuffle_idx]

        # make sure the largest batch comes first to OOM sooner rather than later
        largest_global_batch = 0
        max_len = 0
        for global_batch_idx, batch in enumerate(global_batches):
            for _, length in batch:
                if length > max_len:
                    max_len = length
                    largest_global_batch = global_batch_idx
        global_batches[0], global_batches[largest_global_batch] = global_batches[largest_global_batch], global_batches[0]

        batches_for_this_rank = [global_batch[self.rank:len(global_batch):self.num_replicas] for global_batch in global_batches]
        indices = [[i for i, _ in batch] for batch in batches_for_this_rank]
        return iter(indices)


# TODO: with 3 GPUs in the pipeline, I think the middle GPU doesn't pull from the iterator.
# Therefore, anything that uses RepeatingLoader.epoch would break. Need to fix.
class PipelineDataLoader:
    # A100 wants padding to multiple of 64, other cards are efficient with smaller, so just do 64
    def __init__(self, dataset, tokenizer, batch_size, gradient_accumulation_steps, data_parallel_world_size, data_parallel_rank, shuffle=True, group_by_length=False, pad_to_multiple_of=64):
        assert data_parallel_rank < data_parallel_world_size
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.pad_to_multiple_of = pad_to_multiple_of
        self.data_sampler = DistributedBatchSamper(
            dataset=dataset,
            batch_size=self.batch_size*self.gradient_accumulation_steps,
            num_replicas=data_parallel_world_size,
            rank=data_parallel_rank,
            shuffle=shuffle,
            group_by_length=group_by_length,
            drop_last=True
        )
        self.reset()

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self._create_dataloader()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_sampler) // self.batch_size

    def __next__(self):
        try:
            macro_batch = next(self.data)
        except StopIteration:
            self._create_dataloader()
            macro_batch = next(self.data)
            self.epoch += 1
        return macro_batch

    def _pull_batches_from_dataloader(self):
        for macro_batch in self.dataloader:
            self.num_batches_pulled += 1
            for batch in split_batch(macro_batch, self.gradient_accumulation_steps):
                yield batch

    def _create_dataloader(self):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        def collate_fn(examples):
            batch = data_collator(examples)
            # input to pipeline is (input_ids, attention_mask, labels)
            # this needs to return (features, labels)
            # it is OK if labels is None (the model just returns the loss anyway)
            return ((batch['input_ids'], batch['attention_mask'], batch['labels']), None)
        self.dataloader = DataLoader(
            self.dataset,
            pin_memory=True,
            batch_sampler=self.data_sampler,
            collate_fn=collate_fn,
            #num_workers=self.num_local_io_workers,
        )
        self.data = self._pull_batches_from_dataloader()
        self.num_batches_pulled = 0
    
    def state_dict(self):
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
        }
    
    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.num_batches_pulled = state_dict['num_batches_pulled']
        self.dataloader = accelerate.skip_first_batches(self.dataloader, self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()
    

# for testing
if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, use_fast=False, legacy=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'

    from datasets import Dataset
    data = []
    for i in range(1, 41):
        input_ids = torch.tensor([i]*i)
        data.append({'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids), 'labels': input_ids})
    dataset = Dataset.from_list(data)

    # dataloader = PipelineDataLoader(dataset, tokenizer, batch_size=2, gradient_accumulation_steps=2, data_parallel_world_size=1, data_parallel_rank=0, group_by_length=True, pad_to_multiple_of=None)
    # for batch in dataloader:
    #     if dataloader.epoch > 1:
    #         break
    #     print(batch)
    #     print()

    batch_size = 2
    gradient_accumulation_steps = 2
    data_parallel_world_size = 2
    data_parallel_rank = 0
    dataloader = PipelineDataLoader(
        dataset,
        tokenizer,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_rank=data_parallel_rank,
        shuffle=False,
        group_by_length=False,
        pad_to_multiple_of=None
    )
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])

    state_dict = dataloader.state_dict()
    dataloader = PipelineDataLoader(
        dataset,
        tokenizer,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_rank=data_parallel_rank,
        shuffle=False,
        group_by_length=False,
        pad_to_multiple_of=None
    )
    dataloader.load_state_dict(state_dict)
    print()
    print('-'*80)
    print()
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])