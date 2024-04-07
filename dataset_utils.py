import os
import os.path
import sys

sys.path.insert(0, os.path.abspath('axolotl/src'))

import torch
import datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.data import prepare_dataset
from tqdm import tqdm
import yaml

from utils import *


def yield_sequences_from_token_batch(tokenizer, token_batch, sequence_len):
    need = sequence_len
    example_tokens = []
    for tokens in tqdm(token_batch):
        tokens = tokens.tolist()
        assert tokens[-1] != tokenizer.eos_token_id, tokens[-1]
        tokens.append(tokenizer.eos_token_id)
        while len(tokens) > 0:
            taken = tokens[:need]
            tokens = tokens[need:]
            need -= len(taken)
            example_tokens.extend(taken)
            if len(example_tokens) >= sequence_len:
                assert len(example_tokens) == sequence_len
                yield example_tokens
                need = sequence_len
                example_tokens = []
    # yield anything remaining
    # TODO: disabled until I get training working with variable length sequences
    # if len(example_tokens) > 0:
    #     yield example_tokens


def slice_into_chunks(x, sequence_len, overlap=0):
    result = []
    step = sequence_len - overlap
    for i in range(0, len(x), step):
        result.append(x[i:i+sequence_len])
    return result


def load_raw_dataset(dataset_path, tokenizer, sequence_len, eval_size, overlap=0, subsample=None):
    if dataset_path.endswith('.txt'):
        dataset = datasets.load_dataset('text', data_files=dataset_path, sample_by='document')['train']
    elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
        dataset = datasets.load_dataset('json', data_files=dataset_path)['train']
    else:
        raise NotImplementedError()
    dataset.set_format(type='torch')

    num_proc = min(64, os.cpu_count())
    if subsample:
        dataset = dataset.shuffle(seed=13).select(list(range(int(subsample*len(dataset)))))
    dataset = dataset.map(lambda x: tokenizer(x['text']), batched=True, batch_size=10, remove_columns=dataset.column_names, desc='tokenizing', num_proc=num_proc)
    # TODO: maybe do it this way instead
    #dataset = dataset.map(lambda x: {'tokens': slice_into_chunks(x['tokens'][0], sequence_len, overlap=overlap)}, batched=True, batch_size=1)
    dataset = dataset.map(lambda x: {'input_ids': list(yield_sequences_from_token_batch(tokenizer, x['input_ids'], sequence_len))}, batched=True, batch_size=None, remove_columns=dataset.column_names, desc='splitting')
    dataset = dataset.map(lambda x: {'attention_mask': torch.ones_like(x['input_ids']), 'labels': x['input_ids']}, desc='adding attention_mask and labels')
    if eval_size > 0:
        split_datasets = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
        train_data = split_datasets['train']
        eval_data = split_datasets['test']
    else:
        train_data = dataset.shuffle(seed=42)
        eval_data = None
    return train_data, eval_data


def load_axolotl_dataset(dataset_path, tokenizer, sequence_len, eval_size):
    with open(dataset_path, 'r') as f:
        cfg = yaml.safe_load(f.read())
    if 'val_set_size' not in cfg and eval_size:
        cfg['val_set_size'] = eval_size
    if 'sequence_len' not in cfg and sequence_len:
        cfg['sequence_len'] = sequence_len
    # these two don't matter, but they have to be set
    cfg['batch_size'] = 1
    cfg['num_epochs'] = 1
    cfg = DictDefault(cfg)
    train_data, eval_data, *_ = prepare_dataset(cfg, tokenizer)
    train_data.set_format(type='torch')
    if eval_data is not None:
        eval_data.set_format(type='torch')
    return train_data, eval_data


def load_single_dataset(dataset_path, dataset_type, tokenizer, sequence_len, eval_size, subsample=None):
    if dataset_type in ['textfile', 'doclist']:
        with zero_first(is_main_process()):
            train_data, eval_data = load_raw_dataset(dataset_path, tokenizer, sequence_len, eval_size, subsample=subsample)
    elif dataset_type == 'axolotl':
        train_data, eval_data = load_axolotl_dataset(dataset_path, tokenizer, sequence_len, eval_size)
    else:
        raise NotImplementedError()
    def add_length(x): return {'length': len(x['input_ids'])}
    with zero_first(is_main_process()):
        train_data = train_data.map(add_length, desc='adding length field')
        if eval_data is not None:
            eval_data = eval_data.map(add_length, desc='adding length field')
    if is_main_process():
        print(f'train_data size: {len(train_data)}')
        if eval_data is not None:
            print(f'eval_data size: {len(eval_data)}')
    return train_data, eval_data


def combine_datasets(dataset_list, config, sample_weights):
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    mode = config.get('dataset_combination_mode', 'concatenate')
    if mode == 'concatenate':
        dataset = datasets.concatenate_datasets(dataset_list)
    elif mode == 'interleave':
        if 'batch_size_tokens' in config:
            # batches are formed so they have equal token counts, so interleave datasets based on token counts, not rows
            avg_lengths = torch.tensor([dataset['length'].to(torch.float32).mean() for dataset in dataset_list], dtype=torch.float32)
            sample_weights = sample_weights / avg_lengths
        sample_weights = sample_weights.to(torch.float64) # float64 or interleave_datasets complains that probs don't sum to 1
        probs = sample_weights / sample_weights.sum()
        dataset = datasets.interleave_datasets(dataset_list, probabilities=probs, seed=42, stopping_strategy=config.get('dataset_interleave_stopping_strategy', 'first_exhausted'))
    else:
        raise ValueError(mode)
    return dataset


def load_datasets(config, tokenizer):
    if 'datasets' not in config:
        raise ValueError('Need to specify at least one dataset')
    train_datasets = []
    sample_weights = []
    eval_datasets = {}
    i = 0
    for dataset_config in config['datasets']:
        if 'name' in dataset_config:
            name = dataset_config['name']
        else:
            name = f'dataset{i}'
            i += 1
        sample_weights.append(dataset_config.get('sample_weight', 1.0))
        train, eval = load_single_dataset(
            dataset_config['dataset_path'],
            dataset_config['dataset_type'],
            tokenizer,
            dataset_config['sequence_len'],
            dataset_config.get('eval_size', 0),
            subsample=dataset_config.get('subsample', None)
        )
        train_datasets.append(train)
        if eval is not None:
            eval_datasets[name] = eval

    for dataset_config in config.get('eval_datasets', []):
        if 'name' in dataset_config:
            name = dataset_config['name']
        else:
            name = f'dataset{i}'
            i += 1
        eval, _ = load_single_dataset(
            dataset_config['dataset_path'],
            dataset_config['dataset_type'],
            tokenizer,
            dataset_config['sequence_len'],
            eval_size=0,
            subsample=dataset_config.get('subsample', None)
        )
        eval_datasets[name] = eval

    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        with zero_first(is_main_process()):
            train_dataset = combine_datasets(train_datasets, config, sample_weights=sample_weights)
    return train_dataset, eval_datasets


# for testing
if __name__ == '__main__':
    import transformers
    # from datasets import disable_caching
    # disable_caching()

    tokenizer = transformers.AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, use_fast=False, legacy=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    train_data1, eval_data1 = load_raw_dataset('/home/anon/data/test/txt/*.txt', tokenizer, 100, 0.5)
    train_data2, eval_data2 = load_raw_dataset('/home/anon/data/test/json/*.jsonl', tokenizer, 100, 0.5)
    print(len(train_data1))
    print(len(train_data2))
