import numpy as np
import json
from datasets import load_dataset
import collections
import os
def load_embeddings(path):
    data = np.load(path)
    all_private_features = data['embeddings']
    return all_private_features
import argparse

from apis import get_api_class_from_name
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api',
        type=str,
        choices=['vllm', 'hf'],
        default='hf',
        help='Which foundation model API to use')

    # dataset
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['imdb', 'openreview','sst5', 'yelp'],
        default='imdb')
    parser.add_argument(
        '--data_file',
        type=str,
        default='data/imdb/imdb_test.csv')

    parser.add_argument(
        '--method',
        type=str,
        choices=['issdp', 'augpe'],
        default='issdp')
    # intra-Sentence DP
    parser.add_argument(
        '--word_embedding_model',
        type=str,
        default='distilbert-base-uncased',
        help='Which word embedding model to use'
    )

    # inter-Sentence DP
    parser.add_argument(
        '--phrase_model',
        type=str,
        default='qwen2.5:1.5b',
        help='Which phrase model to use'
    )

    parser.add_argument(
        '--combine_divide',
        type=int,
        default=2,
        help='Initial combine divide for inter-Sentence DP'
)
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs for inter-Sentence DP'
    )
    parser.add_argument(
        '--num_private_samples',
        type=int,
        default=20,
        help='Number of private samples to load')

    parser.add_argument(
        '--result_folder',
        type=str,
        default='result',
        help='Folder for storing results')

    parser.add_argument(
        '--feature_extractor_batch_size',
        type=int,
        default=1024,
        help='Batch size for feature extraction')

    parser.add_argument(
        '--feature_extractor',
        type=str,
        default='all-mpnet-base-v2',
        choices=["sentence-t5-xl", "sentence-t5-large",  "sentence-t5-base",
                 "all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2", "stsb-roberta-base-v2",
                 "roberta-large-nli-stsb-mean-tokens", "distilbert-base-nli-stsb-mean-tokens", 'text-embedding-ada-002'],
        help='Which image feature extractor to use')

    parser.add_argument(
        '--noise_multiplier',
        type=float,
        default=0.2,
        help='Noise multiplier for DP NN histogram')

    parser.add_argument(
        '--num_nearest_neighbor',
        type=int,
        default=1,
        help='Number of nearest neighbors for DP NN histogram'
    )

    parser.add_argument(
        '--nn_mode',
        choices=['L2', 'IP', 'cos_sim'],
        default='L2',
    )
    parser.add_argument(
        '--count_threshold',
        type=float,
        default=0.0,
        help='Threshold for DP NN histogram'
    )
    parser.add_argument(
        '--select_syn_mode',
        type=str,
        default='rank',
        choices=['prob', 'rank'],
    )

    parser.add_argument(
        '--save_syn_mode',
        type=str,
        default='selected',
        choices=['selected', 'all'],
    )
    args, api_args = parser.parse_known_args()
    if args.dataset == 'openreview':
        args.variation_type = 'openreview_rephrase'
    elif args.dataset == 'imdb':
        args.variation_type = 'imdb_rephrase'
    elif args.dataset == 'sst5':
        args.variation_type = 'sst5_rephrase'
    elif args.dataset == 'yelp':
        args.variation_type = 'yelp_rephrase'
    api_class = get_api_class_from_name(args.api)
    api = api_class.from_command_line_args(api_args)
    print(args)
    return args, api


def sample_dataset(data_name, dataset, label_name='label1', sample_size=5000):
    if sample_size == -1:
        return dataset
    else:
        return dataset.shuffle(seed=42).select(range(sample_size))

def load_data(dataset="openreview", data_file="data/test/test_openreview.csv", num_samples=100):
    print("data_file", data_file)
    raw_datasets = load_dataset("csv", data_files=data_file)
    original_data = sample_dataset(dataset, raw_datasets['train'], sample_size=num_samples)
    if dataset == 'openreview' or dataset == 'yelp':
        original_data = original_data.sort(['label1', 'label2'])
    elif dataset == 'imdb':
        original_data = original_data.sort('label')
    elif dataset =='sst5':
        original_data = original_data.sort('label_text')
    prompt_counter = collections.Counter()
    prompt_idexer = dict()
    data = []
    labels = []
    for i, line in enumerate(original_data):
        if dataset == 'openreview' or dataset == 'yelp':
            prompt = f"{line['label1']}\t{line['label2']}"
        elif dataset == 'imdb':
            prompt = 'positive' if line['label'] == 1 else 'negative'
        elif dataset == 'sst5':
            prompt = line['label_text']
        prompt_counter[prompt] += 1
        if prompt not in prompt_idexer.keys():
            prompt_idexer[prompt] = [i]
        else:
            prompt_idexer[prompt].append(i)
        data.append(line['text'])
        labels.append(prompt)

    return data, labels, prompt_counter, prompt_idexer

def save_samples(samples, labels, folder, dataset, filename='sample.json'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    datadict = []
    for i in range(len(samples)):
        if dataset == 'openreview' or dataset == 'yelp':
            label = labels[i].split('\t')
            datadict.append({'label1': label[0], 'label2': label[1], 'text': samples[i]})
        elif dataset == 'imdb':
            datadict.append({'label': labels[i], 'text': samples[i]})
        elif dataset == 'sst5':
            datadict.append({'label': labels[i], 'text': samples[i]})
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(datadict, f, sort_keys=False, indent=4)

def load_samples(file, dataset):
    if os.path.isdir(file):
        file = os.path.join(file, 'sample.json')
    with open(file, 'r') as f:
        datadict = json.load(f)
    data = []
    labels = []
    for line in datadict:
        if dataset == 'openreview' or dataset == 'yelp':
            data.append(line['text'])
            labels.append(f'{line["label1"]}\t{line["label2"]}')
        elif dataset == 'imdb' or dataset == 'sst5':
            data.append(line['text'])
            labels.append(line['label'])
    return data, labels