from collections import namedtuple
import json
import os

import numpy as np

DATASET_ALIASES = {
    'cifar': 'cifar10',
    'cifar10': 'cifar10',
    'gtsrb': 'gtsrb',
}


def normalize_dataset_name(name):
    key = str(name).strip().lower().replace('-', '').replace('_', '')
    if key not in DATASET_ALIASES:
        raise ValueError(
            'Unsupported dataset {!r}. Choose one of: cifar10, gtsrb'.format(name)
        )
    return DATASET_ALIASES[key]


def resolve_runtime_config(config):
    """Resolve dataset-dependent paths while keeping one switch in the JSON file."""
    dataset = normalize_dataset_name(config['data']['dataset'])
    prefix = 'cifar' if dataset == 'cifar10' else dataset
    config['data']['dataset'] = dataset

    paths = config['data'].get('paths', {})
    explicit_path = config['data'].get('path')
    if not explicit_path and dataset not in paths:
        raise KeyError('No data path configured for dataset {!r}'.format(dataset))
    config['data']['path'] = explicit_path or paths[dataset]
    downloads = config['data'].get('download', False)
    if isinstance(downloads, dict):
        config['data']['download'] = bool(downloads.get(dataset, False))

    format_values = {'dataset': dataset, 'dataset_prefix': prefix}
    config['model']['output_dir'] = config['model']['output_dir'].format(**format_values)
    config['model']['checkpoints'] = {
        role: path.format(**format_values)
        for role, path in config['model']['checkpoints'].items()
    }
    return config


def get_config(config_path=None):
    config_path = config_path or os.environ.get('DFTND_CONFIG', 'config_traincifar.json')
    with open(config_path) as config_file:
        #print(config_file)
        base_config = json.load(config_file)

    if os.path.exists('job_parameters.json'):
        with open('job_parameters.json') as param_config_file:
            param_config = json.load(param_config_file)
    else:
        param_config = {}

    config = base_config
    for section, d in param_config.items():
        for k in d:
            assert k in config[section]
        config[section].update(d)
    return resolve_runtime_config(config)


def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value) 
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj
