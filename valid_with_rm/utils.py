import os
import json
import datasets
import warnings
from typing import List, Union
import json 
from tqdm import tqdm


def load_jsonl(jsonl_path):
    return [json.loads(line) for line in tqdm(open(jsonl_path, "rb"))]

def load_single_dataset(dataset_path: str, dataset_split: str = None) -> datasets.Dataset:
    # load from file
    if os.path.isfile(dataset_path):
        if dataset_path.endswith("jsonl"):
            dataset = load_jsonl(dataset_path)
        elif dataset_path.endswith("json"):
            dataset = json.load(open(dataset_path, "r"))
        elif dataset_path.endswith("parquet"):
            dataset = datasets.load_dataset('parquet', data_files=dataset_path)
        elif dataset_path.endswith("arrow"):
            dataset = datasets.Dataset.from_file(dataset_path)
        else: 
            raise RuntimeError(f"No support file type for {dataset_path.split('.')[-1]}")
        if isinstance(dataset, list):
            dataset = datasets.Dataset.from_list(dataset)
    
    # load from fold
    else:
        try:
            return datasets.load_dataset(dataset_path, split=dataset_split)
        except ValueError:
            dataset = datasets.load_from_disk(dataset_path)
    
    # dataset split
    if dataset_split is not None and isinstance(dataset, datasets.DatasetDict):
        dataset = dataset[dataset_split]
    print(dataset)
    return dataset


def load_dataset(dataset_paths: List[str]) -> datasets.Dataset:
    dataset = []
    for dataset_path in dataset_paths:
        try:
            dataset.append(load_single_dataset(dataset_path))
        except Exception as e:
            warnings.warn(f"Unvalid dataset, dataset: {dataset_path}, Error: {e}")
    if len(dataset) == 0:
        raise RuntimeError("No valid dataset")
    return datasets.concatenate_datasets(dataset)


def save_dataset(obj: Union[datasets.Dataset, List], save_path: str):
    if save_path.endswith(".json"):
        if isinstance(obj, datasets.Dataset):
            obj = obj.to_list()
        open(save_path, "w").write(json.dumps(obj, ensure_ascii=False, indent=2))
    elif save_path.endswith(".jsonl"):
        if isinstance(obj, datasets.Dataset):
            obj = obj.to_list()
        with open(save_path, "w") as fp:
            for row in obj:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif save_path.endswith(".parquet"):
        if isinstance(obj, List):
            obj = datasets.Dataset.from_list(obj)
        obj.to_parquet(save_path)
    else:
        if isinstance(obj, List):
            obj = datasets.Dataset.from_list(obj)
        obj.save_to_disk(save_path)


def split_batch(prompts: List[str], batch_size: int) -> List[List[str]]:
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]