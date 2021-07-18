import json
import random
from pathlib import Path

import pyarrow as pa
from datasets import DatasetDict, Dataset


def load_data(path, sample_num = -1):
    path = Path(path)
    with open(path, 'rb') as f:
        json_data = json.load(f)
        data = json_data["data"]

    dataset = {
        "question": [],
        "context": [],
        "answers": []
    }

    num = sample_num if sample_num > 0 else len(data)
    for section in range(0, num):
        for parag in range(0, len(data[section]["paragraphs"])):
            context = data[section]["paragraphs"][parag]["context"]
            for qas in range(0, len(data[section]["paragraphs"][parag]["qas"])):
                question = data[section]["paragraphs"][parag]["qas"][qas]["question"]
                for ans in range(0, len(data[section]["paragraphs"][parag]["qas"][qas]["answers"])):
                    answer = data[section]["paragraphs"][parag]["qas"][qas]["answers"][ans]["text"]
                    answer_start = data[section]["paragraphs"][parag]["qas"][qas]["answers"][ans]["answer_start"]
                    dataset["context"].append(context)
                    dataset["question"].append(question)
                    dataset["answers"].append({"text": answer, "answer_start": [answer_start]})

    return dataset



def split_database(dataset, train_rate=0.9):
    db_length = len(dataset["question"])
    random.seed(4)
    all = list(zip(dataset["question"], dataset["answers"], dataset["context"]))
    random.shuffle(all)
    dataset["question"], dataset["answers"], dataset["context"] = zip(*all)

    train_no = round(db_length * train_rate)
    train = {
        "question": dataset["question"][0:train_no],
        "context": dataset["context"][0:train_no],
        "answers": dataset["answers"][0:train_no]
    }
    validation = {
        "question": dataset["question"][train_no+1:],
        "context": dataset["context"][train_no+1:],
        "answers": dataset["answers"][train_no+1:]
    }

    return train, validation


def create_dataset(sampling=False):
    train = load_data("data/COVID-QA-train.json", 15 if sampling else -1)
    validation = load_data("data/COVID-QA-val.json", 5 if sampling else -1)

    train_table = pa.Table.from_pydict(train)
    validation_table = pa.Table.from_pydict(validation)
    dataset = DatasetDict({"train": Dataset(train_table), "validation": Dataset(validation_table)})

    return dataset


def load_context_for_inference(path):
    path = Path(path)
    with open(path, 'rb') as f:
        json_data = json.load(f)
        data = json_data["data"]

    context = ""

    for section in range(0, len(data)):
        for parag in range(0, len(data[section]["paragraphs"])):
            context += data[section]["paragraphs"][parag]["context"]

    return context
