import json
import random

import pyarrow as pa
from datasets import DatasetDict, Dataset


def load_data(sampling = False):
    f_in = open("COVID-QA.json")
    json_data = json.load(f_in)
    data = json_data["data"]


    dataset = {
        "question": [],
        "context": [],
        "answers": []
    }

    num = 10 if sampling else len(data)
    for section in range(0, num):
        for parag in range(0, len(data[section]["paragraphs"])):
            context = data[section]["paragraphs"][parag]["context"]
            text = extract_text_from_context(context)
            for qas in range(0, len(data[section]["paragraphs"][parag]["qas"])):
                question = data[section]["paragraphs"][parag]["qas"][qas]["question"]
                dataset["question"].append(question)
                dataset["context"].append(text)
                for ans in range(0, len(data[section]["paragraphs"][parag]["qas"][qas]["answers"])):
                    answer = data[section]["paragraphs"][parag]["qas"][qas]["answers"][ans]["text"]
                    answer_start = data[section]["paragraphs"][parag]["qas"][qas]["answers"][ans]["answer_start"]
                    dataset["answers"].append({"text": answer, "answer_start": [answer_start]})

    return dataset


def extract_text_from_context(context):
    text_position_in_context = context.find("Text: ")
    if text_position_in_context > -1:
        text = context[text_position_in_context + 6:]
    else:
        text = ""
    return text


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

def create_dataset(data):
    train, validation = split_database(data)

    train_table = pa.Table.from_pydict(train)
    validation_table = pa.Table.from_pydict(validation)
    dataset = DatasetDict({"train": Dataset(train_table), "validation": Dataset(validation_table)})

    return dataset


