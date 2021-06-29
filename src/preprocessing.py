import json
import random

import pyarrow as pa
from datasets import DatasetDict, Dataset
import config as cf

def load_data():
    f_in = open('COVID-QA.json')
    json_data = json.load(f_in)
    data = json_data['data']


    database = {
        "question": [],
        "context": [],
        "answers": []
    }

    for section in range(0, len(data)):
        for parag in range(0, len(data[section]['paragraphs'])):
            context = data[section]['paragraphs'][parag]['context']
            if context.find('corona') > 0:
                for qas in range(0, len(data[section]['paragraphs'][parag]['qas'])):
                    question = data[section]['paragraphs'][parag]['qas'][qas]['question']
                    database["question"].append(question)
                    database["context"].append(context)
                    for ans in range(0, len(data[section]['paragraphs'][parag]['qas'][qas]['answers'])):
                        answer = data[section]['paragraphs'][parag]['qas'][qas]['answers'][ans]['text']
                        answer_start = data[section]['paragraphs'][parag]['qas'][qas]['answers'][ans]['answer_start']
                        database["answers"].append({"text": answer, "answer_start": [answer_start]})

    return database


def split_database(database, train_rate=0.9):
    db_length = len(database["question"])
    random.seed(4)
    all = list(zip(database["question"], database["answers"], database["context"]))
    random.shuffle(all)
    database["question"], database["answers"], database["context"] = zip(*all)

    train_no = round(db_length * train_rate)
    train = {
        "question": database["question"][0:train_no],
        "context": database["context"][0:train_no],
        "answers": database["answers"][0:train_no]
    }
    validation = {
        "question": database["question"][train_no+1:],
        "context": database["context"][train_no+1:],
        "answers": database["answers"][train_no+1:]
    }

    return train, validation

def create_dataset():
    data = load_data()

    train, validation = split_database(data)

    train_table = pa.Table.from_pydict(train)
    validation_table = pa.Table.from_pydict(validation)
    dataset = DatasetDict({"train": Dataset(train_table), "validation": Dataset(validation_table)})

    return dataset


