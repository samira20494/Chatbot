import json


def create_database():
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
                database["context"].append(context)
                for qas in range(0, len(data[section]['paragraphs'][parag]['qas'])):
                    question = data[section]['paragraphs'][parag]['qas'][qas]['question']
                    database["question"].append(question)
                    for ans in range(0, len(data[section]['paragraphs'][parag]['qas'][qas]['answers'])):
                        answer = data[section]['paragraphs'][parag]['qas'][qas]['answers'][ans]['text']
                        answer_start = data[section]['paragraphs'][parag]['qas'][qas]['answers'][ans]['answer_start']
                        database["answers"].append({"text": answer, "answer_start": [answer_start]})

    return database
