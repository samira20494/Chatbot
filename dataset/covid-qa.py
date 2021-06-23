import json

f_in = open('200421_covidQA.json')
json_data = json.load(f_in)
data = json_data['data']

f_out = open('200421_covidQA.txt', 'a')

for context in range(0, len(data)):
    for parag in range(0, len(data[context]['paragraphs'])):
        for qas in range(0, len(data[context]['paragraphs'][parag]['qas'])):
            q = data[context]['paragraphs'][parag]['qas'][qas]['question']
            for ans in range(0, len(data[context]['paragraphs'][parag]['qas'][qas]['answers'])):
                a = data[context]['paragraphs'][parag]['qas'][qas]['answers'][ans]['text']
            if q.find('corona') > 0:
                f_out.writelines("question: " + q + "\n")
                f_out.writelines("answer: " + a + "\n\n")



f_in.close()
f_out.close()
