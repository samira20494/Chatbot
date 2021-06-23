import json

f_in = open('COVID-QA.json')
json_data = json.load(f_in)
data = json_data['data']

f_out = open('COVID-QA.txt', 'a')

for section in range(0, len(data)):
    for parag in range(0, len(data[section]['paragraphs'])):
        context = data[section]['paragraphs'][parag]['context']
        if context.find('corona') > 0:
            f_out.writelines(
                "\n\n######################### section: " + str(section) + " ################################\n")
            f_out.writelines("context: " + context + "\n")
            for qas in range(0, len(data[section]['paragraphs'][parag]['qas'])):
                q = data[section]['paragraphs'][parag]['qas'][qas]['question']
                for ans in range(0, len(data[section]['paragraphs'][parag]['qas'][qas]['answers'])):
                    a = data[section]['paragraphs'][parag]['qas'][qas]['answers'][ans]['text']
                f_out.writelines("question: " + q + "\n")
                f_out.writelines("answer: " + a + "\n\n")

f_in.close()
f_out.close()
