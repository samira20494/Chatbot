# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering
from transformers import AutoTokenizer
import src.config as cf
from src.preprocessing import load_context_for_inference
import requests
import json


class ActionCovid(Action):

    def name(self) -> Text:
        return "action_answer_covid_question"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        question = tracker.latest_message['text']
        text = load_context_for_inference("src/data/COVID-QA.json")

        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        # model = TFAutoModelForQuestionAnswering.from_pretrained(cf.setting["model_checkpoint"])

        inputs = tokenizer(question, text, add_special_tokens=True, truncation=True)
        input_ids = inputs["input_ids"]

        batch = [dict(inputs)]

        input_data = {'instances': batch}
        r = requests.post('http://localhost:8501/v1/models/bert:predict', data=json.dumps(input_data))
        output = json.loads(r.text)['predictions'][0]

        # output = model(inputs)
        answer_start = tf.argmax([output['start_logits']], axis=1).numpy()[
            0]  # Get the most likely beginning of answer with the argmax of the score
        answer_end = (tf.argmax([output['end_logits']], axis=1) + 1).numpy()[
            0]  # Get the most likely end of answer with the argmax of the score
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        start = answer.index("SEP")
        answer = answer[start:]
        answer = answer.replace("SEP", "")
        dispatcher.utter_message(text=f"{answer}")

        return []
