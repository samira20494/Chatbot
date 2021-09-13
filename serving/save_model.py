import os
import logging

from transformers import TFAutoModelForQuestionAnswering


def create_saved_model() -> None:
    if not os.path.exists('model'):
        model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model.save_pretrained('model', saved_model=True)
        logging.info('SavedModel saved to model/ directory')
    else:
        logging.info('model/ directory already exists; no new SavedModel objects saved')


if __name__ == "__main__":
    create_saved_model()