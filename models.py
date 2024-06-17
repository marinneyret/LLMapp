import tensorflow as tf
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
from transformers import pipeline
from transformers import AutoTokenizer


class Models:

    def __init__(self):
        self.describe = "je me regale"
        self.classifier_pipe= pipeline("zero-shot-classification")
        self.translator_pipe = pipeline("translation_en_to_fr", model="google-t5/t5-small")

    def zero_shot_classification(self, sequence: str):
        res = self.classifier_pipe(
            sequence,
            candidate_labels=["tech", "business"],
        )
        return res

    def translation(self, sequence: str):
        res = self.translator_pipe(sequence)
        return res

