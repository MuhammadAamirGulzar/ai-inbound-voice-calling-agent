import random
import numpy as np
from typing import Dict, List, Union
import ollama
from constants import clean_test, get_faqs_dict


class Minibot:
    def __init__(
        self, args: Dict[str, List[str]] = None, THRESHOLD: float = 0.75
    ) -> None:
        self.args = args
        self.THRESHOLD = THRESHOLD
        self.UNKNOWN = "*"

    def fill_faqs(self, args: Dict[str, List[str]] = None) -> None:
        if args is None:
            args = self.args
        self.args = args
        self.FAQS_DICT = get_faqs_dict(args)
        questions = list(self.FAQS_DICT.keys())
        self.answers = list(self.FAQS_DICT.values())
        self.questions_vectors = self.generate_embeddings(questions)

    def generate_embeddings(
        self, sentences: List[str], model: str = "nomic-embed-text"
    ) -> np.ndarray:
        embeddings = []
        for sentence in sentences:
            print(f'Generating embeddings for "{sentence}"')
            while True:
                try:
                    response = ollama.embeddings(
                        model=model,
                        prompt=sentence,
                    )
                    break
                except:
                    continue
            embeddings.append(response["embedding"])
        return np.array(embeddings)

    def get_response(
        self, query: str = "", threshold: float = 0.75
    ) -> Union[str, float]:
        print(f'Query to get response: "{query}"')
        query_vector = self.generate_embeddings([query])[0]
        similarities = np.dot(self.questions_vectors, query_vector) / (
            np.linalg.norm(self.questions_vectors, axis=1)
            * np.linalg.norm(query_vector)
        )
        max_similarity = np.max(similarities)
        best_match_idx = np.argmax(similarities)
        if max_similarity >= threshold:
            return random.choice(self.answers[best_match_idx]), max_similarity
        else:
            return "*", 0.0
