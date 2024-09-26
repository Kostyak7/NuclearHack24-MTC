import re
import sys
import nltk
import json
import numpy as np
import pandas as pd
import config as cf
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer


def check_filepath(filepath: str, suffix: str = ".json") -> bool:
    if len(filepath) == 0:
        return False
    filepath = Path(filepath)
    return filepath.exists() and filepath.is_file() and filepath.suffix == suffix


class FormsCloud:
    def __init__(self):
        self.stop_words = set(stopwords.words('russian'))
        self.model = SentenceTransformer("sentence-transformers/LaBSE")

    def preprocess(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text.lower())
        # words = [word for word in text.split() if word not in self.stop_words]
        # return ' '.join(text)
        return text
    
    def parse_json(self, filepath: str) -> dict:
        if not check_filepath(filepath):
            return {}
        
        result = dict()
        with open(filepath, 'r', encoding='utf-8') as file:
            parsed_data = json.load(file)
            if "quiz" not in parsed_data or "questions" not in parsed_data["quiz"]:
                return {}
            for question in parsed_data["quiz"]["questions"]:
                if "question" in question:
                    result[question["question"]] = []
                    if "answers" in question and len(question["answers"]) > 0:
                        for answer in question["answers"]:
                            string = self.preprocess(answer)
                            if len(string):
                                result[question["question"]].append(string)
        return result
    
    def get_labse_embedding(self, question: str, answer: str):
        #combined_text = f"{question} [SEP] {answer}"
        combined_text = f"{answer}"
        embedding = self.model.encode(combined_text, convert_to_numpy=True, clean_up_tokenization_spaces=True)
        return embedding
    
    def run_dbscn(self, question: str, answers: list, word_vectors) -> list:
        result = []
        clusters = {}

        dbscan = DBSCAN(eps=0.41, min_samples=1, metric="cosine")
        labels = dbscan.fit_predict(word_vectors)

        for answer, label in zip(answers, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(answer)

        for label, cluster_words in clusters.items():
            word_counts = Counter(cluster_words)
            most_common_answer, most_common_count = word_counts.most_common(1)[0]
            theme_dict = {
                "percent": len(cluster_words) / len(answers) * 100,
                "theme": most_common_answer[0].upper() + most_common_answer[1:],
                "answers": cluster_words                
            }
            result.append(theme_dict)
        return result

    def get_one_cloud(self, question: str, answers: list) -> list:
        word_vectors = np.array([self.get_labse_embedding(question, answer) for answer in answers])    
        result = self.run_dbscn(question, answers, word_vectors)
        return sorted(result, key=lambda x: x["percent"], reverse=True)

    def get_clouds(self, filepath: str):
        quiz = self.parse_json(filepath)
        result = dict()
        for question, asnwers in quiz.items():
            result[question] = self.get_one_cloud(question, asnwers)
        return result


if __name__ == "__main__":
    arguments = sys.argv
    forms_cloud = FormsCloud()
    if (len(arguments) >= 2):
        clouds = forms_cloud.get_clouds(arguments[1])
        
        # Визуализация
        for question, cloud in clouds.items():
            print(f"\n\nВопрос: {question}")
            for answer in cloud:
                print(f"\t{answer["theme"]}: {round(answer["percent"], 1)}%\n\t\t\t[{', '.join(set(answer["answers"]))}]")

            theme_names = [theme["theme"] for theme in cloud]
            percent_values = [theme["percent"] for theme in cloud]

            plt.figure(figsize=(10, 6))
            plt.barh(theme_names, percent_values, color='skyblue')
            plt.xlabel('Процент', fontsize=14)
            plt.ylabel('Темы', fontsize=14)
            plt.title(question, fontsize=16)
            plt.grid(axis='x')
            for index, value in enumerate(percent_values):
                plt.text(value, index, f'{value:.1f}%', va='center')

            plt.tight_layout()
            plt.show()
            break





# Что еще следовало бы добавить?
# Убирать лишние пробелы
# Больше тестов
# Уточнение модели
# Улучшенная обработка опечаток
# Поддержка двух языков (ENG + RUS)
# Обработка жаргонных выржаний (бабосики)
# Обработка сокращений (зп
