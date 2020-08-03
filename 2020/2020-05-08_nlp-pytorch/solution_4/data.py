import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typeguard import typechecked
from typing import Dict


class Surnames(Dataset):
    def __init__(self, raw_surnames: pd.DataFrame):
        self.raw_surnames = raw_surnames
        self.surname_vectorizer = CountVectorizer(analyzer='char')
        self.nationality_vectorizer = CountVectorizer()

        self.surname_vectorizer.fit(raw_surnames.surname)
        self.nationality_vectorizer.fit(raw_surnames.nationality)

        self.transform = transforms.Compose([transforms.ToTensor()])


    @typechecked
    def class_weights(self) -> torch.Tensor:
        class_counts = self.raw_surnames.nationality.value_counts().to_dict()

        def sort_key(item):
            return self.nationality_vectorizer.vocabulary_[item[0].lower()]
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        return 1.0 / torch.tensor(frequencies, dtype=torch.float32)


    def __len__(self) -> int:
        return len(self.raw_surnames)


    @typechecked
    def __getitem__(self, index: int) -> Dict:
        # https://stackoverflow.com/a/47604605/2543689
        row = self.raw_surnames.iloc[index]
        v_surname = self.surname_vectorizer.transform([row.surname]).toarray()
        v_nationality = self.nationality_vectorizer.transform([row.nationality]).toarray()

        return {
            'x_data': self.transform(v_surname),
            'y_target': self.transform(v_nationality)
        }
