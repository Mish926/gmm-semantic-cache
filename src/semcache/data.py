"""Load CLINC150. Each intent stands in for one cacheable answer; out-of-scope = -1."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Split:
    texts: List[str]
    y: np.ndarray  # intent index, -1 for out-of-scope


@dataclass
class Clinc:
    train: Split
    val: Split
    test: Split
    intents: List[str]

    @property
    def n_intents(self) -> int:
        return len(self.intents)

    @property
    def all_texts(self) -> List[str]:
        return self.train.texts + self.val.texts + self.test.texts


def load_clinc(path) -> Clinc:
    d = json.loads(Path(path).read_text())
    intents = sorted({label for _, label in d["train"]})
    index = {name: i for i, name in enumerate(intents)}

    def make(rows):
        texts = [text for text, _ in rows]
        y = np.array([index.get(label, -1) for _, label in rows])
        return Split(texts, y)

    return Clinc(
        train=make(d["train"]),
        val=make(d["val"] + d["oos_val"]),
        test=make(d["test"] + d["oos_test"]),
        intents=intents,
    )
