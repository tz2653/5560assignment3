import re
from collections import defaultdict
from typing import Dict, List

class BigramModel:
    def __init__(self, corpus: List[str]) -> None:
        tokenized = [re.findall(r"\b\w+\b", s.lower()) for s in corpus]
        self.transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for words in tokenized:
            for a, b in zip(words, words[1:]):
                self.transitions[a][b] += 1

    def next_word(self, word: str) -> str:
        options = self.transitions.get(word)
        if not options:
            return ""
        return max(options, key=options.get)  # 取出现频次最高的下一个词

    def generate_text(self, start: str, length: int) -> str:
        if length <= 1:
            return start
        out = [start.lower()]
        for _ in range(length - 1):
            nxt = self.next_word(out[-1])
            if not nxt:
                break
            out.append(nxt)
        return " ".join(out)
