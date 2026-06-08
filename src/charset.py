from src.config import BLANK_TOKEN


class Charset:
    def __init__(self, alphabet: str):
        self.alphabet = "".join(dict.fromkeys(alphabet))
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(self.alphabet)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(self.alphabet)}

    @property
    def num_classes(self) -> int:
        return len(self.alphabet) + 1

    def encode(self, text: str) -> list[int]:
        missing = [ch for ch in text if ch not in self.char_to_idx]
        if missing:
            raise ValueError(f"Label contains characters outside alphabet: {sorted(set(missing))}")
        return [self.char_to_idx[ch] for ch in text]

    def decode_ctc(self, token_ids: list[int]) -> str:
        chars = []
        previous = None
        for token_id in token_ids:
            if token_id != BLANK_TOKEN and token_id != previous:
                chars.append(self.idx_to_char.get(token_id, ""))
            previous = token_id
        return "".join(chars)

