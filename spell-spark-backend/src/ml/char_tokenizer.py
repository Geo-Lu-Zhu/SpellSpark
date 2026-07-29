PAD_IDX = 0
SEP_IDX = 1


class CharTokenizer:
    def __init__(self):
        self.char2idx = {"<PAD>": PAD_IDX, "<SEP>": SEP_IDX}

    def fit(self, words):
        for word in words:
            for ch in str(word).lower():
                if ch not in self.char2idx:
                    self.char2idx[ch] = len(self.char2idx)
        return self

    def encode(self, word):
        return [self.char2idx.get(ch, PAD_IDX) for ch in str(word).lower()]

    def encode_pair(self, correct, incorrect):
        return self.encode(correct) + [SEP_IDX] + self.encode(incorrect)

    @property
    def vocab_size(self):
        return len(self.char2idx)

    @property
    def idx2char(self):
        return {idx: ch for ch, idx in self.char2idx.items()}


def build_tokenizer_from_pairs(pairs):
    all_words = [word for pair in pairs for word in pair]
    all_words.extend(list("abcdefghijklmnopqrstuvwxyz -'"))
    return CharTokenizer().fit(all_words)