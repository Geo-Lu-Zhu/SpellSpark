# tokenizer.py

PAD_IDX = 0
SEP_IDX = 1

class CharTokenizer:
    def __init__(self, char2idx=None):
        if char2idx is not None:
            self.char2idx = char2idx
        else:
            self.char2idx = {"<PAD>": PAD_IDX, "<SEP>": SEP_IDX}

    def fit(self, words):
        """Builds character-to-index mapping from a list of words/sequences."""
        for word in words:
            for ch in str(word):
                if ch not in self.char2idx:
                    self.char2idx[ch] = len(self.char2idx)
        return self

    def encode(self, text):
        """Encodes a string into a list of integer character IDs."""
        return [self.char2idx.get(ch, PAD_IDX) for ch in str(text)]

    def encode_pair(self, correct, incorrect):
        """Encodes pair format: [correct_ids] + [SEP_IDX] + [incorrect_ids]"""
        return self.encode(correct) + [SEP_IDX] + self.encode(incorrect)

    @property
    def vocab_size(self):
        return len(self.char2idx)

    @property
    def idx2char(self):
        return {idx: ch for ch, idx in self.char2idx.items()}


def build_tokenizer_from_pairs(pairs=None):
    """
    Builds and fits a CharTokenizer using standard alphabet characters
    plus optional input word pairs.
    """
    all_words = []
    if pairs:
        all_words = [word for pair in pairs for word in pair]
    # Standard character set from training pipeline
    all_words.extend(list("abcdefghijklmnopqrstuvwxyz -'"))
    return CharTokenizer().fit(all_words)


# Default tokenizer instance initialized with the base alphabet
_DEFAULT_TOKENIZER = build_tokenizer_from_pairs()


def encode_pair(correct_word, incorrect_word, char2idx=None):
    """
    Helper function for score.py. Encodes a (correct, incorrect) pair.
    
    If char2idx is provided (e.g. loaded from model1_config.json), 
    it uses that mapping; otherwise, it falls back to default alphabet indexing.
    """
    if char2idx is not None:
        tok = CharTokenizer(char2idx=char2idx)
    else:
        tok = _DEFAULT_TOKENIZER
    return tok.encode_pair(correct_word, incorrect_word)