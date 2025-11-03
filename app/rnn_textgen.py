import torch
import torch.nn as nn

class TinyCharRNN(nn.Module):
    def __init__(self, vocab_size=50, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        logits = self.out(out)
        return logits, h

_index_to_char = [chr(i) for i in range(32, 32 + 50)]
_char_to_index = {ch: i for i, ch in enumerate(_index_to_char)}

def _char_to_idx(c):
    return _char_to_index.get(c, 0)

def _idx_to_char(i):
    return _index_to_char[i % len(_index_to_char)]

_global_rnn = TinyCharRNN()
_global_rnn.eval()

def generate_with_rnn_inference(start_word: str, length: int):
    """
    Generate random text sequence given a start word.
    """
    if length < len(start_word):
        length = len(start_word)

    seed_idx = torch.tensor([_char_to_idx(c) for c in start_word], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        _, h = _global_rnn(seed_idx)

    generated = list(start_word)
    cur_idx = seed_idx[:, -1:]

    for _ in range(length - len(start_word)):
        logits, h = _global_rnn(cur_idx, h)
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=0)
        next_idx = torch.multinomial(probs, num_samples=1)
        generated.append(_idx_to_char(int(next_idx)))
        cur_idx = next_idx.view(1, 1)

    return "".join(generated)
