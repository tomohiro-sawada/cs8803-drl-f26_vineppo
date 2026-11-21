from types import SimpleNamespace

import pytest
from tokenizers import AddedToken

from core.prompt_protocol import OLMO3, prompt_protocol


def test_olmo_chat_tokens_use_added_special_entries():
    start, end, eos = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
    vocab = {start: 1, end: 2, eos: 3}
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=vocab.__getitem__,
        added_tokens_decoder={
            i: AddedToken(text, special=True) for text, i in vocab.items()
        },
        eos_token_id=3,
        all_special_ids=[3],
    )
    profile = prompt_protocol(OLMO3, "math")
    assert profile.eos_ids(tokenizer) == [2, 3]
    assert profile.context_special_ids(tokenizer) == (1, 2, 3)
    tokenizer.added_tokens_decoder[2] = AddedToken(end, special=False)
    with pytest.raises(AssertionError):
        profile.eos_ids(tokenizer)
