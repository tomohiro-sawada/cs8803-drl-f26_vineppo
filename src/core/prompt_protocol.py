"""Model-specific prompts and evaluation stopping, shared by train and eval."""

from dataclasses import dataclass

PROMPT = "[MATH_TASK] Problem:\n{question}\n\nSolution:"
STOP = "\n\n\nProblem:"
RHOMATH = "rhomath"
OLMO3 = "olmo3-sft"
OLMO3_REVISION = "e1452fc572d51966ff4aaeb25118b891eb93e549"

OLMO_INSTRUCTIONS = {
    "math": "Solve the following problem. Put your final answer in \\boxed{...}.",
    "gsm8k": "Solve the following problem. Write the final numeric answer on its own line starting with #### followed by a space.",
}


def _added_special_id(tokenizer, text):
    token_id = tokenizer.convert_tokens_to_ids(text)
    assert type(token_id) is int
    token = tokenizer.added_tokens_decoder[token_id]
    assert token.special and token.content == text
    return token_id


@dataclass(frozen=True)
class PromptProtocol:
    key: str
    template: str
    context_size: int
    stop: str | None

    @property
    def add_special_tokens(self):
        return self.key == RHOMATH

    def format(self, tokenizer, question):
        text = self.template.replace("{question}", question)
        if self.key == OLMO3:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return text

    def encode(self, tokenizer, prompts):
        ids = tokenizer(
            prompts,
            add_special_tokens=self.add_special_tokens,
            padding=False,
        )["input_ids"]
        assert all(ids), "Expected nonempty prompts"
        if self.key == RHOMATH:
            assert tokenizer.bos_token_id is not None
            assert all(p[0] == tokenizer.bos_token_id for p in ids), "Expected BOS"
        return ids

    def eos_ids(self, tokenizer):
        assert self.key == OLMO3
        end = _added_special_id(tokenizer, "<|im_end|>")
        eos = tokenizer.eos_token_id
        assert type(eos) is int and eos in tokenizer.all_special_ids and eos != end
        return [end, eos]

    def context_special_ids(self, tokenizer):
        if self.key == RHOMATH:
            return ()
        return (_added_special_id(tokenizer, "<|im_start|>"), *self.eos_ids(tokenizer))


def prompt_protocol(key=RHOMATH, task=None):
    if key == RHOMATH:
        return PromptProtocol(RHOMATH, PROMPT, 2047, STOP)
    assert key == OLMO3 and task in OLMO_INSTRUCTIONS
    return PromptProtocol(OLMO3, OLMO_INSTRUCTIONS[task] + "\n\n{question}", 2048, None)
