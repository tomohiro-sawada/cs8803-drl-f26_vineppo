"""
Inference backend used for:

- efficient rollouts
- running evals using a trained policy

Desiderata:
- We definitely want to fallback to Huggingface transformers (flash attention or sdpa) in case vLLM fails for whatever reason.

- Use generate for both kinds of rollout. Pass the original prompt for a full trajectory. For a VinePPO value estimate, pass prompt_ids + trajectory_ids[:step_boundary] with num_samples=K, then average the returns. Reuse the exact prefix token IDs so the branch starts at the intended state.
- Keep PPO’s old log-probs fixed across update epochs. For a first implementation, sample with temperature 1 and no top-k or top-p filtering, then score the sampled tokens once with Policy under torch.no_grad() before the first optimizer step. This assumes the rollout backend was synced to those weights. In the loss, use a separate completion mask aligned with the [B, T-1] scores. attention_mask handles padding.
- Keep backend construction concrete. Start with HFInference(policy: Policy) sharing the training model, so sync_weights can be a no-op for that instance. Give VLLMInference its own constructor and engine, and sync it at the start of each rollout round. Put the lazy vLLM import at the start of its __init__. In your stub, move @abstractclass onto the interface methods and remove self.engine: Policy from the base class.
"""

# typing
from dataclasses import dataclass
from email import policy
from typing import Literal, Sequence
from abc import abstractclass
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedModel, AutoTokenizer


# inference engines are lazy imported



@dataclass
class Completion:
    token_ids: list[int]     # continuation only
    finish_reason: Literal["stop", "length"]
    logprobs: list[float] | None = None  # one per returned token


class Policy(Module):
    def __init__(self, model: PreTrainedModel) -> None:
        """
        A wrapper around the model that provides a consistent interface for inference.
        Interface used during both training an inference.
        """
        super().__init__(self)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def forward(
        self,
        input_ids: Tensor,       # [B, T], prompt + completion
        attention_mask: Tensor,  # [B, T], excludes padding
    ) -> Tensor:                 # [B, T-1], log p(input_ids[:, 1:])
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate(
        self,
        input_ids: Tensor,       # [B, T], prompt + completion
        attention_mask: Tensor,  # [B, T], excludes padding
    ) -> Tensor:                 # [B, T-1], log p(input_ids[:, 1:])
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask)

    def tokenize(self, text: str) -> Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids

    def detokenize(self, input_ids: Tensor) -> str:
        return self.tokenizer.decode(input_ids[0])



@abstractclass
class LLMInference:
    def __init__(self,
            policy: Policy
    ):
        """
        Contract for inference methods.
        We want different frameworks (vllm, HF transformers, etc.) to abide by the same contract.
        Defining the class/method signature here will also let us start writing the
        main training algorithms without implementing the inference pipeline yet.
        """
        self.engine: Policy = policy

    @abstractclass
    def generate(
        self,
        prefixes: Sequence[Sequence[int]],
        *,
        num_samples: int = 1,
        max_new_tokens: int,
        temperature: float = 1.0,
        return_logprobs: bool = False,
    ) -> list[list[Completion]]:   # [prefix][sample]
        """
        Generate completions for each prefix/sample pair, returning token IDs and termination reasons.

        Args:
            prefixes: list of prompt token IDs
            num_samples: how many completions per prefix
            max_new_tokens: maximum length of each completion
            temperature: sampling temperature
            return_logprobs: whether to return log-probabilities

        Returns:
            list of list of Completion objects (one per prefix, then per sample)
        """
        raise NotImplementedError("generate not implemented")


    @abstractclass
    def sync_weights(self, policy: Policy) -> None:
        """
        Syncs the model weights from the training backend to the inference backend.
        """
        raise NotImplementedError("sync_weights not implemented")

class HFInference(LLMInference):
    def __init__(self, policy: Policy):
        """
        Initializes the HFInference backend.
        """
        super().__init__(policy)
        # lazy import HF inference backend
        # inifitaluize the inference
        pass


    def generate(self, ):
        pass


class VLLMInference(LLMInference):
    def __init__(self, policy: Policy):
        """
        Initializes the VLLMInference backend.
        """
        # Lazy check of dependencies; mostly vllm
        try:
            import vllm
        except ImportError:
            raise ImportError("vLLM is not installed")
        else:
            self.engine = vllm.LLM(model=policy.model)


    def generate(self,
        prefixes: Sequence[Sequence[int]],
        *,
        num_samples: int = 1,
        max_new_tokens: int,
        temperature: float = 1.0,
        return_logprobs: bool = False,
    ) -> list[list[Completion]]:
        """
        batched token-ID prompts → generated token IDs, lengths/termination information, and sampled-token log-probabilities when needed.
        """
        pass


    def sync_weights(self, ):
        """
        Syncs the model weights from the training backend to the inference backend.
        """
        pass
