# src/evaluation/lm_eval_adapter.py

import torch
from lm_eval.api.model import TemplateLM


class MoREvalAdapter(TemplateLM):

    def __init__(
        self,
        model,
        tokenizer,
        device,
        batch_size=1,
        max_gen_toks=64,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.cfg.max_seq_len

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(
        self,
        string,
        left_truncate_len=None,
        add_special_tokens=False,
    ):
        tokens = self.tokenizer.encode(
            string,
            add_special_tokens=False,
        )

        if left_truncate_len:
            tokens = tokens[-left_truncate_len:]

        return tokens

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @torch.inference_mode()
    def _model_call(self, inps):
        inps = inps.to(self.device)

        output = self.model(inps)

        return output["logits"]

    @torch.inference_mode()
    def _model_generate(
        self,
        context,
        max_length,
        stop,
        **generation_kwargs,
    ):
        tokens = context.to(self.device)

        while tokens.shape[1] < max_length:

            current = tokens[:, -self.max_length:]

            output = self.model(current)

            next_token = output[
                "logits"
            ][:, -1].argmax(
                dim=-1,
                keepdim=True,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

            if (
                self.eot_token_id is not None
                and torch.all(
                    next_token
                    == self.eot_token_id
                )
            ):
                break

        return tokens