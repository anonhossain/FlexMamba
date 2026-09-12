# src/evaluation/lm_eval_adapter.py

# import torch
# from lm_eval.api.model import TemplateLM


# class MoREvalAdapter(TemplateLM):

#     def __init__(
#         self,
#         model,
#         tokenizer,
#         device,
#         batch_size=1,
#         max_gen_toks=64,
#     ):
#         super().__init__()

#         self.model = model
#         self.tokenizer = tokenizer
#         self._device = device
#         self._batch_size = batch_size
#         self._max_gen_toks = max_gen_toks

#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = (
#                 self.tokenizer.eos_token_id
#             )

#     @property
#     def eot_token_id(self):
#         return self.tokenizer.eos_token_id

#     @property
#     def max_length(self):
#         return self.model.cfg.max_seq_len

#     @property
#     def max_gen_toks(self):
#         return self._max_gen_toks

#     @property
#     def batch_size(self):
#         return self._batch_size

#     @property
#     def device(self):
#         return self._device

#     def tok_encode(
#         self,
#         string,
#         left_truncate_len=None,
#         add_special_tokens=False,
#     ):
#         tokens = self.tokenizer.encode(
#             string,
#             add_special_tokens=False,
#         )

#         if left_truncate_len:
#             tokens = tokens[-left_truncate_len:]

#         return tokens

#     def tok_decode(self, tokens):
#         return self.tokenizer.decode(tokens)

#     @torch.inference_mode()
#     def _model_call(self, inps):
#         inps = inps.to(self.device)

#         output = self.model(inps)

#         return output["logits"]

#     @torch.inference_mode()
#     def _model_generate(
#         self,
#         context,
#         max_length,
#         stop,
#         **generation_kwargs,
#     ):
#         tokens = context.to(self.device)

#         while tokens.shape[1] < max_length:

#             current = tokens[:, -self.max_length:]

#             output = self.model(current)

#             next_token = output[
#                 "logits"
#             ][:, -1].argmax(
#                 dim=-1,
#                 keepdim=True,
#             )

#             tokens = torch.cat(
#                 [tokens, next_token],
#                 dim=1,
#             )

#             if (
#                 self.eot_token_id is not None
#                 and torch.all(
#                     next_token
#                     == self.eot_token_id
#                 )
#             ):
#                 break

#         return tokens



import torch
import torch.nn.functional as F

from lm_eval.api.model import TemplateLM


class MoREvalAdapter(TemplateLM):

    def __init__(self, model, tokenizer, device, batch_size=1, max_gen_toks=64):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


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


    def tok_encode(self, string, left_truncate_len=None, **kwargs):
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
    def _model_call(self, tokens):
        return self.model(
            tokens.to(self.device)
        )["logits"]


    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):

        results = []

        for cache_key, context, continuation in requests:

            context = context or [self.eot_token_id]

            context = context[
                -(self.max_length - len(continuation) + 1):
            ]

            sequence = context + continuation

            inputs = torch.tensor(
                sequence[:-1],
                device=self.device,
            ).unsqueeze(0)

            logits = self._model_call(inputs)
            logits = logits[:, -len(continuation):]

            targets = torch.tensor(
                continuation,
                device=self.device,
            ).unsqueeze(0)

            log_probs = F.log_softmax(
                logits.float(),
                dim=-1,
            )

            score = torch.gather(
                log_probs,
                -1,
                targets.unsqueeze(-1),
            ).sum().item()

            greedy = torch.equal(
                logits.argmax(-1),
                targets,
            )

            results.append(
                (score, greedy)
            )

        return results


    def loglikelihood_rolling(self, requests, disable_tqdm=False):

        results = []

        for request in requests:

            tokens = self.tok_encode(
                request.args[0]
            )

            total = 0.0

            for i in range(1, len(tokens)):

                context = tokens[
                    max(0, i - self.max_length + 1):i
                ]

                score, _ = self._loglikelihood_tokens([
                    (
                        None,
                        context,
                        [tokens[i]],
                    )
                ])[0]

                total += score

            results.append(total)

        return results


    @torch.inference_mode()
    def generate_until(self, requests, disable_tqdm=False):

        results = []

        for request in requests:

            context, kwargs = request.args

            stop = kwargs.get("until", [])
            stop = [stop] if isinstance(stop, str) else stop

            tokens = self.tok_encode(context)
            generated = []

            for _ in range(
                kwargs.get(
                    "max_gen_toks",
                    self.max_gen_toks,
                )
            ):

                current = torch.tensor(
                    tokens[-self.max_length:],
                    device=self.device,
                ).unsqueeze(0)

                logits = self._model_call(current)

                next_token = int(
                    logits[:, -1]
                    .argmax(-1)
                    .item()
                )

                tokens.append(next_token)
                generated.append(next_token)

                text = self.tok_decode(generated)

                if next_token == self.eot_token_id:
                    break

                if any(x in text for x in stop if x):
                    break

            text = self.tok_decode(generated)

            for x in stop:
                if x in text:
                    text = text.split(x)[0]

            results.append(text)

        return results