from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset


class PackedFineWebDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        dataset_config: str,
        seq_len: int,
        mode: str = "train",
        validation_docs: int = 2000,
        shuffle_buffer: int = 5000,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.mode = mode
        self.validation_docs = validation_docs
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _stream(self):
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
            streaming=True,
        )
        if self.mode == "validation":
            return ds.take(self.validation_docs)
        ds = ds.skip(self.validation_docs)
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        return ds

    def __iter__(self) -> Iterator[torch.Tensor]:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.bos_token_id
        buffer = []
        target_len = self.seq_len + 1

        for row in self._stream():
            text = row.get("text", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(eos)
            buffer.extend(ids)

            while len(buffer) >= target_len:
                chunk = buffer[:target_len]
                buffer = buffer[target_len:]
                yield torch.tensor(chunk, dtype=torch.long)
