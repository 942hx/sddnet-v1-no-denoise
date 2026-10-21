"""Serializable shuffled anchor order; every valid low image appears exactly once per epoch."""
from __future__ import annotations
import torch
from torch.utils.data import Sampler


class AnchorSampler(Sampler):
    """Default workers=0 also permits exact cursor restoration during short smoke tests."""
    def __init__(self, size, seed=42):
        self.size = int(size)
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.position, self.epoch = [], 0, 0

    def start_epoch(self, epoch):
        if self.epoch != epoch:
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
            self.position, self.epoch = 0, epoch

    def __iter__(self):
        if not self.order:
            self.start_epoch(1)
        while self.position < len(self.order):
            index = self.order[self.position]
            self.position += 1
            yield index

    def __len__(self):
        return self.size

    def state_dict(self):
        return dict(size=self.size, order=list(self.order), position=self.position,
                    epoch=self.epoch, generator=self.generator.get_state())

    def load_state_dict(self, state):
        if state["size"] != self.size or sorted(state["order"]) != list(range(self.size)):
            raise ValueError("Sampler state does not match the audited anchor set.")
        self.order, self.position, self.epoch = list(state["order"]), int(state["position"]), int(state["epoch"])
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid sampler cursor.")
        self.generator.set_state(state["generator"])
