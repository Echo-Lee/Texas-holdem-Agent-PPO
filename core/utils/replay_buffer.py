import random

import torch


PPO_KEYS = ["states", "actions", "log_probs", "rewards", "dones", "masks"]


def clone_batch_to_cpu(batch_tensors):
    return {
        key: value.detach().cpu().clone()
        for key, value in batch_tensors.items()
        if key in PPO_KEYS
    }


def move_batch_to_device(batch_tensors, device):
    return {key: value.to(device) for key, value in batch_tensors.items()}


def merge_batches(batches, device):
    merged = {}
    for key in PPO_KEYS:
        merged[key] = torch.cat([batch[key].to(device) for batch in batches], dim=0)
    return merged


class ReplayBatchBuffer:
    def __init__(self, max_stage1_batches=0, max_stage2_batches=0):
        self.max_stage1_batches = max_stage1_batches
        self.max_stage2_batches = max_stage2_batches
        self.stage1_batches = []
        self.stage2_batches = []

    def _trim(self, items, max_items):
        if max_items <= 0:
            items.clear()
            return
        while len(items) > max_items:
            items.pop(0)

    def add_stage1_batch(self, batch_tensors):
        if self.max_stage1_batches <= 0:
            return
        self.stage1_batches.append(clone_batch_to_cpu(batch_tensors))
        self._trim(self.stage1_batches, self.max_stage1_batches)

    def add_stage2_batch(self, batch_tensors):
        if self.max_stage2_batches <= 0:
            return
        self.stage2_batches.append(clone_batch_to_cpu(batch_tensors))
        self._trim(self.stage2_batches, self.max_stage2_batches)

    def sample(self, num_stage1_batches=0, num_stage2_batches=0, device="cpu"):
        batches = []

        if num_stage1_batches > 0 and self.stage1_batches:
            sample_count = min(num_stage1_batches, len(self.stage1_batches))
            batches.extend(random.sample(self.stage1_batches, sample_count))

        if num_stage2_batches > 0 and self.stage2_batches:
            sample_count = min(num_stage2_batches, len(self.stage2_batches))
            batches.extend(random.sample(self.stage2_batches, sample_count))

        return [move_batch_to_device(batch, device) for batch in batches]
