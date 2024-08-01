# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple
import math
import os
import gin

import torch

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
    
@gin.configurable
def create_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    shuffle: bool,
    prefetch_factor: int = 128,
    num_workers: int = os.cpu_count(),
    drop_last: bool = False,
) -> Tuple[Optional[torch.utils.data.distributed.DistributedSampler], torch.utils.data.DataLoader]:
    # print(f"num_workers={num_workers}")
    if isinstance(dataset, torch.utils.data.IterableDataset):
        # For IterableDataset, we manually handle data partitioning and don't use a sampler.
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
        )
    else:
        if shuffle:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=0,
                drop_last=drop_last,
            )
        else:
            sampler = None
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            prefetch_factor=prefetch_factor,
        )
    return sampler, data_loader