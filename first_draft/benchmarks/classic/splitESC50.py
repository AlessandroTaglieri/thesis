from dataset.esc50 import ESC50_v2
import os
from typing import Optional, Sequence, Union, Any
from pathlib import Path
from avalanche.benchmarks import nc_benchmark

def CLEsc50(n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        dataset_root: Union[str, Path] = None):

    train_dataset = ESC50_v2(root= os.path.abspath(os.getcwd()),download =True, data_aug=True, train=True)
    test_dataset = ESC50_v2(root= os.path.abspath(os.getcwd()),download =True, data_aug=False, train=False)

    if return_task_id:
        return nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True)
    else:
        return nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle)