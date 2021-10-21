from benchmarks.dataset.esc50.esc50 import ESC50_v2, ESC50
import os
from typing import Optional, Sequence, Union, Any
from pathlib import Path
from avalanche.benchmarks import ni_benchmark

def CLEsc50(n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        shuffle: bool = True,
        balance_experiences=True,
        dataset_root: Union[str, Path] = None):

    train_dataset = ESC50_v2(root= os.path.abspath(os.getcwd()),download =True, data_aug=True, train=True)
    test_dataset = ESC50_v2(root= os.path.abspath(os.getcwd()),download =True, data_aug=False, train=False)

    
    return ni_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_experiences,
            task_labels=return_task_id,
            seed=seed,
            shuffle=shuffle,
            balance_experiences = balance_experiences)
    

def CLEsc50_v2(n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        shuffle: bool = True,
        balance_experiences=True,
        dataset_root: Union[str, Path] = None):

    train_dataset = ESC50(root= os.path.abspath(os.getcwd()),download =True, data_aug=True, train=True)
    test_dataset = ESC50(root= os.path.abspath(os.getcwd()),download =True, data_aug=False, train=False)

    return ni_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_experiences,
            task_labels=return_task_id,
            seed=seed,
            shuffle=shuffle,
            balance_experiences = balance_experiences)