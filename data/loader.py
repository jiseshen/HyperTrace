from data.base import UserData
from typing import List
from .prism import load_prism


def load_data(dataset_name: str, n_users: int = None, seed: int = 42) -> List[UserData]:
    if dataset_name == "prism":
        return load_prism(n_users=n_users, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")