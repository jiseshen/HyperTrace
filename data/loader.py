from data.base import UserData
from typing import List


def load_data(dataset_name: str, n_users: int = None, seed: int = 42) -> List[UserData]:
    if dataset_name == "prism":
        from .prism import load_prism
        return load_prism(n_users=n_users, seed=seed)
    elif dataset_name == "personamem_v2":
        from .personamem_v2 import load_personamem_v2
        return load_personamem_v2(n_users=n_users, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
