from .prism import load_prism


def load_data(dataset_name: str, n_users: int = None):
    if dataset_name == "prism":
        return load_prism(n_users=n_users)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")