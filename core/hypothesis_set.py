from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload
from collections import OrderedDict
from dataclasses import dataclass
import logging
import faiss
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Hypothesis:
    id: str
    category: str
    content: str
    
    def format(self) -> str:
        return f"ID: {self.id}\nCategory: {self.category}\nContent: {self.content}\n"
    
@dataclass
class Update:
    id: str
    category: Optional[str] = None
    content: Optional[str] = None
    likelihood: float = None
    
@dataclass
class RepoConfig:
    backend: str = "openai"
    model: str = "text-embedding-3-small"
    dim: int = 1536
    metric: str = "ip"
    capacity: int = 1000

def embed(
    text: str,
    *,
    backend: str = "openai",
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """
    Embed an experience into a vector.

    Args:
        experience: Experience dict or string to embed.
        backend: "openai" or "transformer".
        model: Embedding model name.

    Returns:
        1D numpy float32 vector.
    """
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float",
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.reshape(1, -1)

    elif "transformer" in backend:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model)
        encoder = AutoModel.from_pretrained(model)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.inference_mode():
            outputs = encoder(**inputs)

        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        vec = pooled[0].cpu().numpy().astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.reshape(1, -1)

    else:
        raise ValueError(f"Unknown embedding backend: {backend}")


class VectorStore:
    """
    Simple FAISS vector store.

    Stores full hypotheses and retrieves them by vector similarity.
    """
    def __init__(
        self,
        *,
        backend: str = "openai",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        metric: str = "ip",
        capacity: int = 1000,
        use_keys: bool = False,
    ) -> None:
        """
        Args:
            dim: Embedding dimension.
            metric: "ip" or "l2".
        """
        self.backend = backend
        self.model = model
        self.dim = dim
        self.metric = metric
        self.max_memories = capacity
        self.use_keys = use_keys

        if metric == "ip":
            base = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            base = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("metric must be 'ip' or 'l2'.")

        self.index = faiss.IndexIDMap2(base)

        self.contents: Dict[int, Dict[str, Any]] | Dict[int, str] = {}
        if self.use_keys:
            self.key2id: Dict[str, int] = {}
            self.id2key: Dict[int, str] = {}
        
        self.lru_order = OrderedDict()
        self.next_id: int = 0

    def store(
        self,
        content: str,
        key: Optional[str] = None,
    ) -> None:
        """
        Store a new hypothesis.

        Args:
            hypothesis: Hypothesis dict.
        """
        idx = self.next_id
        if self.use_keys:
            if key is None:
                raise ValueError("Key must be provided when use_keys is True.")
            if key in self.key2id:
                logger.warning(f"Key {key} already exists. Updating existing entry.")
                self.update(key, content)
                return
            self.key2id[key] = idx
            self.id2key[idx] = key
        self.next_id += 1
        
        if len(self.contents) >= self.max_memories:
            self.delete(self.lru_order.popitem(last=False)[0])
        
        vec = embed(
            content,
            backend=self.backend,
            model=self.model,
        )

        self.index.add_with_ids(vec, np.asarray([idx], dtype=np.int64))
        self.contents[idx] = content
        self.lru_order[idx] = None
        self.lru_order.move_to_end(idx)

    def retrieve(
        self,
        content: str,
        top_k: int = 5,
        return_keys: bool = False,
    ) -> Tuple[List[str | int], List[float]]:
        """
        Retrieve top-k most similar hypotheses.

        Args:
            hypothesis: Query hypothesis string.
            top_k: Maximum number of results.

        Returns:
            List of most related hypotheses, and their similarity scores.
        """
        if len(self.contents) == 0:
            return [], []
        vec = embed(
            content,
            backend=self.backend,
            model=self.model
        )
        k = min(top_k, len(self.contents))
        scores, indices = self.index.search(vec, k)
        retrieved_hypotheses: List[str] = []
        retrieval_scores: List[float] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            self.lru_order.move_to_end(int(idx))
            if return_keys:
                retrieved_hypotheses.append(self.id2key[int(idx)] if self.use_keys else int(idx))
            else:
                retrieved_hypotheses.append(self.contents[int(idx)])
            if self.metric == "l2":
                retrieval_scores.append(float(np.exp(-score)))
            else:
                retrieval_scores.append(float(score))
        return retrieved_hypotheses, retrieval_scores

    def get_index(self, id: int | str) -> int:
        if self.use_keys and isinstance(id, str):
            if id in self.key2id:
                return self.key2id[id]
            else:
                raise ValueError(f"Key {id} not found in vector store.")
        else:
            return int(id)

    def delete(self, id: int | str) -> None:
        idx = self.get_index(id)
        if idx not in self.contents:
            raise ValueError(f"ID {id} not found in vector store.")
        self.index.remove_ids(np.asarray([idx], dtype=np.int64))
        self.contents.pop(idx)
        if self.use_keys:
            if isinstance(id, str):
                self.key2id.pop(id)
                self.id2key.pop(idx)
            else:
                key = self.id2key.pop(idx)
                self.key2id.pop(key)
        self.lru_order.pop(idx, None)

    def update(self, id: int | str, content: str) -> None:
        idx = self.get_index(id)
        if idx not in self.contents:
            raise ValueError(f"ID {id} not found in vector store.")
        vec = embed(
            content,
            backend=self.backend,
            model=self.model,
        )
        self.index.remove_ids(np.asarray([idx], dtype=np.int64))
        self.index.add_with_ids(vec, np.asarray([idx], dtype=np.int64))
        self.contents[idx] = content
        self.lru_order.move_to_end(idx)

    def similarity(self, id1: int | str, id2: int | str) -> float:
        idx1 = self.get_index(id1)
        idx2 = self.get_index(id2)
        vec1 = np.zeros((1, self.dim), dtype=np.float32)
        vec2 = np.zeros((1, self.dim), dtype=np.float32)
        self.index.reconstruct(idx1, vec1[0])
        self.index.reconstruct(idx2, vec2[0])
        if self.metric == "ip":
            return float(np.dot(vec1, vec2.T))
        elif self.metric == "l2":
            diff = vec1 - vec2
            return float(np.exp(-np.dot(diff, diff)))
        
    def similarity_all(self) -> Tuple[np.ndarray, Optional[List[str]]]:
        n = len(self.contents)
        X = np.zeros((n, self.dim), dtype=np.float32)
        id_list = []
        if self.use_keys:
            for i, id in enumerate(self.id2key.keys()):
                self.index.reconstruct(id, X[i])
                id_list.append(self.id2key[id])
        else:
            for i, id in enumerate(self.contents.keys()):
                self.index.reconstruct(id, X[i])
                id_list.append(id)
        if self.metric == "ip":
            sim_matrix = np.dot(X, X.T)
        elif self.metric == "l2":
            sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
            sim_matrix = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
            sim_matrix = np.exp(-sim_matrix)
        return sim_matrix, id_list

    def clear(self) -> None:
        self.index.reset()
        self.contents.clear()
        if self.use_keys:
            self.key2id.clear()
            self.id2key.clear()
        self.lru_order.clear()
        self.next_id = 0
        
        
class HypothesisSet:
    def __init__(
        self,
        *,
        cfg: Optional[RepoConfig] = None
    ) -> None:
        cfg = cfg or RepoConfig()
        self.vector_store = VectorStore(
            backend=cfg.backend,
            model=cfg.model,
            dim=cfg.dim,
            metric=cfg.metric,
            capacity=cfg.capacity,
            use_keys=True,
        )
        self.global_prior: Dict[str, float] = {}
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.category_counts: Dict[str, int] = {}
    
    @overload
    def __getitem__(self, idx: str) -> Tuple[Hypothesis, float]: ...
    
    @overload
    def __getitem__(self, idx: List[str]) -> Tuple[List[Hypothesis], List[float]]: ...
    
    def __getitem__(self, key: Union[str, List[str]]) -> Union[Tuple[Hypothesis, float], Tuple[List[Hypothesis], List[float]]]:
        if isinstance(key, list):
            return [self.hypotheses[k] for k in key], [self.global_prior[k] for k in key]
        else:
            return self.hypotheses[key], self.global_prior[key]
    
    def add_hypotheses(
        self,
        hypotheses: List[Dict]
    ):
        hids = []
        for hyp in hypotheses:
            cat = hyp['category']
            self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
            hid = cat[:3] + str(self.category_counts[cat])
            h = Hypothesis(
                id=hid,
                category=cat,
                content=hyp['content']
            )
            self.hypotheses[hid] = h
            if 'prior' in hyp:
                self.global_prior[hid] = hyp['prior']
            else:
                self.global_prior[hid] = 1.0
            self.vector_store.store(h.content, key=hid)
            hids.append(hid)
        return hids
            
    def retrieve_hypotheses(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Hypothesis]:
        retrieved, _ = self.vector_store.retrieve(query, top_k=top_k, return_keys=True)
        priors = [self.global_prior.get(hid) for hid in retrieved]
        return retrieved, priors
    
    def update_hypotheses(
        self,
        updates: List[Union[Update, Hypothesis]]
    ):
        for update in updates:
            hid = update.id
            hyp = self.hypotheses[hid]
            if update.category is not None:
                hyp.category = update.category
            if update.content is not None:
                hyp.content = update.content
                self.vector_store.update(hid, hyp.content)
            
    def get_similarity(
        self,
        key1: str,
        key2: str
    ):
        return self.vector_store.similarity(key1, key2)
    
    def remove_hypothesis(
        self,
        key: str
    ):
        if key in self.hypotheses:
            del self.hypotheses[key]
            self.vector_store.delete(key)
            
    def get_similarity_groups(self, threshold: float = 0.8) -> List[List[str]]:
        sim_matrix, id_list = self.vector_store.similarity_all()
        similar_groups = []
        checked = set()
        for i, hid in enumerate(id_list):
            if hid in checked:
                continue
            group = [hid]
            checked.add(hid)
            has_similar = False
            for j, other_hid in enumerate(id_list[i+1:], i + 1):
                if other_hid not in checked and sim_matrix[i, j] > threshold:
                    group.append(other_hid)
                    checked.add(other_hid)
                    has_similar = True
            if has_similar:
                similar_groups.append(group)
        return similar_groups
            
    def consolidate_belief(
        self,
        ids: List[str],
        weights: np.ndarray,
        importance: float = 0.5,  # importance of current conversation, in terms of valid length or other heuristics
        alpha: float = 0.5
    ):
        for hid, w in zip(ids, weights):
            prev_prior = self.global_prior.get(hid)
            self.global_prior[hid] = prev_prior * (1 - alpha * importance) + w * alpha * importance
    
    def merge_hypotheses(
        self,
        ids: List[str],
        merged_hypothesis: Dict[str, str]
    ):
        priors = [self.global_prior[hid] for hid in ids]
        total_prior = sum(priors)
        for hid in ids:
            self.remove_hypothesis(hid)
        self.add_hypotheses([{"category": merged_hypothesis['category'], "content": merged_hypothesis['content'], "prior": total_prior}])
    
    def top_p_retrieve(self, p: float = 0.8, max_k: int = 10) -> List[Hypothesis]:
        prior_sum = sum(self.global_prior.values())
        if prior_sum == 0:
            return []
        normalized_priors = {hid: prior / prior_sum for hid, prior in self.global_prior}
        sorted_hids = sorted(normalized_priors, key=normalized_priors.get, reverse=True)
        cumulative_prob = 0.0
        selected_hids = []
        for hid in sorted_hids[:max_k]:
            cumulative_prob += normalized_priors[hid]
            selected_hids.append(hid)
            if cumulative_prob >= p:
                break
        return [self.hypotheses[hid] for hid in selected_hids]


class WorkingBelief:
    def __init__(
        self,
        ids: List[str],
        priors: List[float],
        repo: HypothesisSet
    ):
        self.ids = ids
        self.weights = np.array(priors, dtype=np.float32)
        self.weights /= (self.weights.sum() + 1e-14)
        self.repo = repo
    
    @overload
    def __getitem__(self, idx: int) -> Tuple[Hypothesis, float]: ...
    
    @overload
    def __getitem__(self, idx: Union[slice, Sequence[int]]) -> Tuple[List[Hypothesis], np.ndarray]: ...
    
    def __getitem__(
        self,
        idx: Union[int, slice, Sequence[int]],
    ) -> Union[Tuple[Hypothesis, float], Tuple[List[Hypothesis], np.ndarray]]:
        if isinstance(idx, int):
            hid = self.ids[idx]
            return self.repo.hypotheses[hid], float(self.weights[idx])
        if isinstance(idx, slice):
            hids = self.ids[idx]
            weights = self.weights[idx]
        else:
            idx_list = list(idx)
            hids = [self.ids[i] for i in idx_list]
            weights = self.weights[idx_list]
        hyps = [self.repo.hypotheses[hid] for hid in hids]
        return hyps, weights
    
    def get_hypotheses(self) -> List[Hypothesis]:
        return [self.repo.hypotheses[hid] for hid in self.ids]
    
    def update(self, updates: List[Update]):
        if all(update.likelihood is not None for update in updates):
            for i, update in enumerate(updates):
                self.weights[i] *= update.likelihood
        self.weights /= (self.weights.sum() + 1e-14)
        self.repo.update_hypotheses(updates)
    
    def ess(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)
    
    def normalized_entropy(self) -> float:
        return -np.sum(self.weights * np.log(self.weights + 1e-14)) / np.log(len(self.weights) + 1e-14)
    
    def resample(self) -> List[List[int]]:
        new_ids: np.ndarray = np.random.choice(self.ids, size=len(self.ids), replace=True, p=self.weights)
        self.ids = new_ids.tolist()
        self.weights = np.ones_like(self.weights) / len(self.ids)
        
        pos = {}
        for i, hid in enumerate(self.ids):
            pos.setdefault(hid, []).append(i)
        return list(pos.values())
    
    def get_similarity_groups(self, threshold: float = 0.8) -> List[List[int]]:
        similar_groups = []
        checked = set()
        for i, hid in enumerate(self.ids):
            if i in checked:
                continue
            group = [i]
            checked.add(i)
            for j, other_hid in enumerate(self.ids[i+1:], i+1):
                if j not in checked and self.repo.get_similarity(hid, other_hid) > threshold:
                    group.append(j)
                    checked.add(j)
            similar_groups.append(group)
        return similar_groups
    
    def consolidate(self, importance: float = 0.5, alpha: float = 0.5):
        self.repo.consolidate_belief(self.ids, self.weights, importance=importance, alpha=alpha)