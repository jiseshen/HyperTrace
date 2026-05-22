from typing import Dict, List, Optional, Sequence, Tuple, Union, overload
from collections import OrderedDict
from dataclasses import dataclass, asdict
import logging
import faiss
import numpy as np
from model import embed, EmbedConfig

logger = logging.getLogger(__name__)

@dataclass
class Hypothesis:
    id: str
    category: str
    content: str
    
    def format(self, include_category: bool = True) -> str:
        if include_category:
            return f"ID: {self.id}\nCategory: {self.category}\nContent: {self.content}\n"
        return f"ID: {self.id}\nContent: {self.content}\n"
    
@dataclass
class Update:
    id: str
    category: Optional[str] = None
    content: Optional[str] = None
    likelihood: float = None
    

class VectorStore:
    """
    Simple FAISS vector store.

    Stores full hypotheses and retrieves them by vector similarity.
    """
    def __init__(
        self,
        *,
        embed_cfg: EmbedConfig,
        metric: str = "ip",
        capacity: int = 1000,
        use_keys: bool = False,
    ) -> None:
        """
        Args:
            dim: Embedding dimension.
            metric: "ip" or "l2".
        """
        self.backend = embed_cfg.backend
        self.model = embed_cfg.model
        self.embed_cfg = embed_cfg
        self.dim = embed_cfg.dim
        self.metric = metric
        self.max_memories = capacity
        self.use_keys = use_keys

        if metric == "ip":
            base = faiss.IndexFlatIP(self.dim)
        elif metric == "l2":
            base = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError("Vector store metric must be 'ip' or 'l2'.")

        self.index = faiss.IndexIDMap2(base)

        self.contents: Dict[int, str] = {}
        if self.use_keys:
            self.key2id: Dict[str, int] = {}
            self.id2key: Dict[int, str] = {}
        
        self.lru_order = OrderedDict()
        self.next_id: int = 0

    def store(
        self,
        contents: List[str],
        keys: Optional[List[str]] = None,
    ) -> None:
        """
        Store a list of new hypotheses.

        Args:
            hypothesis: Hypothesis dict.
        """
        if self.use_keys:
            if keys is None:
                raise ValueError("Keys must be provided when use_keys is True.")
            for k in keys:
                if k in self.key2id:
                    raise ValueError(f"Key {k} already exists in vector store.")
        while len(self.contents) + len(contents) >= self.max_memories:
            self.delete(self.lru_order.popitem(last=False)[0])
        
        vec = embed(contents, embed_cfg=self.embed_cfg)
        indices = []
        for i, content in enumerate(contents):
            idx = self.next_id
            self.next_id += 1
            indices.append(idx)
            if self.use_keys:
                key = keys[i]
                self.key2id[key] = idx
                self.id2key[idx] = key
            self.contents[idx] = content
            self.lru_order[idx] = None
            self.lru_order.move_to_end(idx)
        self.index.add_with_ids(vec, np.asarray(indices, dtype=np.int64))
            

    def retrieve(
        self,
        content: str,
        top_k: int = 5,
        return_keys: bool = False,
        exclude_ids: Optional[List[str]] = None
    ) -> Tuple[List[Union[str, int]], List[float]]:
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
        vec = embed(content, embed_cfg=self.embed_cfg)
        k = min(top_k, len(self.contents))
        if exclude_ids:
            exclude_indices = set(self.get_index(exclude_ids))
            exclude_indices.discard(None)
            sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(list(exclude_indices)))
            scores, indices = self.index.search(vec, k, params=faiss.SearchParameters(sel=sel))
        else:    
            scores, indices = self.index.search(vec, k)
        retrieved_hypotheses: List[Union[str, int]] = []
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

    def get_index(self, id: int | str | List[int] | List[str]) -> int | List[int]:
        if isinstance(id, list):
            return [self.get_index(i) for i in id]
        if self.use_keys and isinstance(id, str):
            if id in self.key2id:
                return self.key2id[id]
            else:
                return None
        else:
            return int(id)

    def delete(self, id: int | str) -> None:
        idx = self.get_index(id)
        if idx is None or idx not in self.contents:
            print(f"ID {id} not found in vector store. Continue...")
            return
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

    def update(self, ids: List[int] | List[str], contents: List[str]) -> None:
        idx = self.get_index(ids)
        for i, content in zip(idx, contents):
            if i is None or i not in self.contents:
                print(f"ID {i} not found in vector store.")
                continue
            self.contents[i] = content
            self.lru_order.move_to_end(i)
        self.index.remove_ids(np.asarray(idx, dtype=np.int64))
        vec = embed(contents, embed_cfg=self.embed_cfg)
        self.index.add_with_ids(vec, np.asarray(idx, dtype=np.int64))

    def similarity(self, id1: int | str, id2: int | str) -> Optional[float]:
        idx1 = self.get_index(id1)
        idx2 = self.get_index(id2)
        if any(i is None or i not in self.contents for i in [idx1, idx2]):
            print(f"One of the IDs {id1}, {id2} not found in vector store.")
            return None
        vec1 = np.zeros((1, self.dim), dtype=np.float32)
        vec2 = np.zeros((1, self.dim), dtype=np.float32)
        self.index.reconstruct(idx1, vec1[0])
        self.index.reconstruct(idx2, vec2[0])
        if self.metric == "ip":
            return float((vec1 @ vec2.T).item())
        elif self.metric == "l2":
            diff = vec1 - vec2
            return float(np.exp(-diff @ diff.T).item())

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
        n_hypotheses: int,
        embed_config: EmbedConfig,
        use_topics: bool = True,
    ) -> None:
        self.vector_store = VectorStore(
            embed_cfg=embed_config,
            use_keys=True,
        )
        self.global_prior: Dict[str, float] = {}
        self.base_prior: float = 1.0 / n_hypotheses
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.category_counts: Dict[str, int] = {}
        self.next_hypothesis_id: int = 1
        self.use_topics = use_topics

    def _embedding_text(self, hypothesis: Hypothesis) -> str:
        if not self.use_topics:
            return hypothesis.content
        return f"Category: {hypothesis.category}\nContent: {hypothesis.content}"

    def _allocate_hid(self) -> str:
        while True:
            hid = f"h{self.next_hypothesis_id}"
            self.next_hypothesis_id += 1
            if hid not in self.hypotheses:
                return hid
    
    @overload
    def __getitem__(self, idx: str) -> Tuple[Hypothesis, float]: ...
    
    @overload
    def __getitem__(self, idx: List[str]) -> Tuple[List[Hypothesis], List[float]]: ...
    
    def __getitem__(self, key: Union[str, List[str]]) -> Union[Tuple[Hypothesis, float], Tuple[List[Hypothesis], List[float]]]:
        if isinstance(key, list):
            return [self.hypotheses[k] for k in key], [self.global_prior[k] for k in key]
        else:
            return self.hypotheses[key], self.global_prior[key]
    
    def add_hypotheses(self, hypotheses: List[Union[Dict, Hypothesis]]) -> List[str]:
        hids = []
        contents = []
        for hyp in hypotheses:
            category = hyp.category if isinstance(hyp, Hypothesis) else hyp["category"]
            content = hyp.content if isinstance(hyp, Hypothesis) else hyp["content"]
            prior = None if isinstance(hyp, Hypothesis) else hyp.get("prior")
            cat = category if self.use_topics else ""
            if self.use_topics:
                self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
            hid = self._allocate_hid()
            h = Hypothesis(
                id=hid,
                category=cat,
                content=content
            )
            self.hypotheses[hid] = h
            if prior is not None:
                self.global_prior[hid] = prior
            else:
                self.global_prior[hid] = self.base_prior
            contents.append(self._embedding_text(h))
            hids.append(hid)
        if contents:
            self.vector_store.store(contents, keys=hids)
        return hids
            
    def retrieve_hypotheses(
        self,
        query: Union[str, List[str]],
        top_k: int = 5,
        exclude_ids: Optional[List[str]] = None
    ) -> Tuple[List[Hypothesis], List[float]]:
        if isinstance(query, list):
            hypotheses = [self.hypotheses[hid] for hid in query]
            priors = [self.global_prior.get(hid, self.base_prior) for hid in query]
            return hypotheses, priors
        retrieved_ids, _ = self.vector_store.retrieve(query, top_k=top_k, return_keys=True, exclude_ids=exclude_ids)
        hypotheses = [self.hypotheses[hid] for hid in retrieved_ids]
        priors = [self.global_prior.get(hid, self.base_prior) for hid in retrieved_ids]
        return hypotheses, priors

    def retrieve_hypotheses_with_scores(
        self,
        query: str,
        top_k: int = 5,
        exclude_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Hypothesis], List[float], List[float]]:
        retrieved_ids, scores = self.vector_store.retrieve(
            query,
            top_k=top_k,
            return_keys=True,
            exclude_ids=exclude_ids,
        )
        hypotheses = [self.hypotheses[hid] for hid in retrieved_ids]
        priors = [self.global_prior.get(hid, self.base_prior) for hid in retrieved_ids]
        return hypotheses, priors, scores
    
    def update_hypotheses(
        self,
        updates: List[Union[Update, Hypothesis]]
    ) -> None:
        changed_ids = []
        changed_contents = []
        for update in updates:
            hid = update.id
            hyp = self.hypotheses[hid]
            
            if self.use_topics and update.category is not None:
                hyp.category = update.category
            if update.content is not None:
                if hyp.content != update.content:
                    changed_ids.append(hid)
                hyp.content = update.content
            if self.use_topics and hid not in changed_ids and update.category is not None:
                changed_ids.append(hid)
            if hid in changed_ids:
                changed_contents.append(self._embedding_text(hyp))
        if changed_ids:
            self.vector_store.update(changed_ids, changed_contents)
        
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
            del self.global_prior[key]
            
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

    def replace_belief_priors(
        self,
        ids: List[str],
        weights: np.ndarray,
    ) -> None:
        for hid, weight in zip(ids, weights):
            if hid in self.global_prior:
                self.global_prior[hid] = float(weight)
    
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
        normalized_priors = {hid: prior / prior_sum for hid, prior in self.global_prior.items()}
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
        self._refresh_indices()
        self.weights = np.array(priors, dtype=np.float32)
        self.weights /= (self.weights.sum() + 1e-14)
        self.repo = repo

    def _refresh_indices(self) -> None:
        self.hid2indices: Dict[str, List[int]] = {}
        for i, hid in enumerate(self.ids):
            self.hid2indices.setdefault(hid, []).append(i)
    
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
        local_positions = {hid: list(indices) for hid, indices in self.hid2indices.items()}
        for update in updates:
            if update.likelihood is not None:
                i = local_positions[update.id].pop(0)
                self.weights[i] *= update.likelihood
        self.weights /= (self.weights.sum() + 1e-14)
        self.repo.update_hypotheses(updates)
    
    def ess(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))
    
    def normalized_entropy(self) -> float:
        return float(-np.sum(self.weights * np.log(self.weights + 1e-14)) / np.log(len(self.weights) + 1e-14))
    
    def resample(self) -> List[List[int]]:
        new_ids: np.ndarray = np.random.choice(self.ids, size=len(self.ids), replace=True, p=self.weights)
        self.ids = new_ids.tolist()
        self._refresh_indices()
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
        
    def log_dict(self, include_category: bool = True) -> List[Dict[str, Union[str, float]]]:
        items = []
        for hid, weight in zip(self.ids, self.weights):
            item = asdict(self.repo.hypotheses[hid])
            if not include_category:
                item.pop("category", None)
            item["weight"] = float(weight)
            items.append(item)
        return items
