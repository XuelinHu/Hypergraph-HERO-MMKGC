from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


class KGDataset:
    """Loader for common MMKGC benchmark layouts used by DB15K, MKG-W, and MKG-Y."""

    def __init__(self, root):
        self.root = Path(root)
        # Mapping files are optional because some public benchmark packages store
        # triples directly as contiguous numeric IDs.
        self.entity2id = self._load_mapping("entity2id.txt")
        self.relation2id = self._load_mapping("relation2id.txt")

        # All splits may extend the mapping when string IDs are used, which keeps
        # demo data and raw benchmark text files compatible with the same loader.
        self.train = self._load_triples(["train.txt", "train2id.txt"], build_mapping=True)
        self.valid = self._load_triples(["valid.txt", "valid2id.txt"], build_mapping=True)
        self.test = self._load_triples(["test.txt", "test2id.txt"], build_mapping=True)
        self._finalize_numeric_mappings()

        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)
        self.all_true = set(self.train) | set(self.valid) | set(self.test)
        self.text_features = self._load_features(["text_features.npy", "text_features.pt", "txt_features.npy", "txt_features.pt"])
        self.visual_features = self._load_features(["visual_features.npy", "visual_features.pt", "img_features.npy", "img_features.pt"])
        self.adj = self._build_adjacency(self.train)

    def _load_mapping(self, filename):
        path = self.root / filename
        mapping = {}
        if not path.exists():
            return mapping
        with open(path, "r", encoding="utf-8") as f:
            # Support both "name id" and "id name" layouts, with or without an
            # initial count line.
            first = f.readline().strip().split()
            rows = f.readlines()
            if len(first) == 2:
                rows.insert(0, " ".join(first) + "\n")
            for line in rows:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                if parts[0].isdigit():
                    idx, name = int(parts[0]), parts[1]
                else:
                    name, idx = parts[0], int(parts[1])
                mapping[name] = idx
        return mapping

    def _id(self, token, mapping):
        if token.isdigit():
            return int(token)
        if token not in mapping:
            mapping[token] = len(mapping)
        return mapping[token]

    def _load_triples(self, names, build_mapping):
        path = next((self.root / name for name in names if (self.root / name).exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing triple file under {self.root}: {names}")
        triples = []
        with open(path, "r", encoding="utf-8") as f:
            rows = [line.strip().split() for line in f if line.strip()]
        if rows and len(rows[0]) == 1 and rows[0][0].isdigit():
            rows = rows[1:]
        for parts in rows:
            if len(parts) < 3:
                continue
            h = self._id(parts[0], self.entity2id) if build_mapping else int(parts[0])
            r = self._id(parts[1], self.relation2id) if build_mapping else int(parts[1])
            t = self._id(parts[2], self.entity2id) if build_mapping else int(parts[2])
            triples.append((h, r, t))
        return triples

    def _finalize_numeric_mappings(self):
        # When only numeric triples are available, infer dense ID spaces from all
        # observed heads, tails, and relations.
        triples = self.train + self.valid + self.test
        if not triples:
            return
        if not self.entity2id:
            max_entity = max(max(h, t) for h, _, t in triples)
            self.entity2id = {str(i): i for i in range(max_entity + 1)}
        if not self.relation2id:
            max_relation = max(r for _, r, _ in triples)
            self.relation2id = {str(i): i for i in range(max_relation + 1)}

    def _load_features(self, names):
        # Public MMKGC repositories use slightly different names for text and
        # image features; try common variants before falling back to None.
        for name in names:
            path = self.root / name
            if not path.exists():
                continue
            if path.suffix == ".npy":
                return torch.tensor(np.load(path), dtype=torch.float32)
            if path.suffix == ".pt":
                data = torch.load(path, map_location="cpu")
                return data.float() if torch.is_tensor(data) else torch.tensor(data, dtype=torch.float32)
        return None

    def _build_adjacency(self, triples):
        # Store an undirected training adjacency for local context extraction and
        # future relation-aware reasoning extensions.
        adj = defaultdict(list)
        for h, r, t in triples:
            adj[h].append((r, t))
            adj[t].append((r, h))
        return adj

    def build_incidence(self):
        # Each relation type is treated as one hyperedge connecting all entities
        # that participate in that relation in the training graph.
        h_mat = torch.zeros(self.num_entities, self.num_relations)
        for h, r, t in self.train:
            h_mat[h, r] = 1.0
            h_mat[t, r] = 1.0
        return h_mat

    def negative_sample(self, triples, negative_ratio=1):
        # Random head/tail corruption creates binary training samples for the
        # reference implementation. More advanced samplers can reuse this output
        # shape: (head, relation, tail, label).
        samples = []
        for h, r, t in triples:
            samples.append((h, r, t, 1.0))
            for _ in range(negative_ratio):
                if np.random.rand() < 0.5:
                    h_neg = int(np.random.randint(0, self.num_entities))
                    samples.append((h_neg, r, t, 0.0))
                else:
                    t_neg = int(np.random.randint(0, self.num_entities))
                    samples.append((h, r, t_neg, 0.0))
        return samples

    def as_tensors(self, samples, device):
        h = torch.tensor([x[0] for x in samples], dtype=torch.long, device=device)
        r = torch.tensor([x[1] for x in samples], dtype=torch.long, device=device)
        t = torch.tensor([x[2] for x in samples], dtype=torch.long, device=device)
        y = torch.tensor([x[3] for x in samples], dtype=torch.float32, device=device)
        return h, r, t, y
