"""
ai_modules/skill_clusterer.py
Groups skills into semantic clusters using KMeans on RoBERTa embeddings.
Useful for identifying skill domains (e.g., "Cloud", "Frontend", "ML").
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from loguru import logger

from ai_modules.embedding_engine import EmbeddingEngine


# Pre-defined cluster labels for Computing/Engineering domains
DOMAIN_LABELS = [
    "Cloud & DevOps",
    "Frontend Development",
    "Backend Development",
    "Data Science & ML",
    "Mobile Development",
    "Databases",
    "Security & Networking",
    "Software Engineering Principles",
]

N_CLUSTERS = len(DOMAIN_LABELS)


class SkillClusterer:
    """
    Clusters a list of skills into domain groups.
    Uses KMeans with RoBERTa embeddings.
    """

    def __init__(self):
        self.engine = EmbeddingEngine()

    def cluster_skills(self, skills: list[str]) -> dict[str, list[str]]:
        """
        Group skills into domain clusters.
        Returns dict mapping cluster label → list of skills in that cluster.
        """
        if len(skills) < N_CLUSTERS:
            # Not enough skills to cluster meaningfully
            return {"General Skills": skills}

        embeddings = self.engine.embed_batch(skills)  # (n, dim)

        k = min(N_CLUSTERS, len(skills))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Group skills by cluster ID
        clusters: dict[int, list[str]] = {}
        for skill, label in zip(skills, labels):
            clusters.setdefault(int(label), []).append(skill)

        # Assign human-readable label to each cluster
        # Match cluster center to closest domain using cosine similarity
        domain_embeddings = self.engine.embed_batch(DOMAIN_LABELS)
        named_clusters: dict[str, list[str]] = {}
        used_labels: set[str] = set()

        for cluster_id, cluster_skills in clusters.items():
            center = kmeans.cluster_centers_[cluster_id]
            center_norm = center / (np.linalg.norm(center) + 1e-8)

            # Find best matching domain label
            sims = [float(np.dot(center_norm, d)) for d in domain_embeddings]
            best_domain_idx = int(np.argmax(sims))
            domain_name = DOMAIN_LABELS[best_domain_idx]

            # Avoid duplicate domain names
            suffix = ""
            counter = 2
            while domain_name + suffix in used_labels:
                suffix = f" {counter}"
                counter += 1
            domain_name = domain_name + suffix
            used_labels.add(domain_name)

            named_clusters[domain_name] = sorted(cluster_skills)

        return named_clusters

    def find_skill_domain(self, skill: str) -> str:
        """
        Assign a single skill to its most likely domain.
        """
        skill_emb = self.engine.embed(skill)
        domain_embeddings = self.engine.embed_batch(DOMAIN_LABELS)
        sims = [float(np.dot(skill_emb, d)) for d in domain_embeddings]
        return DOMAIN_LABELS[int(np.argmax(sims))]
