from dataclasses import dataclass

@dataclass
class Skill:
    phrase: str
    cluster_id: str
    confidence: float
