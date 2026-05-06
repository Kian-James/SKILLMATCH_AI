# core/__init__.py — public surface of the core pipeline package
from core.extraction  import extract_text
from core.skills      import discover_skills_dynamically, extract_skills_section
from core.certs       import extract_certificates, extract_cert_section
from core.scoring     import calculate_cosine_similarity, identify_skill_gap
from core.validation  import fuzzy_skill_verify
from core.utils       import normalize