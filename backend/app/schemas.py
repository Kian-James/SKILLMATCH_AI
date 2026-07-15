"""
app/schemas.py
==============
Pydantic response models for the /thesis-full-analysis endpoint.

Typed models give auto-generated OpenAPI docs, response validation, and
IDE autocompletion when assembling the response in main.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Stage-level report models ─────────────────────────────────────────────────

class Stage1Report(BaseModel):
    text_source:   str  = Field(..., description="'native_pdf' or 'ocr'")
    char_count:    int  = Field(..., description="Total characters extracted")
    ocr_quality:   str  = Field(..., description="Alphanumeric ratio as percentage string")
    ocr_triggered: bool = Field(..., description="True when EasyOCR fallback was used")


class Stage2Report(BaseModel):
    llm_model:          str = Field(..., description="HuggingFace model string used")
    total_skills_found: int = Field(..., description="Number of skills extracted")


class Stage2BReport(BaseModel):
    llm_model:         str = Field(..., description="HuggingFace model string used")
    total_certs_found: int = Field(..., description="Number of certifications extracted")


class PipelineAccuracyReport(BaseModel):
    STAGE_1_TEXT_ACQUISITION: Stage1Report
    STAGE_2_SKILL_EXTRACTION: Stage2Report
    STAGE_2B_CERT_EXTRACTION: Stage2BReport
    STAGE_3_NLP_ACCURACY:     str = Field(..., description="NLP accuracy vs best-matched job")
    STAGE_4_FAISS_INTEGRITY:  str = Field(..., description="FAISS round-trip integrity score")


# ── Job match models ──────────────────────────────────────────────────────────

class NlpValidation(BaseModel):
    NLP_ACCURACY:     str       = Field(..., description="Accuracy percentage string")
    matched_skills:   list[str] = Field(default_factory=list)
    unmatched_skills: list[str] = Field(default_factory=list)


class JobMatch(BaseModel):
    job_id:          str         = Field(..., description="Firestore document ID")
    title:           str
    industry:        str
    score:           float        = Field(..., description="Hybrid score (0–1)")
    cosine_score:    float
    match_ratio:     float
    gap_count:       int
    required_skills: list[str]   = Field(default_factory=list)
    matched_skills:  list[str]   = Field(default_factory=list)
    missing_skills:  list[str]   = Field(default_factory=list)
    nlp_validation:  NlpValidation


class BestMatchSummary(BaseModel):
    job_title:              str       = Field(..., description="Title of the best-matched job")
    confidence_score:       float
    gap_count:              int
    required_skills:        list[str] = Field(default_factory=list)
    matched_skills:         list[str] = Field(default_factory=list)
    missing_skills:         list[str] = Field(default_factory=list)
    matched_certifications: list[str] = Field(default_factory=list)
    missing_certifications: list[str] = Field(default_factory=list)


class SkillRecommendations(BaseModel):
    target_job:         str       = Field(..., description="Second-best matched job title")
    skills_to_learn:    list[str] = Field(default_factory=list, alias="skill(s)_to_learn")

    class Config:
        populate_by_name = True


class CareerGuidanceSystem(BaseModel):
    JOB_RECOMMENDATIONS:  list[JobMatch]
    BEST_MATCH_SUMMARY:   BestMatchSummary
    SKILL_RECOMMENDATIONS: SkillRecommendations


# ── Extraction summary models ─────────────────────────────────────────────────

class ExtractionSummary(BaseModel):
    file_name:                str
    text_source:              str
    discovered_skills:        list[str] = Field(default_factory=list)
    total_skills_found:       int
    discovered_certificates:  list[str] = Field(default_factory=list)
    total_certificates_found: int


# ── Debug report models ───────────────────────────────────────────────────────

class DebugStage1(BaseModel):
    text_source: str
    ocr_quality: str
    char_count:  int
    raw_text:    str


class DebugStage2(BaseModel):
    llm_model:                 str
    skills_section_rule_based: str
    discovered_skills:         list[str] = Field(default_factory=list)


class DebugStage2B(BaseModel):
    llm_model:               str
    cert_section_rule_based: str
    discovered_certificates: list[str] = Field(default_factory=list)


class PerJobNlp(BaseModel):
    rank:               int
    job_id:             str
    title:              str
    nlp_accuracy:       str
    matched_skills:     list[str] = Field(default_factory=list)
    unmatched_skills:   list[str] = Field(default_factory=list)
    ground_truth_count: int


class DebugStage3(BaseModel):
    ground_truth_source: str
    ground_truth_count:  int
    nlp_accuracy:        str
    matched_skills:      list[str] = Field(default_factory=list)
    unmatched_skills:    list[str] = Field(default_factory=list)
    per_job_nlp:         list[PerJobNlp] = Field(default_factory=list)


class DebugStage4(BaseModel):
    score:       str
    index_total: int


class DebugReport(BaseModel):
    STAGE_1_TEXT_ACQUISITION: DebugStage1
    STAGE_2_SKILL_EXTRACTION: DebugStage2
    STAGE_2B_CERT_EXTRACTION: DebugStage2B
    STAGE_3_NLP_VALIDATION:   DebugStage3
    STAGE_4_FAISS_INTEGRITY:  DebugStage4


# ── Top-level response model ──────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    PIPELINE_ACCURACY_REPORT: PipelineAccuracyReport
    CAREER_GUIDANCE_SYSTEM:   CareerGuidanceSystem
    EXTRACTION_SUMMARY:       ExtractionSummary
    DEBUG_REPORT:             DebugReport