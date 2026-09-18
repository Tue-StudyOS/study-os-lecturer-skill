from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str
    created_at: str | None = None


class ConversationEvent(StrictModel):
    event_type: str
    rating: int | None = Field(default=None, ge=1, le=5)
    created_at: str | None = None


class Task(StrictModel):
    id: str
    title: str
    question: str
    topic: str | None = None
    reference_answer: str | None = None
    instructor_context: str | None = None


class Conversation(StrictModel):
    id: str
    user_key: str | None = None
    task_id: str
    messages: list[Message]
    events: list[ConversationEvent] = Field(default_factory=list)


class TaskRating(StrictModel):
    task_id: str
    user_key: str | None = None
    rating: int = Field(ge=1, le=5)


class Dataset(StrictModel):
    tasks: dict[str, Task]
    conversations: list[Conversation]
    task_ratings: list[TaskRating] = Field(default_factory=list)
    source_name: str = "in-memory"


class AnalysisConfig(StrictModel):
    provider: Literal["openai", "openrouter"] = "openai"
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    model: str | None = None
    language: Literal["de", "en"] = "de"
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=8)
    max_output_tokens: int = Field(default=2000, ge=256, le=16000)
    max_transcript_chars: int = Field(default=50000, ge=2000, le=500000)
    max_conversations: int | None = Field(default=None, ge=1)

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == "openrouter":
            return "openai/gpt-5.4-mini"
        return "gpt-5.4-mini"


class RubricConcept(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9_\-]{1,48}$")
    name: str
    description: str
    lecturer_response: str
    diagnostic_question: str


class RubricMisconception(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9_\-]{1,48}$")
    label: str
    description: str
    lecturer_response: str
    diagnostic_question: str


class TaskRubric(StrictModel):
    expected_reasoning: str
    concepts: list[RubricConcept]
    common_misconceptions: list[RubricMisconception]

    @field_validator("concepts", "common_misconceptions")
    @classmethod
    def unique_ids(cls, values):
        ids = [item.id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric ids must be unique")
        return values


class DimensionScores(StrictModel):
    conceptual_accuracy: int | None = Field(ge=0, le=3)
    reasoning_quality: int | None = Field(ge=0, le=3)
    application_transfer: int | None = Field(ge=0, le=3)


class ConceptAssessment(StrictModel):
    concept_id: str
    status: Literal["not_demonstrated", "developing", "secure"]


class MisconceptionAssessment(StrictModel):
    misconception_id: str
    status: Literal["possible", "present"]


class ConversationAssessment(StrictModel):
    evidence_status: Literal["assessable", "insufficient_evidence"]
    evidence_reason: str
    understanding_level: Literal[
        "insufficient_evidence", "misconception", "emerging", "mostly_correct", "secure"
    ]
    dimensions: DimensionScores
    progress: Literal["improved", "unchanged", "declined", "not_assessable"]
    concepts: list[ConceptAssessment]
    misconceptions: list[MisconceptionAssessment]


class ApiUsage(StrictModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class RunError(StrictModel):
    task_id: str | None
    stage: Literal["rubric", "assessment"]
    error_type: str
    message: str
    affected_conversations: int = 1


class Distribution(StrictModel):
    count: int
    percent: float


class ConceptMetric(StrictModel):
    concept_id: str
    name: str
    assessed_count: int
    secure_count: int
    developing_count: int
    not_demonstrated_count: int
    secure_percent: float | None


class MisconceptionMetric(StrictModel):
    misconception_id: str
    label: str
    count: int
    percent_of_assessable: float | None
    lecturer_response: str
    diagnostic_question: str


class LecturerAction(StrictModel):
    priority: int
    task_id: str
    task_title: str
    issue: str
    recommendation: str
    diagnostic_question: str
    affected_count: int


class FeedbackMetric(StrictModel):
    count: int
    average: float | None


class TaskReport(StrictModel):
    task_id: str
    title: str
    topic: str | None
    rubric_source: Literal["instructor_reference", "model_generated", "unavailable"]
    conversation_count: int
    candidate_conversation_count: int
    distinct_student_count: int
    assessable_count: int
    insufficient_evidence_count: int
    low_evidence: bool
    understanding_distribution: dict[str, Distribution]
    dimension_averages: dict[str, float | None]
    progress_distribution: dict[str, Distribution]
    concepts: list[ConceptMetric]
    misconceptions: list[MisconceptionMetric]
    aha_feedback: FeedbackMetric
    task_rating: FeedbackMetric
    mistrust_count: int
    lecturer_actions: list[LecturerAction]


class CohortReport(StrictModel):
    task_count: int
    conversation_count: int
    candidate_conversation_count: int
    distinct_student_count: int
    assessable_count: int
    insufficient_evidence_count: int
    understanding_distribution: dict[str, Distribution]
    dimension_averages: dict[str, float | None]
    low_evidence: bool
    lecturer_actions: list[LecturerAction]


class SourceSummary(StrictModel):
    source_name: str
    task_count: int
    conversation_count: int
    system_only_conversation_count: int
    candidate_conversation_count: int
    orphan_conversation_count: int


class RunMetadata(StrictModel):
    generated_at: str
    provider: str
    model: str
    language: str
    complete: bool
    api_usage: ApiUsage


class AggregateReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run: RunMetadata
    source: SourceSummary
    cohort: CohortReport
    tasks: list[TaskReport]
    errors: list[RunError]
    warnings: list[str]


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


SourceValue = str | Path | Dataset
