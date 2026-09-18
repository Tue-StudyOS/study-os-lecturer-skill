from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Callable

from .models import (
    AggregateReport,
    AnalysisConfig,
    ApiUsage,
    CohortReport,
    ConceptMetric,
    Conversation,
    ConversationAssessment,
    Dataset,
    Distribution,
    FeedbackMetric,
    LecturerAction,
    MisconceptionMetric,
    RunError,
    RunMetadata,
    SourceSummary,
    SourceValue,
    Task,
    TaskReport,
    TaskRubric,
    utc_now_iso,
)
from .providers import Evaluator, create_evaluator
from .sources import load_source


UNDERSTANDING_LEVELS = (
    "insufficient_evidence",
    "misconception",
    "emerging",
    "mostly_correct",
    "secure",
)
PROGRESS_LEVELS = ("improved", "unchanged", "declined", "not_assessable")
DIMENSIONS = ("conceptual_accuracy", "reasoning_quality", "application_transfer")


def _candidate(conversation: Conversation) -> bool:
    return any(message.role == "user" and message.content.strip() for message in conversation.messages)


def _distribution(values: list[str], labels: tuple[str, ...]) -> dict[str, Distribution]:
    counts = Counter(values)
    total = len(values)
    return {
        label: Distribution(
            count=counts[label],
            percent=round((counts[label] / total) * 100, 1) if total else 0.0,
        )
        for label in labels
    }


def _dimension_averages(assessments: list[ConversationAssessment]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for dimension in DIMENSIONS:
        values = [
            getattr(assessment.dimensions, dimension)
            for assessment in assessments
            if getattr(assessment.dimensions, dimension) is not None
        ]
        result[dimension] = round(mean(values), 2) if values else None
    return result


def _feedback(values: list[int]) -> FeedbackMetric:
    return FeedbackMetric(count=len(values), average=round(mean(values), 2) if values else None)


def _clean_assessment(assessment: ConversationAssessment, rubric: TaskRubric) -> ConversationAssessment:
    """Enforce cross-object invariants that JSON Schema cannot express."""
    concept_ids = {concept.id for concept in rubric.concepts}
    misconception_ids = {item.id for item in rubric.common_misconceptions}
    assessment.concepts = [item for item in assessment.concepts if item.concept_id in concept_ids]
    assessment.misconceptions = [
        item for item in assessment.misconceptions if item.misconception_id in misconception_ids
    ]
    if assessment.evidence_status == "insufficient_evidence":
        assessment.understanding_level = "insufficient_evidence"
        assessment.progress = "not_assessable"
        assessment.dimensions.conceptual_accuracy = None
        assessment.dimensions.reasoning_quality = None
        assessment.dimensions.application_transfer = None
        assessment.concepts = []
        assessment.misconceptions = []
    elif assessment.understanding_level == "insufficient_evidence":
        assessment.evidence_status = "insufficient_evidence"
        return _clean_assessment(assessment, rubric)
    else:
        by_id = {item.concept_id: item for item in assessment.concepts}
        assessment.concepts = [
            by_id.get(concept.id)
            or {"concept_id": concept.id, "status": "not_demonstrated"}
            for concept in rubric.concepts
        ]
        assessment = ConversationAssessment.model_validate(assessment.model_dump())
    return assessment


def _task_report(
    *,
    task: Task,
    conversations: list[Conversation],
    candidate_conversations: list[Conversation],
    rubric: TaskRubric | None,
    assessments: list[ConversationAssessment],
    ratings: list[int],
) -> TaskReport:
    assessable = [item for item in assessments if item.evidence_status == "assessable"]
    insufficient = len(assessments) - len(assessable)
    level_values = [item.understanding_level for item in assessments]
    progress_values = [item.progress for item in assessments]

    concepts: list[ConceptMetric] = []
    misconceptions: list[MisconceptionMetric] = []
    if rubric:
        for concept in rubric.concepts:
            statuses = [
                item.status
                for assessment in assessable
                for item in assessment.concepts
                if item.concept_id == concept.id
            ]
            counts = Counter(statuses)
            concepts.append(
                ConceptMetric(
                    concept_id=concept.id,
                    name=concept.name,
                    assessed_count=len(statuses),
                    secure_count=counts["secure"],
                    developing_count=counts["developing"],
                    not_demonstrated_count=counts["not_demonstrated"],
                    secure_percent=round(counts["secure"] / len(statuses) * 100, 1)
                    if statuses
                    else None,
                )
            )

        rubric_misconceptions = {item.id: item for item in rubric.common_misconceptions}
        misconception_counts = Counter(
            item.misconception_id
            for assessment in assessable
            for item in assessment.misconceptions
            if item.status == "present"
        )
        for misconception_id, count in misconception_counts.most_common():
            item = rubric_misconceptions[misconception_id]
            misconceptions.append(
                MisconceptionMetric(
                    misconception_id=misconception_id,
                    label=item.label,
                    count=count,
                    percent_of_assessable=round(count / len(assessable) * 100, 1)
                    if assessable
                    else None,
                    lecturer_response=item.lecturer_response,
                    diagnostic_question=item.diagnostic_question,
                )
            )

    misconceptions.sort(key=lambda item: (-item.count, item.label.casefold()))

    actions = [
        LecturerAction(
            priority=index,
            task_id=task.id,
            task_title=task.title,
            issue=item.label,
            recommendation=item.lecturer_response,
            diagnostic_question=item.diagnostic_question,
            affected_count=item.count,
        )
        for index, item in enumerate(misconceptions[:5], start=1)
    ]
    if rubric and len(actions) < 5:
        concept_by_id = {item.id: item for item in rubric.concepts}
        weak_concepts = sorted(
            (
                item
                for item in concepts
                if item.assessed_count and item.secure_count < item.assessed_count
            ),
            key=lambda item: (
                -(item.developing_count + item.not_demonstrated_count),
                item.name.casefold(),
            ),
        )
        for metric in weak_concepts:
            if len(actions) >= 5:
                break
            concept = concept_by_id[metric.concept_id]
            actions.append(
                LecturerAction(
                    priority=len(actions) + 1,
                    task_id=task.id,
                    task_title=task.title,
                    issue=concept.name,
                    recommendation=concept.lecturer_response,
                    diagnostic_question=concept.diagnostic_question,
                    affected_count=metric.developing_count + metric.not_demonstrated_count,
                )
            )

    aha_values = [
        event.rating
        for conversation in conversations
        for event in conversation.events
        if event.event_type.casefold() == "aha" and event.rating is not None
    ]
    mistrust_count = sum(
        1
        for conversation in conversations
        for event in conversation.events
        if event.event_type.casefold() == "mistrust"
    )
    students = {conversation.user_key for conversation in candidate_conversations if conversation.user_key}
    rubric_source = (
        "unavailable"
        if rubric is None
        else "instructor_reference"
        if task.reference_answer
        else "model_generated"
    )
    return TaskReport(
        task_id=task.id,
        title=task.title,
        topic=task.topic,
        rubric_source=rubric_source,
        conversation_count=len(conversations),
        candidate_conversation_count=len(candidate_conversations),
        distinct_student_count=len(students),
        assessable_count=len(assessable),
        insufficient_evidence_count=insufficient,
        low_evidence=len(assessable) < 3,
        understanding_distribution=_distribution(level_values, UNDERSTANDING_LEVELS),
        dimension_averages=_dimension_averages(assessable),
        progress_distribution=_distribution(progress_values, PROGRESS_LEVELS),
        concepts=concepts,
        misconceptions=misconceptions,
        aha_feedback=_feedback(aha_values),
        task_rating=_feedback(ratings),
        mistrust_count=mistrust_count,
        lecturer_actions=actions,
    )


def _cohort_report(
    task_reports: list[TaskReport],
    all_assessments: list[ConversationAssessment],
    candidate_conversations: list[Conversation],
) -> CohortReport:
    assessable = [item for item in all_assessments if item.evidence_status == "assessable"]
    students = {conversation.user_key for conversation in candidate_conversations if conversation.user_key}
    actions = sorted(
        (action for report in task_reports for action in report.lecturer_actions),
        key=lambda item: (-item.affected_count, item.task_title.casefold(), item.priority),
    )[:8]
    actions = [action.model_copy(update={"priority": index}) for index, action in enumerate(actions, 1)]
    return CohortReport(
        task_count=len(task_reports),
        conversation_count=sum(report.conversation_count for report in task_reports),
        candidate_conversation_count=len(candidate_conversations),
        distinct_student_count=len(students),
        assessable_count=len(assessable),
        insufficient_evidence_count=len(all_assessments) - len(assessable),
        understanding_distribution=_distribution(
            [item.understanding_level for item in all_assessments], UNDERSTANDING_LEVELS
        ),
        dimension_averages=_dimension_averages(assessable),
        low_evidence=len(assessable) < 3,
        lecturer_actions=actions,
    )


def analyze(
    source: SourceValue,
    config: AnalysisConfig | None = None,
    *,
    evaluator: Evaluator | None = None,
    progress: Callable[[str], None] | None = None,
) -> AggregateReport:
    """Analyze chats and return an aggregate-only report.

    Conversation-level model results are intentionally held in local memory and are
    not part of the returned public contract.
    """
    config = config or AnalysisConfig()
    dataset: Dataset = load_source(source)
    task_ids = set(dataset.tasks)
    orphan_count = sum(1 for item in dataset.conversations if item.task_id not in task_ids)
    valid_conversations = [item for item in dataset.conversations if item.task_id in task_ids]
    candidates = [item for item in valid_conversations if _candidate(item)]
    system_only_count = sum(1 for item in valid_conversations if not _candidate(item))
    warnings: list[str] = []
    if orphan_count:
        warnings.append(
            f"{orphan_count} Gespräch(e) verweisen auf eine fehlende Aufgabe und wurden übersprungen."
            if config.language == "de"
            else f"{orphan_count} conversation(s) reference a missing task and were skipped."
        )
    if system_only_count:
        warnings.append(
            f"{system_only_count} Gespräch(e) enthalten keinen Studierendenbeitrag und wurden nicht bewertet."
            if config.language == "de"
            else f"{system_only_count} conversation(s) contain no student message and were not assessed."
        )
    if not dataset.conversations:
        warnings.append(
            "Die Eingabe enthält keine Gespräche."
            if config.language == "de"
            else "The input contains no conversations."
        )
    if config.max_conversations is not None and len(candidates) > config.max_conversations:
        candidates = candidates[: config.max_conversations]
        warnings.append(
            f"Die Analyse wurde auf die ersten {config.max_conversations} auswertbaren Chat-Kandidaten begrenzt."
            if config.language == "de"
            else f"Analysis was limited to the first {config.max_conversations} candidate conversation(s)."
        )

    if candidates and evaluator is None:
        evaluator = create_evaluator(config)

    conversations_by_task: dict[str, list[Conversation]] = defaultdict(list)
    candidates_by_task: dict[str, list[Conversation]] = defaultdict(list)
    for conversation in valid_conversations:
        conversations_by_task[conversation.task_id].append(conversation)
    for conversation in candidates:
        candidates_by_task[conversation.task_id].append(conversation)

    ratings_by_task: dict[str, list[int]] = defaultdict(list)
    for rating in dataset.task_ratings:
        if rating.task_id in task_ids:
            ratings_by_task[rating.task_id].append(rating.rating)

    errors: list[RunError] = []
    task_assessments: dict[str, list[ConversationAssessment]] = defaultdict(list)
    rubrics: dict[str, TaskRubric] = {}
    assessment_number = 0
    assessment_total = len(candidates)

    for task_id, task_candidates in candidates_by_task.items():
        task = dataset.tasks[task_id]
        try:
            if evaluator is None:  # pragma: no cover - guarded above
                raise RuntimeError("evaluator unavailable")
            if progress:
                progress(f"Generating rubric: {task.title}")
            rubric = evaluator.build_rubric(task)
            rubrics[task_id] = rubric
        except Exception as error:
            assessment_number += len(task_candidates)
            if progress:
                progress(f"Skipping {len(task_candidates)} conversation(s): rubric failed")
            errors.append(
                RunError(
                    task_id=task_id,
                    stage="rubric",
                    error_type=type(error).__name__,
                    message="The assessment rubric could not be generated.",
                    affected_conversations=len(task_candidates),
                )
            )
            continue

        for conversation in task_candidates:
            assessment_number += 1
            try:
                if progress:
                    progress(
                        f"Assessing conversation {assessment_number}/{assessment_total}: {task.title}"
                    )
                assessment = evaluator.assess(task, rubric, conversation)
                task_assessments[task_id].append(_clean_assessment(assessment, rubric))
            except Exception as error:
                errors.append(
                    RunError(
                        task_id=task_id,
                        stage="assessment",
                        error_type=type(error).__name__,
                        message="One conversation could not be assessed.",
                    )
                )

    task_reports = [
        _task_report(
            task=task,
            conversations=conversations_by_task.get(task.id, []),
            candidate_conversations=candidates_by_task.get(task.id, []),
            rubric=rubrics.get(task.id),
            assessments=task_assessments.get(task.id, []),
            ratings=ratings_by_task.get(task.id, []),
        )
        for task in dataset.tasks.values()
    ]
    all_assessments = [item for values in task_assessments.values() for item in values]

    if candidates and len(all_assessments) < 3:
        warnings.append(
            "Weniger als drei Gespräche enthalten bewertbare Lernnachweise; Ergebnisse vorsichtig interpretieren."
            if config.language == "de"
            else "Fewer than three conversations contain assessable learning evidence; interpret findings cautiously."
        )
    if any(report.rubric_source == "model_generated" for report in task_reports):
        warnings.append(
            "Mindestens ein Bewertungsrahmen wurde mangels Referenzantwort vom Modell erstellt und muss fachlich geprüft werden."
            if config.language == "de"
            else "At least one rubric was model-generated because no instructor reference answer was supplied; lecturer review is required."
        )
    if errors:
        warnings.append(
            "Der Bericht ist unvollständig, weil mindestens ein Modellaufruf fehlgeschlagen ist."
            if config.language == "de"
            else "The report is incomplete because one or more model calls failed."
        )

    usage = evaluator.usage if evaluator is not None else ApiUsage()
    return AggregateReport(
        run=RunMetadata(
            generated_at=utc_now_iso(),
            provider=config.provider,
            model=config.resolved_model(),
            language=config.language,
            complete=not errors,
            api_usage=usage,
        ),
        source=SourceSummary(
            source_name=dataset.source_name,
            task_count=len(dataset.tasks),
            conversation_count=len(dataset.conversations),
            system_only_conversation_count=system_only_count,
            candidate_conversation_count=len(candidates),
            orphan_conversation_count=orphan_count,
        ),
        cohort=_cohort_report(task_reports, all_assessments, candidates),
        tasks=task_reports,
        errors=errors,
        warnings=list(dict.fromkeys(warnings)),
    )
