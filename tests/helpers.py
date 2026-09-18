from lecturer_feedback.models import (
    ApiUsage,
    ConceptAssessment,
    Conversation,
    ConversationAssessment,
    Dataset,
    DimensionScores,
    Message,
    MisconceptionAssessment,
    RubricConcept,
    RubricMisconception,
    Task,
    TaskRubric,
)


class FakeEvaluator:
    def __init__(self, fail=False):
        self.usage = ApiUsage()
        self.fail = fail

    def build_rubric(self, task):
        self.usage.calls += 1
        return TaskRubric(
            expected_reasoning="Mass cancels from mg=ma.",
            concepts=[
                RubricConcept(
                    id="mass_independence",
                    name="Massenunabhängigkeit",
                    description="a=g",
                    lecturer_response="Have students derive the acceleration from Newton's law.",
                    diagnostic_question="Does doubling mass change mg/m?",
                )
            ],
            common_misconceptions=[
                RubricMisconception(
                    id="heavier_faster",
                    label="Schwerere Körper fallen schneller",
                    description="Force is confused with acceleration.",
                    lecturer_response="Contrast force and acceleration using F=ma.",
                    diagnostic_question="What happens when mg=ma is simplified?",
                )
            ],
        )

    def assess(self, task, rubric, conversation):
        self.usage.calls += 1
        text = " ".join(m.content for m in conversation.messages if m.role == "user")
        if self.fail:
            raise RuntimeError("raw private text must never reach the report: " + text)
        if "Musterlösung" in text:
            return ConversationAssessment(
                evidence_status="insufficient_evidence",
                evidence_reason="No original reasoning.",
                understanding_level="insufficient_evidence",
                dimensions=DimensionScores(
                    conceptual_accuracy=None, reasoning_quality=None, application_transfer=None
                ),
                progress="not_assessable",
                concepts=[],
                misconceptions=[],
            )
        misconception = "schwer" in text.casefold()
        return ConversationAssessment(
            evidence_status="assessable",
            evidence_reason="Student makes a reasoned physics claim.",
            understanding_level="misconception" if misconception else "secure",
            dimensions=DimensionScores(
                conceptual_accuracy=0 if misconception else 3,
                reasoning_quality=1 if misconception else 3,
                application_transfer=1 if misconception else 2,
            ),
            progress="unchanged",
            concepts=[
                ConceptAssessment(
                    concept_id="mass_independence",
                    status="developing" if misconception else "secure",
                )
            ],
            misconceptions=[
                MisconceptionAssessment(misconception_id="heavier_faster", status="present")
            ]
            if misconception
            else [],
        )


def make_dataset():
    task = Task(id="fall", title="Fall", question="PRIVATE QUESTION", reference_answer="a=g")
    conversations = [
        Conversation(
            id="private-conversation-one",
            user_key="private-user-one",
            task_id="fall",
            messages=[Message(role="user", content="Beide sind gleich schnell. SECRET_ALPHA")],
        ),
        Conversation(
            id="private-conversation-two",
            user_key="private-user-two",
            task_id="fall",
            messages=[Message(role="user", content="Die schwere Kugel ist schneller. SECRET_BETA")],
        ),
        Conversation(
            id="private-conversation-three",
            user_key="private-user-three",
            task_id="fall",
            messages=[Message(role="user", content="Bitte gib die Musterlösung. SECRET_GAMMA")],
        ),
        Conversation(
            id="empty",
            user_key="private-user-four",
            task_id="fall",
            messages=[Message(role="system", content="system")],
        ),
    ]
    return Dataset(tasks={task.id: task}, conversations=conversations, source_name="private.db")
