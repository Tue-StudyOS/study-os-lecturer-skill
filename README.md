# Lecturer Feedback

Ein eigenständiges Python-Paket, das aufgabenbezogene Lernchats mit einem LLM
auswertet und **ausschließlich aggregierte** Rückmeldungen für Lehrende erzeugt.
Es verändert die Quellanwendung und deren Datenbank nicht.

## Was das Paket liefert

- Abdeckung: Aufgaben, Chats mit Studierendenbeitrag und bewertbare Chats
- Verteilung des Verständnisstands statt individueller Noten
- fachliche Richtigkeit, Begründungsqualität und Anwendung/Transfer (ordinal 0–3)
- sichtbare Lernentwicklung, Konzepte und wiederkehrende Fehlvorstellungen
- konkrete Lehrmaßnahmen und diagnostische Rückfragen
- vorhandene Aha-, Aufgabenbewertungs- und Mistrust-Signale, getrennt vom Lernstand

Jeder Bericht enthält Evidenzzähler und warnt bei weniger als drei bewertbaren
Chats. Eine Bitte um die Musterlösung, organisatorischer Text oder reine
Zustimmung gilt nicht als Verständnisnachweis.

## Installation

Voraussetzung ist Python 3.10 oder neuer.

```bash
cd lecturer_feedback_package
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

API-Schlüssel werden nur über die Umgebung übergeben:

```bash
# Direkt über OpenAI
export OPENAI_API_KEY="..."

# Oder über OpenRouter
export OPENROUTER_API_KEY="..."
```

Die `.env.example` dokumentiert alle Einstellungen. Das Paket lädt `.env`
absichtlich nicht automatisch, damit Schlüssel nicht unbemerkt aus Dateien
übernommen werden.

## Schnellstart mit dem vorhandenen Export

Zuerst kann die Eingabe ohne API-Aufruf geprüft werden:

```bash
lecturer-feedback validate --input ../exports/analysis_export.db
```

Danach wird der Bericht erzeugt:

```bash
lecturer-feedback analyze \
  --input ../exports/analysis_export.db \
  --output ./report \
  --provider openai
```

Für OpenRouter:

```bash
lecturer-feedback analyze \
  --input ../exports/analysis_export.db \
  --output ./report \
  --provider openrouter
```

Standardmodelle sind `gpt-5.4-mini` beziehungsweise
`openai/gpt-5.4-mini`. Ein anderes Modell kann mit `--model` oder
`LECTURER_FEEDBACK_MODEL` gesetzt werden. Das gewählte Modell muss strikte
JSON-Schema-Ausgaben unterstützen.

Sehr lange Chats werden standardmäßig auf insgesamt 50.000 Zeichen aus Anfang
und Ende begrenzt. Das Limit kann mit `--max-transcript-chars` angepasst werden;
`--max-conversations` eignet sich für einen begrenzten Pilotlauf.

## Ausgabe

Im Ausgabeordner entstehen:

- `report.json`: primärer, versionierter Datenvertrag (`schema_version: 1.0`)
- `task_metrics.csv`: eine Zeile pro Aufgabe
- `concept_metrics.csv`: Konzepte und Fehlvorstellungen
- `lecturer_report.html`: eigenständige lokale Ansicht mit eingebettetem JSON

Die HTML-Datei kann direkt per Doppelklick geöffnet werden. Sie verwendet keine
CDNs, lädt keine Daten nach und enthält weder Rohchats noch Benutzer- oder
Gesprächs-IDs.

## Eingabeformate

### SQLite

Die Datei muss eine normale SQLite-Datenbank mit diesen drei Tabellen und
Mindestspalten sein:

- `tasks`: `id`, `title`, `question`; optional `sidebar_label`,
  `reference_answer`, `task_prompt`
- `conversations`: `id`, `task_id`; optional `user_id`
- `messages`: `conversation_id`, `role`, `content`; optional `id`, `created_at`

`conversations.task_id` muss auf `tasks.id` zeigen und
`messages.conversation_id` auf `conversations.id`. Unterstützte Nachrichtenrollen
sind `system`, `developer`, `user` und `assistant`. Nur Gespräche mit mindestens
einer nichtleeren `user`-Nachricht werden an das Modell geschickt.

Optional können folgende Tabellen enthalten sein:

- `conversation_events`: `conversation_id`, `event_type`; optional `rating`,
  `created_at` (`aha` und `mistrust` werden im Bericht berücksichtigt)
- `task_ratings`: `task_id`, `rating`; optional `user_id`

Ein minimales Schema sieht beispielsweise so aus:

```sql
CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, question TEXT);
CREATE TABLE conversations (id TEXT PRIMARY KEY, task_id TEXT, user_id TEXT);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  conversation_id TEXT,
  role TEXT,
  content TEXT,
  created_at TEXT
);
```

Die Aufgabenstellung aus `tasks.question` ist maßgeblich. Gespeicherte
Systemnachrichten werden nicht als Aufgabenquelle oder Lernnachweis verwendet.
`free_*`- und `admin_*`-Tabellen sowie Nutzerkonto-Felder werden nicht gelesen.

### JSONL

Jede Zeile enthält eine Aufgabe und einen Chat. Ein vollständiges synthetisches
Beispiel liegt unter `examples/conversations.jsonl`.

```json
{
  "task": {
    "id": "task-1",
    "title": "Freier Fall",
    "question": "...",
    "topic": "Kinematik",
    "reference_answer": "Optionale, von Lehrenden geprüfte Referenz"
  },
  "conversation": {
    "id": "interne-id",
    "user_key": "internes-pseudonym",
    "task_id": "task-1",
    "messages": [
      {"role": "user", "content": "...", "created_at": null}
    ],
    "events": []
  }
}
```

`id` und `user_key` werden nur zum Ordnen und Zählen im Arbeitsspeicher benutzt
und erscheinen nicht im Bericht. Eine Referenzantwort verbessert die fachliche
Validität. Ohne sie erstellt das Modell einen vorläufigen Bewertungsrahmen und
der Bericht kennzeichnet die notwendige Prüfung durch Lehrende.

### Komprimierter Beta-Log-Export

`beta-ai-logs.json.gz` kann direkt eingelesen werden. Erwartet werden ein
`results`-Array sowie je Ergebnis `beta_exercise_id`,
`beta_exercise_result_id`, `exercise_title`, `user` und `conversation`.
Nachrichtenrollen werden automatisch von `student`/`tutor` auf
`user`/`assistant` abgebildet. Ergebnisse derselben `beta_exercise_id` werden
für den Bericht zu einer Aufgabe gruppiert; dokumentierte Konzepte aus
`trace_history` dienen als zusätzlicher Curriculum-Kontext. Sonstige bereits
berechnete Trace-Bewertungen werden nicht ungeprüft als neuer Lernstand
übernommen.

```bash
lecturer-feedback analyze \
  --input ../logdata/beta-ai-logs.json.gz \
  --output ./beta-report \
  --provider openai
```

## Python-Integration

```python
from lecturer_feedback import AnalysisConfig, analyze
from lecturer_feedback.reporting import write_report

report = analyze(
    "analysis_export.db",
    AnalysisConfig(provider="openai", language="de"),
)
write_report(report, "report")
```

Alternativ kann eine validierte `Dataset`-Instanz direkt an `analyze` übergeben
werden. Damit kann die bestehende Anwendung später einen eigenen Exportadapter
bereitstellen, ohne die Analysepipeline zu ändern.

## Datenschutz und fachliche Grenzen

- Rohchats werden zur Auswertung an den konfigurierten API-Anbieter übertragen.
- Bei OpenRouter fordert der Client `data_collection: deny`, ZDR-Routing und
  Unterstützung aller Parameter an. Das ersetzt keine institutionelle Prüfung
  von Auftragsverarbeitung, Datenstandort und Aufbewahrung.
- Konversationseinzelbewertungen werden nicht gespeichert. Fehlertexte werden
  bereinigt; Bericht und CSVs enthalten keine Chattexte oder Personenkennungen.
- LLM-Einstufungen sind diagnostische Hinweise, keine Prüfungsnoten. Besonders
  modellgenerierte Bewertungsrahmen müssen fachlich geprüft werden.
- Aktivität, Ratings und Aha-Signale werden separat gezeigt und nicht als Beleg
  für Verständnis interpretiert.

## Entwicklung und Übergabe

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Die Tests benötigen keinen API-Schlüssel. Optionale manuelle Smoke-Tests können
mit dem synthetischen JSONL-Beispiel durchgeführt werden.

Die Übergabe-ZIP wird aus dem übergeordneten Verzeichnis erstellt:

```bash
python -m zipfile -c lecturer_feedback_package.zip lecturer_feedback_package
```
