from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import AggregateReport


TASK_FIELDS = (
    "task_id",
    "title",
    "topic",
    "conversation_count",
    "candidate_conversation_count",
    "distinct_student_count",
    "assessable_count",
    "insufficient_evidence_count",
    "low_evidence",
    "conceptual_accuracy_average",
    "reasoning_quality_average",
    "application_transfer_average",
    "misconception_count",
    "aha_count",
    "aha_average",
    "task_rating_count",
    "task_rating_average",
    "mistrust_count",
)

CONCEPT_FIELDS = (
    "task_id",
    "task_title",
    "metric_type",
    "metric_id",
    "label",
    "assessed_count",
    "secure_count",
    "developing_count",
    "not_demonstrated_count",
    "secure_percent",
    "affected_count",
    "affected_percent",
    "lecturer_response",
    "diagnostic_question",
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv_safe(value):
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_task_csv(path: Path, report: AggregateReport) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS)
        writer.writeheader()
        for task in report.tasks:
            row = {
                    "task_id": task.task_id,
                    "title": task.title,
                    "topic": task.topic or "",
                    "conversation_count": task.conversation_count,
                    "candidate_conversation_count": task.candidate_conversation_count,
                    "distinct_student_count": task.distinct_student_count,
                    "assessable_count": task.assessable_count,
                    "insufficient_evidence_count": task.insufficient_evidence_count,
                    "low_evidence": str(task.low_evidence).lower(),
                    "conceptual_accuracy_average": task.dimension_averages["conceptual_accuracy"],
                    "reasoning_quality_average": task.dimension_averages["reasoning_quality"],
                    "application_transfer_average": task.dimension_averages["application_transfer"],
                    "misconception_count": sum(item.count for item in task.misconceptions),
                    "aha_count": task.aha_feedback.count,
                    "aha_average": task.aha_feedback.average,
                    "task_rating_count": task.task_rating.count,
                    "task_rating_average": task.task_rating.average,
                    "mistrust_count": task.mistrust_count,
            }
            writer.writerow({key: _csv_safe(value) for key, value in row.items()})


def _write_concept_csv(path: Path, report: AggregateReport) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONCEPT_FIELDS)
        writer.writeheader()
        for task in report.tasks:
            for concept in task.concepts:
                row = {
                        "task_id": task.task_id,
                        "task_title": task.title,
                        "metric_type": "concept",
                        "metric_id": concept.concept_id,
                        "label": concept.name,
                        "assessed_count": concept.assessed_count,
                        "secure_count": concept.secure_count,
                        "developing_count": concept.developing_count,
                        "not_demonstrated_count": concept.not_demonstrated_count,
                        "secure_percent": concept.secure_percent,
                        "affected_count": "",
                        "affected_percent": "",
                        "lecturer_response": "",
                        "diagnostic_question": "",
                }
                writer.writerow({key: _csv_safe(value) for key, value in row.items()})
            for misconception in task.misconceptions:
                row = {
                        "task_id": task.task_id,
                        "task_title": task.title,
                        "metric_type": "misconception",
                        "metric_id": misconception.misconception_id,
                        "label": misconception.label,
                        "assessed_count": task.assessable_count,
                        "secure_count": "",
                        "developing_count": "",
                        "not_demonstrated_count": "",
                        "secure_percent": "",
                        "affected_count": misconception.count,
                        "affected_percent": misconception.percent_of_assessable,
                        "lecturer_response": misconception.lecturer_response,
                        "diagnostic_question": misconception.diagnostic_question,
                }
                writer.writerow({key: _csv_safe(value) for key, value in row.items()})


def build_html(report: AggregateReport) -> str:
    data = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    # Prevent a model-generated string from terminating the script element.
    safe_data = data.replace("<", "\\u003c").replace("&", "\\u0026")
    return HTML_TEMPLATE.replace("__REPORT_JSON__", safe_data)


def write_report(report: AggregateReport, output_dir: str | Path, *, include_html: bool = True) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = [
        output_path / "report.json",
        output_path / "task_metrics.csv",
        output_path / "concept_metrics.csv",
    ]
    _write_json(paths[0], report.model_dump(mode="json"))
    _write_task_csv(paths[1], report)
    _write_concept_csv(paths[2], report)
    if include_html:
        html_path = output_path / "lecturer_report.html"
        html_path.write_text(build_html(report), encoding="utf-8")
        paths.append(html_path)
    return paths


HTML_TEMPLATE = r'''<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Lecturer Feedback</title>
  <style>
    :root{--ink:#17212b;--muted:#5f6b76;--paper:#f4f7f9;--card:#fff;--line:#dce4e9;--blue:#276b8d;--teal:#1a7f78;--amber:#b66a16;--red:#ad3b46;--shadow:0 12px 30px rgba(22,40,55,.08)}
    *{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#edf4f5,#f8f5ee);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:36px 24px 70px}
    header{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:25px}h1{font:700 clamp(28px,4vw,44px)/1.05 Georgia,serif;margin:0 0 8px}.subtitle,.muted{color:var(--muted)}.meta{text-align:right;font-size:13px}
    .warnings{display:grid;gap:8px;margin-bottom:20px}.warning{background:#fff7e9;border-left:4px solid var(--amber);padding:10px 14px;border-radius:5px}.error{border-color:var(--red);background:#fff1f2}
    .cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:20px 0}.card,.panel{background:var(--card);border:1px solid rgba(210,220,226,.8);border-radius:13px;box-shadow:var(--shadow)}.card{padding:17px}.card strong{display:block;font-size:28px}.card span{color:var(--muted);font-size:13px}
    .toolbar{display:flex;align-items:center;gap:12px;margin:28px 0 15px}select{max-width:520px;width:100%;padding:11px 13px;border:1px solid #b9c8d0;border-radius:8px;background:white;color:var(--ink)}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{padding:20px;min-width:0}.wide{grid-column:1/-1}h2{font:700 23px/1.2 Georgia,serif;margin:0 0 13px}h3{font-size:16px;margin:19px 0 8px}.bar-row{display:grid;grid-template-columns:150px 1fr 65px;gap:10px;align-items:center;margin:9px 0}.track{height:10px;background:#e7edef;border-radius:99px;overflow:hidden}.bar{height:100%;background:var(--blue);border-radius:99px}.bar.good{background:var(--teal)}.bar.bad{background:var(--red)}
    table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 7px;vertical-align:top}th{color:var(--muted);font-weight:600}.pill{display:inline-block;background:#e7f1f4;color:#245d74;padding:3px 8px;border-radius:99px;font-size:12px}.low{background:#fff0d8;color:#854e12}
    ol.actions{padding-left:22px;margin-bottom:0}.actions li{padding:0 0 14px 5px}.actions p{margin:4px 0}.empty{padding:25px;color:var(--muted);text-align:center;border:1px dashed #c5d0d6;border-radius:9px}.foot{margin-top:24px;color:var(--muted);font-size:12px}
    @media(max-width:800px){.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.wide{grid-column:auto}header{display:block}.meta{text-align:left;margin-top:12px}.bar-row{grid-template-columns:115px 1fr 55px}}
    @media print{body{background:#fff}main{max-width:none;padding:10px}.card,.panel{box-shadow:none;break-inside:avoid}.toolbar{display:none}}
  </style>
</head>
<body><main>
  <header><div><h1 id="title"></h1><div class="subtitle" id="subtitle"></div></div><div class="meta" id="meta"></div></header>
  <div class="warnings" id="warnings"></div>
  <section class="cards" id="cards"></section>
  <div class="toolbar"><label for="taskSelect" id="taskLabel"></label><select id="taskSelect"></select></div>
  <section class="grid">
    <article class="panel"><h2 id="distributionTitle"></h2><div id="distribution"></div></article>
    <article class="panel"><h2 id="dimensionTitle"></h2><div id="dimensions"></div></article>
    <article class="panel wide"><h2 id="conceptTitle"></h2><div id="concepts"></div></article>
    <article class="panel wide"><h2 id="misconceptionTitle"></h2><div id="misconceptions"></div></article>
    <article class="panel wide"><h2 id="actionTitle"></h2><div id="actions"></div></article>
  </section>
  <p class="foot" id="foot"></p>
</main>
<script>
const REPORT=__REPORT_JSON__;
const de={title:"Lernstands-Feedback",subtitle:"Aggregierte Hinweise aus aufgabenbezogenen Lernchats",task:"Ansicht",all:"Gesamtübersicht",distribution:"Verständnisverteilung",dimensions:"Dimensionen (0–3)",concepts:"Konzepte",misconceptions:"Erkannte Fehlvorstellungen",actions:"Empfohlene Lehrmaßnahmen",students:"Studierende",candidates:"Chats mit Studierendenbeitrag",assessable:"Bewertbare Chats",tasks:"Aufgaben",empty:"Keine belastbaren Daten vorhanden.",low:"Geringe Evidenz",generated:"Erstellt",model:"Modell",secure:"Sicher",mostly_correct:"Überwiegend korrekt",emerging:"Im Aufbau",misconception:"Fehlvorstellung",insufficient_evidence:"Nicht ausreichend",conceptual_accuracy:"Fachliche Richtigkeit",reasoning_quality:"Begründungsqualität",application_transfer:"Anwendung/Transfer",metric:"Konzept",assessed:"Bewertet",securePct:"Sicher",affected:"Betroffen",response:"Maßnahme",question:"Diagnosefrage",foot:"Ordinalwerte sind diagnostische Hinweise, keine Noten. Aktivität und Bewertungen werden nicht als Verständnisnachweis verwendet."};
const en={title:"Learning feedback",subtitle:"Aggregate signals from task-based learning chats",task:"View",all:"Overall summary",distribution:"Understanding distribution",dimensions:"Dimensions (0–3)",concepts:"Concepts",misconceptions:"Observed misconceptions",actions:"Recommended teaching actions",students:"Students",candidates:"Chats with student input",assessable:"Assessable chats",tasks:"Tasks",empty:"No defensible evidence available.",low:"Low evidence",generated:"Generated",model:"Model",secure:"Secure",mostly_correct:"Mostly correct",emerging:"Emerging",misconception:"Misconception",insufficient_evidence:"Insufficient evidence",conceptual_accuracy:"Conceptual accuracy",reasoning_quality:"Reasoning quality",application_transfer:"Application/transfer",metric:"Concept",assessed:"Assessed",securePct:"Secure",affected:"Affected",response:"Response",question:"Diagnostic question",foot:"Ordinal values are diagnostic signals, not grades. Activity and ratings are not treated as evidence of understanding."};
const T=REPORT.run.language==="en"?en:de;document.documentElement.lang=REPORT.run.language;
const $=id=>document.getElementById(id);const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function cards(data,isCohort){const values=[[isCohort?data.distinct_student_count:data.distinct_student_count,T.students],[data.candidate_conversation_count,T.candidates],[data.assessable_count,T.assessable],[isCohort?data.task_count:1,T.tasks]];$("cards").innerHTML=values.map(v=>`<div class="card"><strong>${v[0]}</strong><span>${v[1]}</span></div>`).join("");}
function bars(distribution){$("distribution").innerHTML=Object.entries(distribution).map(([key,v])=>`<div class="bar-row"><span>${esc(T[key]||key)}</span><div class="track"><div class="bar ${key==="secure"?"good":key==="misconception"?"bad":""}" style="width:${v.percent}%"></div></div><b>${v.count} · ${v.percent}%</b></div>`).join("");}
function dimensions(values){$("dimensions").innerHTML=Object.entries(values).map(([key,v])=>`<div class="bar-row"><span>${esc(T[key]||key)}</span><div class="track"><div class="bar good" style="width:${v==null?0:v/3*100}%"></div></div><b>${v==null?"–":v.toFixed(2)}</b></div>`).join("");}
function conceptTable(items){if(!items?.length)return `<div class="empty">${T.empty}</div>`;return `<table><thead><tr><th>${T.metric}</th><th>${T.assessed}</th><th>${T.securePct}</th></tr></thead><tbody>${items.map(x=>`<tr><td>${esc(x.name)}</td><td>${x.assessed_count}</td><td>${x.secure_percent==null?"–":x.secure_percent+"%"}</td></tr>`).join("")}</tbody></table>`;}
function misconceptionTable(items){if(!items?.length)return `<div class="empty">${T.empty}</div>`;return `<table><thead><tr><th>${T.misconceptions}</th><th>${T.affected}</th><th>${T.response}</th></tr></thead><tbody>${items.map(x=>`<tr><td><b>${esc(x.label)}</b></td><td>${x.count}${x.percent_of_assessable==null?"":" · "+x.percent_of_assessable+"%"}</td><td>${esc(x.lecturer_response)}<br><span class="muted">${T.question}: ${esc(x.diagnostic_question)}</span></td></tr>`).join("")}</tbody></table>`;}
function actionList(items){if(!items?.length)return `<div class="empty">${T.empty}</div>`;return `<ol class="actions">${items.map(x=>`<li><b>${esc(x.task_title)}: ${esc(x.issue)}</b><p>${esc(x.recommendation)}</p><p class="muted">${T.question}: ${esc(x.diagnostic_question)}</p></li>`).join("")}</ol>`;}
function render(value){const task=value==="all"?null:REPORT.tasks.find(x=>x.task_id===value);const data=task||REPORT.cohort;cards(data,!task);bars(data.understanding_distribution);dimensions(data.dimension_averages);$("concepts").innerHTML=task?conceptTable(task.concepts):`<div class="empty">${T.task}: ${T.all}</div>`;$("misconceptions").innerHTML=task?misconceptionTable(task.misconceptions):misconceptionTable(REPORT.tasks.flatMap(x=>x.misconceptions.map(m=>({...m,label:x.title+": "+m.label}))));$("actions").innerHTML=actionList(data.lecturer_actions);}
$("title").textContent=T.title;$("subtitle").textContent=T.subtitle;$("taskLabel").textContent=T.task;$("distributionTitle").textContent=T.distribution;$("dimensionTitle").textContent=T.dimensions;$("conceptTitle").textContent=T.concepts;$("misconceptionTitle").textContent=T.misconceptions;$("actionTitle").textContent=T.actions;$("foot").textContent=T.foot;
$("meta").innerHTML=`${T.generated}: ${esc(REPORT.run.generated_at)}<br>${T.model}: ${esc(REPORT.run.provider)} / ${esc(REPORT.run.model)}`;
$("warnings").innerHTML=REPORT.warnings.map(x=>`<div class="warning">${esc(x)}</div>`).join("")+(REPORT.run.complete?"":`<div class="warning error">Report incomplete: ${REPORT.errors.length} error(s)</div>`);
const select=$("taskSelect");select.innerHTML=`<option value="all">${T.all}</option>`+REPORT.tasks.map(x=>`<option value="${esc(x.task_id)}">${esc(x.title)}${x.low_evidence?" · "+T.low:""}</option>`).join("");select.addEventListener("change",()=>render(select.value));render("all");
</script></body></html>'''
