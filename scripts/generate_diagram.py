#!/usr/bin/env python3
"""Generate Mermaid architecture diagrams from the codebase.

Produces separate .mmd files per concern in docs/ and updates
diagram image references in README.md between marker pairs.

Usage:
    python scripts/generate_diagram.py
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PIPELINE = REPO / "src" / "invoices" / "pipeline.py"
QUEUE = REPO / "src" / "invoices" / "queue.py"
COMPOSE = REPO / "docker-compose.yml"
OUT_DIR = REPO / "docs"
README = REPO / "README.md"

DIAGRAM_LABELS = {
    "architecture": "Architecture",
    "docker-services": "Docker Services",
    "data-flow": "Data Flow",
    "ml-loop": "Active Learning Loop",
    "azure-integration": "Azure Integration",
}

# -- Palette (Base16 Embers) -------------------------------------------------

STYLES = {
    "node": "fill:#40312F,stroke:#51504B,color:#dbd6d1",
    "accent": "fill:#46696B,stroke:#69AAB2,color:#dbd6d1",
    "io": "fill:#2A1B16,stroke:#778C89,color:#dbd6d1",
    "decision": "fill:#5C5A72,stroke:#C6C2D2,color:#dbd6d1",
}


def style_defs() -> list[str]:
    return [
        "",
        "    %% Palette",
        *(f"    classDef {k} {v}" for k, v in STYLES.items()),
    ]


def write_diagram(name: str, lines: list[str]) -> None:
    path = OUT_DIR / f"{name}.mmd"
    path.write_text("\n".join(lines) + "\n")
    print(f"  {path.relative_to(REPO)} ({len(lines)} lines)")


# -- 1. Pipeline (hero diagram) ---------------------------------------------


def parse_pipeline_stages() -> list[str]:
    # pipeline.py documents stages in its docstring: tokenize → candidates → decode → emit.
    # The function run_document_pipeline chains these in order; we derive the list from
    # its inline comments rather than parsing a bulk import (orchestrator.py is deleted).
    text = PIPELINE.read_text()
    # Scan for "Stage N:" comments inside run_document_pipeline to confirm order.
    stages = []
    for line in text.splitlines():
        if "Stage 1" in line or "Tokenize" in line.split("#")[-1]:
            if "tokenize" not in stages:
                stages.append("tokenize")
        elif "Stage 2" in line or "Candidates" in line.split("#")[-1]:
            if "candidates" not in stages:
                stages.append("candidates")
        elif "Stage 3" in line or "Decode" in line.split("#")[-1]:
            if "decode" not in stages:
                stages.append("decode")
        elif "Stage 4" in line or "Emit" in line.split("#")[-1]:
            if "emit" not in stages:
                stages.append("emit")
    return stages if len(stages) == 4 else ["tokenize", "candidates", "decode", "emit"]


def build_architecture() -> list[str]:
    stages = parse_pipeline_stages()
    lines = ["flowchart LR"]
    lines.append('    PDF(["PDF"]):::io')
    for i, stg in enumerate(stages):
        prev = "PDF" if i == 0 else f"stg_{stages[i - 1]}"
        lines.append(f"    {prev} --> stg_{stg}[{stg}]:::accent")
    lines.append(f'    stg_{stages[-1]} --> JSON(["JSON"]):::io')
    lines += style_defs()
    return lines


# -- 2. Docker services -----------------------------------------------------

SERVICE_GROUPS = {
    "Core": ["worker", "postgrest", "nginx"],
    "Storage": ["postgres"],
    "ML": ["mlflow"],
    "UI": ["grafana"],
    "Observability": ["otel-collector", "tempo", "loki", "prometheus", "alertmanager"],
}


def build_docker_services() -> list[str]:
    compose = yaml.safe_load(COMPOSE.read_text())
    services = compose.get("services", {})
    lines = [
        "%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%",
        "flowchart TD",
    ]

    for group, members in SERVICE_GROUPS.items():
        present = [m for m in members if m in services]
        if not present:
            continue
        safe = group.replace(" ", "")
        spec0 = services.get(present[0], {})
        suffix = f" ({spec0['profiles'][0]})" if spec0.get("profiles") else ""
        lines.append(f'    subgraph grp_{safe}["{group}{suffix}"]')
        for svc in present:
            spec = services[svc]
            port = ""
            for p in spec.get("ports", []):
                port = f" :{str(p).split(':')[0]}"
            for p in spec.get("expose", []):
                port = port or f" :{p}"
            sid = svc.replace("-", "_")
            lines.append(f'        svc_{sid}["{svc}{port}"]:::node')
        lines.append("    end")

    for svc_name, spec in services.items():
        deps = spec.get("depends_on", {})
        if isinstance(deps, list):
            deps = {d: {} for d in deps}
        src = svc_name.replace("-", "_")
        lines.extend(f"    svc_{src} --> svc_{dep.replace('-', '_')}" for dep in deps)

    lines.append('    User(["User :80"]):::io --> svc_nginx')
    lines += style_defs()
    return lines


# -- 3. Data flow ------------------------------------------------------------


def build_data_flow() -> list[str]:
    # Valid backends (source of truth: src/invoices/settings.py):
    #   DOCUMENT_SOURCE: "none" | "sharepoint"
    #   STORAGE_BACKEND: "blob"  (MLflow model artifacts on Azure blob)
    #   OUTPUT_BACKEND:  "postgres" | "dataverse"
    return [
        "flowchart LR",
        '    sp["SharePoint\\nPDFs"]:::io --> worker["worker\\n(pgqueuer + pipeline)"]:::accent',
        '    worker --> docs["docs\\n(JSONB)"]:::node',
        '    worker --> labels["labels\\n(JSONB)"]:::node',
        '    worker --> evals["doc_evaluations\\n(JSONB)"]:::node',
        '    worker --> schema["contract_schema\\n(JSONB)"]:::node',
        '    worker --> mlflow["MLflow\\n(model artifacts)"]:::node',
        '    docs -- labels --> retrain["XGBoost Retrain"]:::accent',
        "    retrain --> mlflow",
        '    postgrest["PostgREST\\nHTTP face"]:::accent --> docs',
        "    postgrest --> labels",
        *style_defs(),
    ]


# -- 4. ML / Active Learning loop -------------------------------------------


def build_ml_loop() -> list[str]:
    # Source of truth: handle_retrain in src/invoices/queue.py.
    # No quality gate gates promotion. quality_gate_passed is written to the
    # manifest and read nowhere.
    return [
        "flowchart LR",
        '    labels["labels table\\nGrafana plugin"]:::node --> retrain["retrain entrypoint\\ncron 0 3 * * * or button"]:::accent',
        '    retrain --> join["latest doc_evaluations\\nper labeled doc and field\\nbuild_training_rows"]:::node',
        '    join --> fit["InvoiceFieldRanker.train\\nXGBRanker on all labeled rows"]:::accent',
        '    fit --> mlflow["MLflow log and register\\n+ manifest.json"]:::accent',
        '    mlflow --> runs["model_runs row"]:::io',
        '    runs --> reextract["reextract\\nevery labeled doc"]:::accent',
        '    join -- "no predictions or no rows" --> reextract',
        '    reextract --> queue["queue\\nre-scored under new model"]:::node',
        *style_defs(),
    ]


# -- 5. Azure integration ---------------------------------------------------


def build_azure_integration() -> list[str]:
    # Valid backends (source of truth: src/invoices/settings.py):
    #   DOCUMENT_SOURCE: "none" | "sharepoint"
    #   STORAGE_BACKEND: "blob"  (MLflow model artifacts only)
    #   OUTPUT_BACKEND:  "postgres" | "dataverse"
    # No filesystem backend exists for any of the three.
    return [
        "flowchart LR",
        '    sp["SharePoint"]:::accent -- DOCUMENT_SOURCE --> doc["Document Source"]:::node',
        '    blob["Blob Storage\\n(MLflow model artifacts)"]:::accent -- STORAGE_BACKEND --> store["Storage Backend"]:::node',
        '    dv["Dataverse"]:::accent -- OUTPUT_BACKEND --> out["Output Backend"]:::node',
        '    pg["Postgres"]:::node -- OUTPUT_BACKEND --> out',
        *style_defs(),
    ]


# -- README updater ----------------------------------------------------------


def update_readme() -> None:
    """Replace content between <!-- diagram:X --> markers in README.md."""
    if not README.exists():
        return

    text = README.read_text()
    marker_re = re.compile(
        r"(<!-- diagram:(\S+) -->\n).*?\n(<!-- /diagram:\2 -->)",
        re.DOTALL,
    )

    def _replace(m: re.Match) -> str:
        name = m.group(2)
        label = DIAGRAM_LABELS.get(name, name)
        return f"{m.group(1)}![{label}](docs/{name}.svg)\n{m.group(3)}"

    text = marker_re.sub(_replace, text)
    README.write_text(text)


# -- Main --------------------------------------------------------------------

DIAGRAMS = {
    "architecture": build_architecture,
    "docker-services": build_docker_services,
    "data-flow": build_data_flow,
    "ml-loop": build_ml_loop,
    "azure-integration": build_azure_integration,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(DIAGRAMS)} diagrams:")
    for name, builder in DIAGRAMS.items():
        write_diagram(name, builder())
    update_readme()
    print("Done.")


if __name__ == "__main__":
    main()
