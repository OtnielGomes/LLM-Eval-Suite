# scripts/populate_raw_docs.py
# Usage: uv run python scripts/populate_raw_docs.py
# Downloads Wikipedia articles for the 28 subjects of MMLU and saves them in raw format.

from __future__ import annotations
import time
from pathlib import Path
import urllib.request
import urllib.parse

RAW_DIR = Path(__file__).parent.parent / "src" / "llm_eval" / "datasets" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_ARTICLES: dict[str, list[str]] = {
    "abstract_algebra":               ["Abstract algebra", "Group theory", "Ring theory"],
    "anatomy":                        ["Human anatomy", "Nervous system", "Circulatory system"],
    "astronomy":                      ["Astronomy", "Solar System", "Black hole"],
    "college_biology":                ["Cell biology", "Genetics", "Evolution"],
    "college_chemistry":              ["Chemistry", "Periodic table", "Chemical bond"],
    "college_computer_science":       ["Algorithm", "Data structure", "Computer science"],
    "college_mathematics":            ["Calculus", "Linear algebra", "Probability theory"],
    "college_physics":                ["Classical mechanics", "Quantum mechanics", "Thermodynamics"],
    "conceptual_physics":             ["Physics", "Electromagnetism", "Wave"],
    "elementary_mathematics":         ["Arithmetic", "Fraction", "Geometry"],
    "formal_logic":                   ["Mathematical logic", "Propositional calculus", "Predicate logic"],
    "global_facts":                   ["World population", "Climate change", "United Nations"],
    "high_school_biology":            ["Biology", "Photosynthesis", "DNA"],
    "high_school_chemistry":          ["Acid–base reaction", "Oxidation state", "Stoichiometry"],
    "high_school_computer_science":   ["Python (programming language)", "Sorting algorithm", "Binary number"],
    "high_school_mathematics":        ["Algebra", "Trigonometry", "Statistics"],
    "high_school_physics":            ["Newton's laws of motion", "Optics", "Nuclear physics"],
    "high_school_psychology":         ["Psychology", "Cognitive psychology", "Behaviorism"],
    "high_school_world_history":      ["World history", "World War II", "Industrial Revolution"],
    "logical_fallacies":              ["Fallacy", "Ad hominem", "Straw man"],
    "machine_learning":               ["Machine learning", "Neural network", "Gradient descent"],
    "moral_scenarios":                ["Ethics", "Utilitarianism", "Deontological ethics"],
    "philosophy":                     ["Philosophy", "Epistemology", "Metaphysics"],
    "professional_accounting":        ["Accounting", "Financial statement", "Double-entry bookkeeping"],
    "professional_law":               ["Law", "Common law", "Contract"],
    "professional_medicine":          ["Medicine", "Pharmacology", "Diagnosis"],
    "professional_psychology":        ["Clinical psychology", "Psychotherapy", "Mental disorder"],
    "world_religions":                ["Religion", "Islam", "Buddhism"],
}

def fetch_wikipedia_plaintext(title: str) -> str | None:
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/w/index.php?title={encoded}&action=raw"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LLM-Eval-Suite/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")

        # Remove heavy markup while keeping the text readable.
        import re
        raw = re.sub(r"\{\{[^}]*\}\}", "", raw)        # templates
        raw = re.sub(r"\[\[File:[^\]]*\]\]", "", raw)  # imagens
        raw = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", raw)  # links [[X|Y]] → Y
        raw = re.sub(r"={2,}(.+?)={2,}", r"\n## \1\n", raw)        # headings
        raw = re.sub(r"<[^>]+>", "", raw)              # tags HTML
        raw = re.sub(r"\n{3,}", "\n\n", raw)           # linhas em branco excessivas
        return raw.strip()
    except Exception as e:
        print(f"  ⚠ Fail '{title}': {e}")
        return None

def main() -> None:
    total_articles = sum(len(v) for v in SUBJECT_ARTICLES.values())
    print(f"Downloading {total_articles} articles for {len(SUBJECT_ARTICLES)} subjects...\n")

    saved = 0
    for subject, articles in SUBJECT_ARTICLES.items():
        subject_dir = RAW_DIR / subject
        subject_dir.mkdir(exist_ok=True)
        for title in articles:
            filename = title.replace(" ", "_").replace("/", "-") + ".txt"
            dest = subject_dir / filename
            if dest.exists():
                print(f"  ↷ It already exists: {subject}/{filename}")
                continue
            print(f"  ↓ {subject}/{filename}")
            content = fetch_wikipedia_plaintext(title)
            if content and len(content) > 200:
                dest.write_text(f"# {title}\n\n{content}", encoding="utf-8")
                saved += 1
            time.sleep(0.3)  # respect Wikipedia's rate limit

    print(f"\n✅ {saved} articles saved in {RAW_DIR}")
    print("Play now: uv run ingest-docs --reset")

if __name__ == "__main__":
    main()