#!/usr/bin/env python3
"""
Split curated_qa.jsonl into author-attributed files with full schema compliance.

This script:
1. Reads curated_qa.jsonl line by line
2. Uses heuristics to detect author/source boundaries
3. Outputs to training_data/sources/{category}/{author}.jsonl
4. Adds full metadata per qa_record.schema.json

Usage:
    python scripts/split_curated_qa.py --dry-run  # Preview without writing
    python scripts/split_curated_qa.py            # Execute split
    python scripts/split_curated_qa.py --verify   # Post-split validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# AUTHOR DETECTION PATTERNS
# =============================================================================
# Order matters: more specific patterns should come first
# Each pattern: (regex, author_key, work_title, namespace, category)

AUTHOR_PATTERNS: list[tuple[str, str, str, str, str]] = [
    # George Jackson
    (
        r"George Jackson|Blood in My Eye|Black colony|Soledad Brother",
        "george_jackson",
        "Blood in My Eye",
        "Library",
        "revolutionary_strategy",
    ),
    # Victor Serge
    (
        r"Victor Serge|Okhrana|agent provocateur|Malinovsky|What Everyone Should Know",
        "victor_serge",
        "What Everyone Should Know About Repression",
        "Library",
        "revolutionary_strategy",
    ),
    # Losurdo
    (
        r"Losurdo|Liberalism.*Counter-History|master-race democracy|exclusion clause|community of the free|Calhoun",
        "losurdo",
        "Liberalism: A Counter-History",
        "Library",
        "historiography",
    ),
    # Fanon
    (
        r"Fanon|Wretched of the Earth|colonial violence|colonized.*mental|national consciousness|lumpenproletariat",
        "fanon",
        "The Wretched of the Earth",
        "Library",
        "anti_colonial",
    ),
    # Sankara
    (
        r"Sankara|Burkina|Burkinabe|Voltaic|Thomas Sankara Speaks|UFB",
        "sankara",
        "Thomas Sankara Speaks",
        "Library",
        "anti_colonial",
    ),
    # Nkrumah
    (
        r"Nkrumah|Neo-[Cc]olonialism|balkanization",
        "nkrumah",
        "Neo-Colonialism: The Last Stage of Imperialism",
        "Library",
        "anti_colonial",
    ),
    # Dunbar-Ortiz
    (
        r"Dunbar-Ortiz|Indigenous Peoples.*History|Roxanne Dunbar|settler colonial.*genocide|Haudenosaunee|Indian wars",
        "dunbar_ortiz",
        "An Indigenous Peoples' History of the United States",
        "Library",
        "anti_colonial",
    ),
    # Pappe
    (
        r"Pappé|Pappe|Ten Myths|Nakba|1948.*ethnic cleansing",
        "pappe",
        "Ten Myths About Israel",
        "Library",
        "historiography",
    ),
    # Feinberg
    (
        r"Feinberg|Transgender Liberation|Transgender Warriors|Two-Spirit|Joan of Arc.*transgender",
        "feinberg",
        "Transgender Liberation",
        "Library",
        "feminist_marxism",
    ),
    # Assata Shakur
    (r"Assata|Shakur", "assata_shakur", "Assata: An Autobiography", "Library", "feminist_marxism"),
    # Cohen (Psychiatric Hegemony)
    (
        r"Cohen.*psychiatric|Psychiatric Hegemony|ADHD.*diagnosis|DSM.*expansion|Rosenhan experiment",
        "cohen_psychiatric",
        "Psychiatric Hegemony",
        "Library",
        "disability_studies",
    ),
    # Disability History
    (
        r"Disability History|Section 504|ugly laws|ADA|League of the Physically Handicapped|Edward Roberts",
        "disability_history",
        "A Disability History of the United States",
        "Library",
        "disability_studies",
    ),
    # Stalin Interviews
    (
        r"Stalin said|Stalin replied|Stalin answered|Emil Ludwig|H\.G\. Wells.*Stalin|Roy Howard.*Stalin",
        "stalin_interviews",
        "Stalin Interviews",
        "Library",
        "historical_interviews",
    ),
    # Mao
    (
        r"Mao.*On Practice|On Contradiction|perceptual knowledge.*rational|guerrilla.*fish.*water",
        "mao",
        "Selected Works of Mao",
        "Library",
        "primary_theory",
    ),
    # Lenin on Revisionism
    (
        r"revisionism|Bernstein.*opportunism|parliamentarism.*betrayal|Second International.*collapse",
        "lenin_revisionism",
        "Marxism and Revisionism",
        "Library",
        "primary_theory",
    ),
    # Marx Capital
    (
        r"surplus value|constant capital|rate of profit|commodity.*fetish|Capital Vol",
        "marx_capital",
        "Capital",
        "Library",
        "primary_theory",
    ),
    # Clara Zetkin on Fascism
    (
        r"Clara Zetkin|Zetkin.*fascism|Italian fascism.*Mussolini",
        "zetkin_fascism",
        "The Struggle Against Fascism",
        "Library",
        "foundational",
    ),
    # Einstein
    (
        r"Einstein.*[Ss]ocialism|Why Socialism",
        "einstein",
        "Why Socialism?",
        "Library",
        "foundational",
    ),
    # PFLP
    (
        r"PFLP|Popular Front for the Liberation|George Habash|Palestinian.*armed struggle",
        "pflp",
        "PFLP Strategy",
        "Library",
        "revolutionary_strategy",
    ),
    # Che Guevara
    (
        r"Che Guevara|Guerrilla Warfare|foco.*insurrection|Cuban Revolution.*guerrilla",
        "che_guevara",
        "Guerrilla Warfare",
        "Library",
        "revolutionary_strategy",
    ),
    # Iranian Fedai
    (
        r"Fedai|Ahmadzadeh|Pouyan|Siahkal",
        "iranian_fedai",
        "Armed Struggle: Both a Strategy and a Tactic",
        "Library",
        "revolutionary_strategy",
    ),
    # Rodney
    (
        r"Walter Rodney|How Europe Underdeveloped|Decolonial Marxism",
        "rodney",
        "How Europe Underdeveloped Africa",
        "Library",
        "anti_colonial",
    ),
    # Immerwahr
    (
        r"Immerwahr|How to Hide an Empire|territorial empire|informal imperialism",
        "immerwahr",
        "How to Hide an Empire",
        "Library",
        "historiography",
    ),
    # Sousa (Soviet history)
    (
        r"Sousa|Lies Concerning.*Soviet|gulag statistics|anti-Soviet propaganda",
        "sousa",
        "Lies Concerning the History of the Soviet Union",
        "Library",
        "historiography",
    ),
    # AV Dremel COVID
    (
        r"[Ll]ong [Cc][Oo][Vv][Ii][Dd]|SARS-CoV-2|pandemic.*bipartisan|viral mechanism|immune evasion|JN\.1|Janus",
        "av_dremel_covid",
        "COVID Essays",
        "Essays",
        "original_essays",
    ),
    # AV Dremel Fascism (Mosse, etc.)
    (
        r"George Mosse|Mosse.*fascism|fascist bifurcation|fascist essentialism|new man.*fascism",
        "av_dremel_fascism",
        "Fascism Analysis Essays",
        "Essays",
        "original_essays",
    ),
    # AV Dremel Queer
    (
        r"queer liberation.*strategy|trans.*revolutionary|intergenerational.*queer",
        "av_dremel_queer",
        "Queer Liberation Essays",
        "Essays",
        "original_essays",
    ),
    # Persephone Labor Aristocracy
    (
        r"labor aristocracy|unequal exchange|imperial core.*proletariat|settler.*bribery|falling rate of bribery",
        "persephone_labor_aristocracy",
        "Labor Aristocracy Essays",
        "Essays",
        "original_essays",
    ),
    # Persephone Political Economy
    (
        r"YIMBY|Billionaires Row|vanilla.*imperialism|basal production|paycheck to paycheck.*misleading",
        "persephone_political_economy",
        "Political Economy Essays",
        "Essays",
        "original_essays",
    ),
    # LGBT Essay
    (
        r"LGBT.*dialectics|biological sex.*social|gender.*identity politics",
        "lgbt_essay",
        "The LGBT Question",
        "Essays",
        "feminist_marxism",
    ),
    # Butler (War is a Racket)
    (
        r"Smedley Butler|War is a Racket|military-industrial.*racket",
        "butler",
        "War is a Racket",
        "Library",
        "anti_colonial",
    ),
    # Game Theory and Organizational Strategy
    (
        r"game theory|Nash equilibrium|percolation theory|organizational.*simulation",
        "organizational_theory",
        "Organizational Theory",
        "Essays",
        "original_essays",
    ),
    # General Dialectics and Philosophy
    (
        r"dialectical materialism|essence.*appearance|quantity.*quality|negation of.*negation|unity of opposites",
        "dialectics",
        "Dialectical Materialism",
        "Main",
        "primary_theory",
    ),
    # Historical materialism
    (
        r"historical materialism|mode of production|base.*superstructure|relations of production",
        "historical_materialism",
        "Historical Materialism",
        "Main",
        "primary_theory",
    ),
    # Class analysis
    (
        r"class consciousness|proletariat|bourgeoisie|petty.*bourgeois|class.*fraction",
        "class_analysis",
        "Class Analysis",
        "Main",
        "primary_theory",
    ),
    # State theory
    (
        r"dictatorship of the proletariat|withering away.*state|proletarian state|bourgeois state",
        "state_theory",
        "State and Revolution",
        "Library",
        "primary_theory",
    ),
    # Imperialism theory
    (
        r"monopoly capital|finance capital|export of capital|super-profits|Lenin.*imperialism",
        "imperialism_theory",
        "Imperialism: The Highest Stage",
        "Library",
        "primary_theory",
    ),
    # Anti-revisionism
    (
        r"anti-revisionism|modern revisionism|Khrushchev.*revisionism|Great Debate|Sino-Soviet",
        "anti_revisionism",
        "Anti-Revisionist Theory",
        "Main",
        "primary_theory",
    ),
    # Cultural Revolution
    (
        r"Cultural Revolution|GPCR|Great Proletarian|Red Guards|capitalist roaders",
        "cultural_revolution",
        "Cultural Revolution",
        "Library",
        "primary_theory",
    ),
    # Soviet History
    (
        r"Soviet Union|USSR|collectivization|Five.*Year.*Plan|New Economic Policy|NEP",
        "soviet_history",
        "Soviet History",
        "Main",
        "historiography",
    ),
    # Anti-imperialism general
    (
        r"anti-imperialist|anti-imperialism|third world|Global South|national liberation",
        "anti_imperialism",
        "Anti-Imperialism",
        "Main",
        "anti_colonial",
    ),
    # Palestinian Resistance
    (
        r"Palestinian|Palestine|intifada|occupation|Hamas|Zionist|IOF|resistance.*armed",
        "palestine",
        "Palestinian Resistance",
        "Main",
        "anti_colonial",
    ),
    # China analysis
    (
        r"CPC|Chinese Communist|Deng Xiaoping|Xi Jinping|reform and opening|socialism.*Chinese",
        "china_analysis",
        "China Analysis",
        "Main",
        "primary_theory",
    ),
    # Korean history
    (
        r"DPRK|North Korea|Kim Il-sung|Juche|Korean War",
        "korea",
        "Korean History",
        "Main",
        "historiography",
    ),
    # Cuban Revolution
    (
        r"Cuban Revolution|Castro|Bay of Pigs|Cuban.*socialism|blockade.*Cuba",
        "cuba",
        "Cuban Revolution",
        "Main",
        "anti_colonial",
    ),
    # African socialism
    (
        r"African socialism|Pan-African|OAU|African Unity|Nyerere|ujamaa",
        "african_socialism",
        "African Socialism",
        "Main",
        "anti_colonial",
    ),
    # US leftism critique
    (
        r"DSA|Democratic Socialists|social democracy|reformism|tailism|opportunism.*left",
        "us_left_critique",
        "US Left Critique",
        "Essays",
        "original_essays",
    ),
    # Organizational practice
    (
        r"democratic centralism|vanguard party|cadre|mass line|party discipline",
        "org_practice",
        "Organizational Practice",
        "Main",
        "revolutionary_strategy",
    ),
    # Revolutionary violence
    (
        r"revolutionary violence|armed struggle|insurrection|people's war|protracted war",
        "rev_violence",
        "Revolutionary Violence",
        "Main",
        "revolutionary_strategy",
    ),
]

# Display names for authors
AUTHOR_DISPLAY_NAMES: dict[str, str] = {
    "george_jackson": "George Jackson",
    "victor_serge": "Victor Serge",
    "losurdo": "Domenico Losurdo",
    "fanon": "Frantz Fanon",
    "sankara": "Thomas Sankara",
    "nkrumah": "Kwame Nkrumah",
    "dunbar_ortiz": "Roxanne Dunbar-Ortiz",
    "pappe": "Ilan Pappé",
    "feinberg": "Leslie Feinberg",
    "assata_shakur": "Assata Shakur",
    "cohen_psychiatric": "Bruce Cohen",
    "disability_history": "Kim E. Nielsen",
    "stalin_interviews": "Joseph Stalin",
    "mao": "Mao Zedong",
    "lenin_revisionism": "Vladimir Lenin",
    "marx_capital": "Karl Marx",
    "zetkin_fascism": "Clara Zetkin",
    "einstein": "Albert Einstein",
    "pflp": "PFLP",
    "che_guevara": "Che Guevara",
    "iranian_fedai": "Iranian Fedai",
    "rodney": "Walter Rodney",
    "immerwahr": "Daniel Immerwahr",
    "sousa": "Mario Sousa",
    "av_dremel_covid": "AV Dremel",
    "av_dremel_fascism": "AV Dremel",
    "av_dremel_queer": "AV Dremel",
    "persephone_labor_aristocracy": "Persephone Raskova",
    "persephone_political_economy": "Persephone Raskova",
    "lgbt_essay": "ProleWiki Contributors",
    "butler": "Smedley Butler",
    "organizational_theory": "ProleWiki Contributors",
    "dialectics": "ProleWiki Contributors",
    "historical_materialism": "ProleWiki Contributors",
    "class_analysis": "ProleWiki Contributors",
    "state_theory": "Vladimir Lenin",
    "imperialism_theory": "Vladimir Lenin",
    "anti_revisionism": "ProleWiki Contributors",
    "cultural_revolution": "ProleWiki Contributors",
    "soviet_history": "ProleWiki Contributors",
    "anti_imperialism": "ProleWiki Contributors",
    "palestine": "ProleWiki Contributors",
    "china_analysis": "ProleWiki Contributors",
    "korea": "ProleWiki Contributors",
    "cuba": "ProleWiki Contributors",
    "african_socialism": "ProleWiki Contributors",
    "us_left_critique": "ProleWiki Contributors",
    "org_practice": "ProleWiki Contributors",
    "rev_violence": "ProleWiki Contributors",
}

# Category display names
CATEGORY_DIRS: dict[str, str] = {
    "primary_theory": "primary_theory",
    "anti_colonial": "anti_colonial",
    "historiography": "historiography",
    "revolutionary_strategy": "revolutionary_strategy",
    "feminist_marxism": "feminist_marxism",
    "disability_studies": "disability_studies",
    "historical_interviews": "historical_interviews",
    "original_essays": "original_essays",
    "foundational": "foundational",
    "uncategorized": "uncategorized",
}


@dataclass
class DetectedSource:
    """Result of source detection for a record."""

    author_key: str
    work_title: str
    namespace: str
    category: str
    confidence: str  # high, medium, low
    matched_pattern: str | None = None


@dataclass
class SourceFile:
    """Accumulator for records going to a specific output file."""

    author_key: str
    category: str
    namespace: str
    work_title: str
    records: list[dict[str, Any]] = field(default_factory=list)


def detect_source(instruction: str, response: str) -> DetectedSource:
    """
    Detect author/source from content using pattern matching.

    Returns DetectedSource with author_key, work, namespace, category, and confidence.
    """
    combined = f"{instruction} {response}"

    for pattern, author_key, work_title, namespace, category in AUTHOR_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return DetectedSource(
                author_key=author_key,
                work_title=work_title,
                namespace=namespace,
                category=category,
                confidence="high",
                matched_pattern=pattern,
            )

    # No match found - return uncategorized
    return DetectedSource(
        author_key="uncategorized",
        work_title="Unknown",
        namespace="Library",
        category="uncategorized",
        confidence="low",
        matched_pattern=None,
    )


def infer_categories(instruction: str, response: str, detected: DetectedSource) -> list[str]:
    """Infer classification categories from content."""
    categories: set[str] = set()
    combined = f"{instruction} {response}".lower()

    # Base category from detected source
    if detected.category != "uncategorized":
        categories.add(detected.category.replace("_", "-"))

    # Topic detection
    topic_patterns: dict[str, list[str]] = {
        "revisionism": ["revisionism", "bernstein", "opportunism"],
        "imperialism": ["imperialism", "imperialist", "unequal exchange"],
        "fascism": ["fascism", "fascist", "nazi"],
        "settler-colonialism": ["settler colonial", "settler-colonial", "zionist", "indigenous"],
        "anti-zionism": ["zionism", "zionist", "palestine", "nakba"],
        "national-liberation": ["national liberation", "decolonization", "anti-colonial"],
        "feminist-marxism": ["gender", "transgender", "women's liberation", "patriarchy"],
        "dialectics": ["dialectic", "contradiction", "unity of opposites"],
        "political-economy": ["surplus value", "rate of profit", "commodity"],
        "revolutionary-strategy": ["vanguard", "party", "revolution", "armed struggle"],
    }

    for category, patterns in topic_patterns.items():
        if any(p in combined for p in patterns):
            categories.add(category)

    return sorted(categories) if categories else ["general-theory"]


def infer_tradition(response: str, detected: DetectedSource) -> str:
    """Infer Marxist tradition from content."""
    resp_lower = response.lower()

    if any(term in resp_lower for term in ["maoism", "mlm", "cultural revolution", "mass line"]):
        return "MLM"
    if detected.author_key in ["mao"]:
        return "MLM"
    if any(
        term in resp_lower for term in ["contested", "debate within", "disagreement among marxists"]
    ):
        return "contested"
    return "ML"


def generate_qa_record(
    line_num: int,
    instruction: str,
    response: str,
    detected: DetectedSource,
    file_index: int,
) -> dict[str, Any]:
    """Generate a schema-compliant Q&A record."""
    # Build qa_id in format: {namespace}/{title}#{index}
    title_slug = detected.author_key.replace("_", "-").title().replace("-", "_")
    qa_id = f"{detected.namespace}/{title_slug}#{file_index:03d}"

    categories = infer_categories(instruction, response, detected)
    tradition = infer_tradition(response, detected)

    # Check for blockquotes
    has_blockquote = bool(
        re.search(r'["\']{2,}|["""].*["""]', response)
        or "quoted" in response.lower()
        or "said:" in response.lower()
    )

    record: dict[str, Any] = {
        "qa_id": qa_id,
        "instruction": instruction,
        "response": response,
        "source": {
            "namespace": detected.namespace,
            "article_title": detected.work_title,
            "author": AUTHOR_DISPLAY_NAMES.get(detected.author_key, detected.author_key),
            "work": detected.work_title,
        },
        "classification": {
            "categories": categories,
            "tradition": tradition,
        },
        "quality": {
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": has_blockquote,
            "human_verified": True,
            "confidence": detected.confidence,
        },
        "provenance": {
            "created_date": "2025-12-17",
            "created_by": "claude-opus",
            "version": 1,
        },
    }

    # Add original line number for traceability
    record["_original_line"] = line_num

    return record


def process_curated_qa(input_path: Path, _dry_run: bool = False) -> dict[str, SourceFile]:
    """
    Process curated_qa.jsonl and group records by detected source.

    Returns dict of author_key -> SourceFile with accumulated records.
    """
    source_files: dict[str, SourceFile] = {}
    file_indices: dict[str, int] = defaultdict(int)  # Track index per author

    with input_path.open() as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line)
            instruction = record.get("instruction", "")
            response = record.get("response", "")

            # Detect source
            detected = detect_source(instruction, response)

            # Get or create source file accumulator
            if detected.author_key not in source_files:
                source_files[detected.author_key] = SourceFile(
                    author_key=detected.author_key,
                    category=detected.category,
                    namespace=detected.namespace,
                    work_title=detected.work_title,
                )

            # Generate schema-compliant record
            file_indices[detected.author_key] += 1
            qa_record = generate_qa_record(
                line_num,
                instruction,
                response,
                detected,
                file_indices[detected.author_key],
            )

            source_files[detected.author_key].records.append(qa_record)

    return source_files


def write_source_files(
    source_files: dict[str, SourceFile],
    output_base: Path,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Write source files to disk.

    Returns dict of file paths -> metadata for manifest update.
    """
    file_metadata: dict[str, dict[str, Any]] = {}

    for author_key, source_file in source_files.items():
        category_dir = CATEGORY_DIRS.get(source_file.category, "uncategorized")
        output_dir = output_base / category_dir
        output_path = output_dir / f"{author_key}.jsonl"

        if dry_run:
            print(f"[DRY-RUN] Would write {len(source_file.records)} records to {output_path}")
            continue

        # Create directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write records
        with output_path.open("w") as f:
            for record in source_file.records:
                # Remove internal tracking field
                record_copy = {k: v for k, v in record.items() if not k.startswith("_")}
                f.write(json.dumps(record_copy) + "\n")

        # Calculate SHA-256
        sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

        file_metadata[str(output_path.relative_to(output_base.parent))] = {
            "filename": str(output_path.relative_to(output_base.parent)),
            "record_count": len(source_file.records),
            "sha256": sha256,
            "author": AUTHOR_DISPLAY_NAMES.get(author_key, author_key),
            "work": source_file.work_title,
            "category": source_file.category,
        }

        print(f"Wrote {len(source_file.records)} records to {output_path}")

    return file_metadata


def print_summary(source_files: dict[str, SourceFile]) -> None:
    """Print summary of detected sources."""
    total = sum(len(sf.records) for sf in source_files.values())

    print("\n" + "=" * 70)
    print("SOURCE DETECTION SUMMARY")
    print("=" * 70)

    # Group by category
    by_category: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for author_key, sf in source_files.items():
        by_category[sf.category].append((author_key, len(sf.records)))

    for category in sorted(by_category.keys()):
        print(f"\n{category.upper().replace('_', ' ')}:")
        for author_key, count in sorted(by_category[category], key=lambda x: -x[1]):
            author_name = AUTHOR_DISPLAY_NAMES.get(author_key, author_key)
            print(f"  {author_name:40} {count:4} records")

    print("\n" + "-" * 70)
    print(f"TOTAL: {total} records across {len(source_files)} sources")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split curated_qa.jsonl into author-attributed files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing files",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training_data/curated_qa.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/sources"),
        help="Output directory for source files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed per-record output",
    )
    args = parser.parse_args()

    # Process input file
    print(f"Processing {args.input}...")
    source_files = process_curated_qa(args.input, args.dry_run)

    # Print summary
    print_summary(source_files)

    # Verbose per-record output
    if args.verbose:
        print("\nDETAILED RECORD ASSIGNMENTS:")
        for author_key, sf in sorted(source_files.items()):
            print(f"\n--- {author_key} ---")
            for rec in sf.records[:3]:  # Show first 3 per source
                print(f"  Line {rec['_original_line']}: {rec['instruction'][:60]}...")

    # Write output files
    if not args.dry_run:
        file_metadata = write_source_files(source_files, args.output, args.dry_run)
        print(f"\nWrote {len(file_metadata)} source files to {args.output}")
    else:
        print("\n[DRY-RUN] No files written. Run without --dry-run to execute.")


if __name__ == "__main__":
    main()
