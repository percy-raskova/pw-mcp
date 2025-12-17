#!/usr/bin/env python3
"""
GRPO Reward Functions for Marxist-Leninist Q&A Training.

These reward functions guide the model toward:
1. Proper <think>...</think> format
2. Semantic coherence via NLI (Natural Language Inference)
3. Structural coherence via dependency parsing
4. Self-consistency (no internal contradictions)
5. Appropriate response length/completeness

Research basis:
- NLI as reward: arxiv.org/abs/2508.18212 (Better LM-Based Judging)
- MO-GRPO normalization: arxiv.org/abs/2509.22047
- Process rewards: arxiv.org/abs/2508.05170 (Posterior-GRPO)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence

# =============================================================================
# GLOBAL SETUP - LAZY LOADING
# =============================================================================

# Lazy-load models to avoid loading at import time
_embedder: SentenceTransformer | None = None
_nli_pipeline: Any | None = None
_spacy_nlp: Any | None = None


def get_embedder() -> SentenceTransformer:
    """Get or initialize the sentence transformer embedder."""
    global _embedder
    if _embedder is None:
        print("[Reward] Loading sentence-transformers embedder...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_nli_pipeline() -> Any:
    """Get or initialize the NLI pipeline (BART-large-MNLI)."""
    global _nli_pipeline
    if _nli_pipeline is None:
        print("[Reward] Loading NLI model (bart-large-mnli)...")
        from transformers import pipeline

        _nli_pipeline = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device="cuda" if _cuda_available() else "cpu",
        )
    return _nli_pipeline


def get_spacy_nlp() -> Any:
    """Get or initialize spaCy NLP pipeline.

    Uses en_core_web_trf (transformer-based) for superior semantic understanding.
    Falls back to en_core_web_md (word vectors) or en_core_web_sm if unavailable.
    """
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy

        # Try transformer model first (best semantic understanding)
        models_to_try = ["en_core_web_trf", "en_core_web_md", "en_core_web_sm"]

        for model_name in models_to_try:
            try:
                print(f"[Reward] Loading spaCy model: {model_name}...")
                _spacy_nlp = spacy.load(model_name)
                print(f"[Reward] Loaded {model_name} successfully")
                break
            except OSError:
                print(f"[Reward] {model_name} not found, trying next...")
                continue

        if _spacy_nlp is None:
            raise OSError(
                "No spaCy model found. Install one with:\n"
                "  uv pip install en_core_web_trf@https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"
            )
    return _spacy_nlp


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


# Reasoning format tokens (DeepSeek-R1 style)
REASONING_START = "<think>"
REASONING_END = "</think>"

# Regex to match format
SOLUTION_END_REGEX = re.compile(rf"{REASONING_END}(.*)", re.DOTALL)

# Marxist terminology for vocabulary reward
MARXIST_TERMS: set[str] = {
    # Core concepts
    "dialectical",
    "materialism",
    "historical materialism",
    "dialectical materialism",
    # Classes
    "bourgeoisie",
    "proletariat",
    "petty bourgeois",
    "petty bourgeoisie",
    "lumpenproletariat",
    "working class",
    "ruling class",
    # Class struggle
    "class struggle",
    "class consciousness",
    "class war",
    "class conflict",
    # Political economy
    "surplus value",
    "commodity",
    "use value",
    "exchange value",
    "labor power",
    "means of production",
    "relations of production",
    "forces of production",
    "mode of production",
    "primitive accumulation",
    "exploitation",
    "capital accumulation",
    # Imperialism
    "imperialism",
    "colonialism",
    "neo-colonialism",
    "settler colonialism",
    "national liberation",
    "self-determination",
    # State and revolution
    "dictatorship of the proletariat",
    "vanguard",
    "vanguard party",
    "democratic centralism",
    "withering away of the state",
    "proletarian dictatorship",
    # Ideology
    "hegemony",
    "superstructure",
    "base",
    "ideology",
    "false consciousness",
    # Revisionism
    "revisionism",
    "opportunism",
    "reformism",
    "social democracy",
    "ultra-leftism",
    # Alienation
    "alienation",
    "fetishism",
    "commodity fetishism",
    "reification",
    # Historical
    "paris commune",
    "october revolution",
    "bolshevik",
    "menshevik",
    # Anti-colonial
    "decolonization",
    "third world",
    "global south",
    "national bourgeoisie",
    "comprador",
}


# =============================================================================
# FORMAT REWARDS (from original notebook)
# =============================================================================


def match_format_exactly(
    completions: Sequence[Sequence[dict[str, str]]], **kwargs: object
) -> list[float]:
    """
    Reward +3.0 if response contains proper </think> tag.

    This encourages the model to use the reasoning format.
    """
    scores: list[float] = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]
        # Match if format is seen exactly
        if SOLUTION_END_REGEX.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(
    completions: Sequence[Sequence[dict[str, str]]], **kwargs: object
) -> list[float]:
    """
    Reward partial format matching.

    +0.5 for exactly one <think> tag
    +0.5 for exactly one </think> tag
    -1.0 for multiple or missing tags
    """
    scores: list[float] = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]

        # Check for proper tag counts
        start_count = response.count(REASONING_START)
        end_count = response.count(REASONING_END)

        score += 0.5 if start_count == 1 else -1.0
        score += 0.5 if end_count == 1 else -1.0

        scores.append(score)
    return scores


# =============================================================================
# SEMANTIC SIMILARITY REWARD
# =============================================================================


def semantic_similarity_reward(
    prompts: Sequence[Sequence[dict[str, str]]],
    completions: Sequence[Sequence[dict[str, str]]],
    answer: Sequence[str],
    **kwargs: object,
) -> list[float]:
    """
    Reward responses that are semantically similar to ground truth.

    Uses sentence-transformers to compute cosine similarity.

    Scoring:
        > 0.75 similarity: +5.0
        > 0.60 similarity: +3.0
        > 0.45 similarity: +1.0
        > 0.30 similarity: -1.0
        <= 0.30 similarity: -3.0
    """
    embedder = get_embedder()
    scores: list[float] = []

    for completion, true_answer in zip(completions, answer, strict=False):
        response = completion[0]["content"]

        # Extract answer after </think> if present
        if REASONING_END in response:
            response = response.split(REASONING_END, 1)[1].strip()

        # Handle empty response
        if not response or len(response.strip()) < 10:
            scores.append(-3.0)
            continue

        # Compute cosine similarity
        emb_response = embedder.encode(response, normalize_embeddings=True)
        emb_truth = embedder.encode(true_answer, normalize_embeddings=True)
        similarity = float(np.dot(emb_response, emb_truth))

        # Scale to reward
        if similarity > 0.75:
            score = 5.0
        elif similarity > 0.60:
            score = 3.0
        elif similarity > 0.45:
            score = 1.0
        elif similarity > 0.30:
            score = -1.0
        else:
            score = -3.0

        scores.append(score)

    return scores


# =============================================================================
# MARXIST TERMINOLOGY REWARD
# =============================================================================


def terminology_reward(
    completions: Sequence[Sequence[dict[str, str]]], **kwargs: object
) -> list[float]:
    """
    Reward use of proper Marxist terminology.

    +0.3 per unique term found, capped at +2.0

    NOTE: This is a shallow reward that can be gamed with "word soup".
    Consider using nli_coherence_reward or structural_coherence_reward
    for more robust evaluation.
    """
    scores: list[float] = []

    for completion in completions:
        response = completion[0]["content"].lower()

        # Count unique terms present
        term_count = sum(1 for term in MARXIST_TERMS if term in response)

        # Reward: 0.3 per term, capped at 2.0
        score = min(term_count * 0.3, 2.0)
        scores.append(score)

    return scores


# =============================================================================
# NLI-BASED COHERENCE REWARD (Research-backed)
# =============================================================================

# Discourse connectives indicating logical structure
DISCOURSE_CONNECTIVES: set[str] = {
    "because",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "however",
    "although",
    "whereas",
    "nevertheless",
    "moreover",
    "furthermore",
    "additionally",
    "specifically",
    "namely",
    "in other words",
    "for example",
    "for instance",
    "such as",
    "as a result",
    "due to",
    "in order to",
    "so that",
    "on the other hand",
    "in contrast",
    "similarly",
    "likewise",
}

# Explanatory phrases that indicate concept is being explained (not just dropped)
EXPLANATORY_PHRASES: set[str] = {
    # Causal explanations
    "because the",
    "because of",
    "this is because",
    "since the",
    "due to the",
    "as a result of",
    "results from",
    "caused by",
    "leads to",
    "results in",
    "enables",
    "produces",
    # Definitional explanations
    "is defined as",
    "refers to",
    "means that",
    "denotes",
    "that is,",
    "in other words",
    "namely",
    "i.e.",
    # Elaboration
    "specifically",
    "in particular",
    "for example",
    "such as",
    "this means",
    "which means",
    "this implies",
    "therefore",
    # Mechanism explanations
    "this occurs when",
    "this happens because",
    "the mechanism",
    "through the process of",
    "by means of",
    "works by",
}

# Hollow buzzwords: activist jargon that signals superficial analysis when used
# without substantive explanation. These are NOT Marxist technical terms.
# Penalty applies when: high density + low depth ratio
HOLLOW_BUZZWORDS: set[str] = {
    # Vague connectors (non-analytical)
    "interconnected",
    "interrelated",
    "intersects with",
    "it's all connected",
    "everything is connected",
    "systemic",
    # Performative activist language
    "centered",
    "centering",
    "uplift",
    "uplifting",
    "do the work",
    "the work",
    "unpack",
    "unpacking",
    "unlearn",
    "unlearning",
    "hold space",
    "sit with",
    "lean into",
    "problematic",
    "harmful",
    "toxic",
    # Vague abstractions without specifics
    "in a way",
    "sort of",
    "kind of",
    "essentially",
    "basically",
    "generally speaking",
    "broadly",
    # Jargon often used without definition
    "praxis",  # Valid Marxist term but often misused without explanation
    "material conditions",  # Valid but often used as hand-wave
    "structural",
    "structurally",  # Often vague without mechanism
    # Identity-focused without class analysis
    "lived experience",
    "as a",  # Often substitutes for analysis
}

# Phrases that signal analytical depth (opposite of hollow)
DEPTH_MARKERS: set[str] = {
    # Historical specificity
    "in 1",
    "in 2",
    "during the",
    "after the",
    "before the",
    # Citing sources/figures
    "marx argued",
    "lenin wrote",
    "engels noted",
    "gramsci",
    "according to",
    "as marx",
    "as lenin",
    # Concrete examples
    "for example",
    "such as",
    "in the case of",
    "consider",
    # Precise definitions
    "defined as",
    "meaning",
    "specifically",
}

# Marxist concept equivalences for topic matching
# Maps canonical term -> set of synonyms/equivalents
CONCEPT_EQUIVALENCES: dict[str, set[str]] = {
    # Class terms
    "bourgeoisie": {"capitalist class", "ruling class", "capitalists", "bourgeois", "capital"},
    "proletariat": {"working class", "workers", "wage laborers", "labor", "labourers"},
    "petty bourgeoisie": {"petit bourgeoisie", "small business", "middle class", "petty bourgeois"},
    "lumpenproletariat": {"lumpen", "underclass", "criminal element"},
    # Economic concepts
    "surplus value": {"unpaid labor", "profit", "extraction", "surplus labor"},
    "means of production": {"productive forces", "capital goods", "factories", "industry"},
    "exploitation": {"extraction", "appropriation", "expropriation"},
    "commodity": {"commodities", "goods", "merchandise"},
    "capital accumulation": {"accumulation", "concentration of capital"},
    "primitive accumulation": {"original accumulation", "so-called primitive accumulation"},
    # Political concepts
    "dictatorship of the proletariat": {
        "workers state",
        "proletarian dictatorship",
        "workers government",
    },
    "vanguard party": {"vanguard", "communist party", "revolutionary party"},
    "democratic centralism": {"party discipline", "centralism"},
    # Imperialism
    "imperialism": {"colonialism", "neo-colonialism", "empire", "colonial"},
    "national liberation": {"decolonization", "anti-colonial", "liberation movement"},
    "settler colonialism": {"settler colony", "colonial settlement"},
    # Ideology
    "revisionism": {"opportunism", "reformism", "right deviation"},
    "hegemony": {"ideological hegemony", "cultural hegemony", "domination"},
    "false consciousness": {"ideology", "mystification"},
    # Philosophy
    "dialectical materialism": {"diamat", "materialist dialectics", "dialectics"},
    "historical materialism": {"histmat", "materialist conception of history"},
    "alienation": {"estrangement", "alienated labor"},
}

# Question words to ignore when extracting topics
QUESTION_WORDS: set[str] = {"what", "how", "why", "who", "when", "where", "which", "whom"}


def nli_coherence_reward(
    completions: Sequence[Sequence[dict[str, str]]],
    answer: Sequence[str],
    **kwargs: object,
) -> list[float]:
    """
    Reward responses that logically ENTAIL the ground truth answer.

    Uses Natural Language Inference (facebook/bart-large-mnli) to check
    if the response is logically consistent with the expected answer.

    This defeats "word soup" attacks because random terminology won't
    logically entail anything - it will be classified as NEUTRAL.

    Scoring:
        entailment: +3.0 (response supports/implies ground truth)
        neutral: -1.0 (response is off-topic or incoherent)
        contradiction: -3.0 (response contradicts ground truth)

    Research basis: arxiv.org/abs/2508.18212
    """
    nli = get_nli_pipeline()
    scores: list[float] = []

    for completion, true_answer in zip(completions, answer, strict=False):
        response = completion[0]["content"]

        # Extract answer part after </think>
        if REASONING_END in response:
            response = response.split(REASONING_END, 1)[1].strip()

        # Handle empty or very short responses
        if not response or len(response.strip()) < 20:
            scores.append(-2.0)
            continue

        # Truncate to model max length (prevent OOM)
        response_truncated = response[:512]
        truth_truncated = true_answer[:512]

        # NLI classification: premise </s></s> hypothesis
        # We check: Does response entail ground truth?
        try:
            input_text = f"{response_truncated}</s></s>{truth_truncated}"
            result = nli(input_text)[0]
            label = result["label"].lower()

            if label == "entailment":
                score = 3.0
            elif label == "neutral":
                score = -1.0
            else:  # contradiction
                score = -3.0

            scores.append(score)

        except Exception as e:
            print(f"[NLI Reward] Error: {e}")
            scores.append(0.0)

    return scores


def self_consistency_reward(
    completions: Sequence[Sequence[dict[str, str]]], **kwargs: object
) -> list[float]:
    """
    Reward responses that are internally self-consistent.

    Checks if any sentence in the response CONTRADICTS another sentence.
    This avoids external ideological bias by only checking within-document
    coherence.

    Scoring:
        No contradictions found: +1.0
        Internal contradiction detected: -2.0

    Research basis: arxiv.org/abs/2508.05170 (process-based rewards)
    """
    nli = get_nli_pipeline()
    nlp = get_spacy_nlp()
    scores: list[float] = []

    for completion in completions:
        response = completion[0]["content"]

        # Parse into sentences
        doc = nlp(response)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

        # Need at least 2 sentences to check consistency
        if len(sentences) < 2:
            scores.append(0.0)
            continue

        # Check pairs of sentences for contradictions
        # (Only check adjacent and near-adjacent to limit compute)
        has_contradiction = False
        max_pairs_to_check = 10
        pairs_checked = 0

        for i, sent_a in enumerate(sentences[:-1]):
            if pairs_checked >= max_pairs_to_check:
                break
            # Check against next 2 sentences
            for j in range(i + 1, min(i + 3, len(sentences))):
                sent_b = sentences[j]
                try:
                    input_text = f"{sent_a[:256]}</s></s>{sent_b[:256]}"
                    result = nli(input_text)[0]
                    if result["label"].lower() == "contradiction":
                        has_contradiction = True
                        break
                    pairs_checked += 1
                except Exception:
                    pass
            if has_contradiction:
                break

        if has_contradiction:
            scores.append(-2.0)
        else:
            scores.append(1.0)

    return scores


def structural_coherence_reward(
    completions: Sequence[Sequence[dict[str, str]]], **kwargs: object
) -> list[float]:
    """
    Reward responses with proper linguistic structure.

    Uses spaCy dependency parsing to verify:
    1. Marxist terms appear in meaningful syntactic roles (subject, object)
    2. Response contains logical discourse connectives
    3. Response has proper sentence structure (not word soup)

    This defeats word soup because random terms won't be in subject/object
    positions - they'll be parsed as fragments.

    Scoring:
        +0.3 per term in subject/object position (max +1.5)
        +0.2 per discourse connective (max +1.0)
        -1.0 if no complete sentences detected

    Research basis: spaCy dependency parsing for coherence evaluation
    """
    nlp = get_spacy_nlp()
    scores: list[float] = []

    for completion in completions:
        response = completion[0]["content"]
        doc = nlp(response)
        score = 0.0

        # Check 1: Are there actual sentences?
        sentences = list(doc.sents)
        if len(sentences) < 1:
            scores.append(-1.0)
            continue

        # Check 2: Marxist terms in meaningful syntactic roles
        terms_in_context = 0
        response_lower = response.lower()

        for term in MARXIST_TERMS:
            if term not in response_lower:
                continue

            # Find tokens matching this term
            for token in doc:
                if term in token.text.lower() or (
                    token.i + 1 < len(doc)
                    and term in f"{token.text} {doc[token.i + 1].text}".lower()
                ):
                    # Reward if token is in a meaningful syntactic role
                    if token.dep_ in (
                        "nsubj",  # nominal subject
                        "nsubjpass",  # passive nominal subject
                        "dobj",  # direct object
                        "pobj",  # object of preposition
                        "attr",  # attribute
                        "appos",  # appositional modifier
                    ):
                        terms_in_context += 1
                        break  # Count each term once
                    # Also reward if connected to a meaningful verb
                    elif token.head.pos_ == "VERB" and token.head.dep_ == "ROOT":
                        terms_in_context += 1
                        break

        score += min(terms_in_context * 0.3, 1.5)

        # Check 3: Discourse connectives (indicates logical structure)
        connective_count = sum(1 for conn in DISCOURSE_CONNECTIVES if conn in response_lower)
        score += min(connective_count * 0.2, 1.0)

        scores.append(score)

    return scores


# =============================================================================
# COMBINED ROBUST COHERENCE REWARD
# =============================================================================


def robust_coherence_reward(
    completions: Sequence[Sequence[dict[str, str]]],
    answer: Sequence[str],
    **kwargs: object,
) -> list[float]:
    """
    Multi-layered coherence check combining NLI, self-consistency, and structure.

    This is the recommended reward function for robust evaluation that defeats
    reward hacking via word soup or other adversarial strategies.

    Layers:
    1. NLI coherence: Does response entail ground truth?
    2. Self-consistency: Does response contradict itself?
    3. Structural coherence: Are terms used in meaningful syntactic roles?

    Scoring (combined):
        NLI entailment + self-consistent + good structure: up to +5.5
        NLI neutral or contradiction: -1.0 to -3.0
        Internal contradiction: -2.0
        Word soup (no structure): -1.0
    """
    # Get individual scores
    nli_scores = nli_coherence_reward(completions, answer, **kwargs)
    consistency_scores = self_consistency_reward(completions, **kwargs)
    structure_scores = structural_coherence_reward(completions, **kwargs)

    # Combine with weights
    combined: list[float] = []
    for nli, consistency, structure in zip(
        nli_scores, consistency_scores, structure_scores, strict=False
    ):
        # If NLI shows contradiction, heavily penalize regardless of other scores
        if nli <= -3.0:
            combined.append(-3.0)
        # If internal contradiction, penalize
        elif consistency <= -2.0:
            combined.append(-2.0)
        # Otherwise combine scores
        else:
            # NLI is primary signal, structure and consistency are bonuses
            total = nli + (consistency * 0.5) + (structure * 0.5)
            combined.append(total)

    return combined


# =============================================================================
# TOPIC RELEVANCE REWARD (Question-Answer Alignment)
# =============================================================================


def _extract_noun_with_preps(token: Any) -> set[str]:
    """
    Extract a noun and its prepositional phrase children.

    For "dictatorship of the proletariat", returns:
    {"dictatorship", "proletariat", "dictatorship of the proletariat"}
    """
    topics: set[str] = set()

    # Add the main noun (lemmatized)
    if token.pos_ in ("NOUN", "PROPN"):
        topics.add(token.lemma_.lower())

        # Check for compound modifiers (e.g., "surplus value" where "surplus" is amod)
        modifiers = []
        for child in token.children:
            if child.dep_ in ("compound", "amod") and child.pos_ in ("NOUN", "ADJ"):
                modifiers.append(child.text.lower())

        if modifiers:
            full_term = " ".join([*modifiers, token.text.lower()])
            topics.add(full_term)

        # Follow prepositional phrases (e.g., "of the proletariat")
        for child in token.children:
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        topics.add(pobj.lemma_.lower())
                        # Build full phrase: "dictatorship of the proletariat"
                        full_phrase = f"{token.text.lower()} {child.text} {pobj.text.lower()}"
                        topics.add(full_phrase)
                        # Also get nested preps
                        topics.update(_extract_noun_with_preps(pobj))

    return topics


def _extract_question_topics(doc: Any) -> set[str]:
    """
    Extract the core topics from a question using spaCy dependency parsing.

    For "What is revisionism?", extracts {"revisionism"}
    For "How does imperialism relate to capitalism?", extracts {"imperialism", "capitalism"}
    For "What is the dictatorship of the proletariat?", extracts
        {"dictatorship", "proletariat", "dictatorship of the proletariat"}
    """
    topics: set[str] = set()

    # Find the ROOT
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break

    if root:
        # Extract from ROOT's children
        for child in root.children:
            # nsubj: "What is [revisionism]?" - revisionism is subject
            # dobj: "Explain [the concept]" - concept is direct object
            # attr: less common but possible
            # nsubjpass: passive subject
            if child.dep_ in ("nsubj", "dobj", "attr", "nsubjpass"):
                # Skip question words ("What is X" - skip "What")
                if child.text.lower() in QUESTION_WORDS:
                    continue
                topics.update(_extract_noun_with_preps(child))

            # pobj in prep attached to ROOT: "relate to [capitalism]"
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        topics.update(_extract_noun_with_preps(pobj))

    # Fallback: extract all noun chunks except question words
    if not topics:
        for chunk in doc.noun_chunks:
            root_text = chunk.root.text.lower()
            if root_text not in QUESTION_WORDS:
                topics.add(chunk.root.lemma_.lower())
                # Also add full chunk for multi-word terms
                chunk_text = chunk.text.lower().strip()
                if " " in chunk_text:
                    topics.add(chunk_text)

    # Final cleanup: remove question words that might have slipped through
    topics = {t for t in topics if t not in QUESTION_WORDS}

    return topics


def _extract_answer_topics(doc: Any) -> set[str]:
    """
    Extract topics discussed in an answer using spaCy.

    Returns lemmatized noun phrases and named entities.
    Strips determiners (the, a, an) for better matching.
    """
    topics: set[str] = set()

    # Determiners to strip from multi-word phrases
    determiners = {"the", "a", "an", "this", "that", "these", "those"}

    # Get noun chunk roots (lemmatized)
    for chunk in doc.noun_chunks:
        topics.add(chunk.root.lemma_.lower())

        # Multi-word terms (strip leading determiners)
        words = chunk.text.lower().strip().split()
        if words and words[0] in determiners:
            words = words[1:]
        chunk_text = " ".join(words)

        if " " in chunk_text and len(chunk_text) < 50:
            topics.add(chunk_text)

    # Get named entities
    for ent in doc.ents:
        ent_text = ent.text.lower()
        # Strip leading determiners from entities too
        words = ent_text.split()
        if words and words[0] in determiners:
            words = words[1:]
        topics.add(" ".join(words))

    return topics


def _expand_with_synonyms(topics: set[str]) -> set[str]:
    """
    Expand a set of topics with Marxist concept synonyms.

    If "bourgeoisie" is in topics, also adds "capitalist class", "ruling class", etc.
    """
    expanded = set(topics)

    for topic in topics:
        # Check if topic matches any canonical term
        if topic in CONCEPT_EQUIVALENCES:
            expanded.update(CONCEPT_EQUIVALENCES[topic])
        # Check if topic matches any synonym (reverse lookup)
        for canonical, synonyms in CONCEPT_EQUIVALENCES.items():
            if topic in synonyms or topic == canonical:
                expanded.add(canonical)
                expanded.update(synonyms)

    return expanded


def _compute_topic_coverage(q_topics: set[str], a_topics: set[str], nlp: Any) -> float:
    """
    Compute how well answer topics cover question topics.

    Uses:
    1. Direct lemma matching
    2. Expanded synonym matching
    3. spaCy word vector similarity (fallback)

    Returns coverage score 0.0 to 1.0
    """
    if not q_topics:
        return 0.5  # Can't evaluate, neutral

    # Expand question topics with synonyms
    q_expanded = _expand_with_synonyms(q_topics)

    # Direct/synonym match
    matched = q_expanded & a_topics
    direct_coverage = len(matched) / len(q_topics) if q_topics else 0

    if direct_coverage >= 0.5:
        return min(direct_coverage, 1.0)

    # Fallback: semantic similarity using spaCy vectors
    # For unmatched q_topics, check if any a_topic is semantically similar
    unmatched_q = q_topics - matched
    semantic_matches = 0

    for q_topic in unmatched_q:
        q_token = nlp(q_topic)
        if not q_token.has_vector:
            continue

        best_sim = 0.0
        for a_topic in a_topics:
            a_token = nlp(a_topic)
            if a_token.has_vector:
                sim = q_token.similarity(a_token)
                best_sim = max(best_sim, sim)

        if best_sim > 0.6:  # Threshold for semantic match
            semantic_matches += 1

    total_matched = len(matched) + semantic_matches
    return min(total_matched / len(q_topics), 1.0) if q_topics else 0.5


def topic_relevance_reward(
    prompts: Sequence[Sequence[dict[str, str]]],
    completions: Sequence[Sequence[dict[str, str]]],
    **kwargs: object,
) -> list[float]:
    """
    Reward answers that are ON-TOPIC with respect to the question.

    Implements f(A) ⊆ f(Q) check where f extracts semantic topics:
    1. Extract core topics from question Q using dependency parsing
    2. Expand Q topics with Marxist concept synonyms
    3. Extract topics from answer A
    4. Compute coverage: how many Q topics are addressed in A

    Scoring:
        > 80% coverage: +2.0 (answer fully addresses question topics)
        > 60% coverage: +1.5 (answer mostly on-topic)
        > 40% coverage: +1.0 (answer partially on-topic)
        > 20% coverage: 0.0 (answer tangentially related)
        <= 20% coverage: -1.5 (answer off-topic)

    This reward ensures the model answers WHAT WAS ASKED, not just
    generates coherent Marxist text about something else.
    """
    nlp = get_spacy_nlp()
    scores: list[float] = []

    for prompt, completion in zip(prompts, completions, strict=False):
        # Extract question (last user message)
        question = prompt[-1]["content"]
        response = completion[0]["content"]

        # Extract answer part after </think>
        if REASONING_END in response:
            response = response.split(REASONING_END, 1)[1].strip()

        # Handle empty response
        if not response or len(response.strip()) < 20:
            scores.append(-1.5)
            continue

        # Parse with spaCy
        q_doc = nlp(question)
        a_doc = nlp(response[:2000])  # Limit for performance

        # Extract topics
        q_topics = _extract_question_topics(q_doc)
        a_topics = _extract_answer_topics(a_doc)

        # Handle case where no topics extracted from question
        if not q_topics:
            # Fallback: just check if answer has substance
            scores.append(0.5 if len(a_topics) > 3 else 0.0)
            continue

        # Compute coverage
        coverage = _compute_topic_coverage(q_topics, a_topics, nlp)

        # Convert to reward score
        if coverage > 0.8:
            score = 2.0
        elif coverage > 0.6:
            score = 1.5
        elif coverage > 0.4:
            score = 1.0
        elif coverage > 0.2:
            score = 0.0
        else:
            score = -1.5

        scores.append(score)

    return scores


def full_coherence_reward(
    prompts: Sequence[Sequence[dict[str, str]]],
    completions: Sequence[Sequence[dict[str, str]]],
    answer: Sequence[str],
    **kwargs: object,
) -> list[float]:
    """
    Complete coherence check: robust_coherence + topic_relevance + depth.

    This is the MOST COMPREHENSIVE reward function, checking:
    1. NLI coherence (A entails ground truth)
    2. Self-consistency (A doesn't contradict itself)
    3. Structural coherence (terms in proper syntactic roles)
    4. Topic relevance (A addresses what Q asked about)
    5. Interconnection depth (rewards deep analysis, penalizes buzzword salad)

    Use this for maximum robustness against reward hacking.
    """
    robust_scores = robust_coherence_reward(completions, answer, **kwargs)
    relevance_scores = topic_relevance_reward(prompts, completions, **kwargs)
    depth_scores = interconnection_depth_reward(completions, **kwargs)

    combined: list[float] = []
    for robust, relevance, depth in zip(
        robust_scores, relevance_scores, depth_scores, strict=False
    ):
        # If severely off-topic, penalize
        if relevance <= -1.5:
            combined.append(-2.0)
        # If robust check failed badly, use that
        elif robust <= -2.0:
            combined.append(robust)
        # If buzzword salad detected (low depth), penalize
        elif depth <= -1.5:
            combined.append(-1.5)
        # Otherwise combine
        else:
            # Robust is primary, relevance and depth are bonuses/penalties
            total = robust + (relevance * 0.4) + (depth * 0.3)
            combined.append(total)

    return combined


# =============================================================================
# INTERCONNECTION DEPTH REWARD (Anti-Buzzword-Salad)
# =============================================================================


def _count_unique_marxist_concepts(text: str) -> int:
    """Count unique Marxist concepts mentioned in text."""
    text_lower = text.lower()
    count = 0
    for term in MARXIST_TERMS:
        if term in text_lower:
            count += 1
    return count


def _compute_depth_ratio(text: str) -> float:
    """
    Compute depth ratio: words per unique Marxist concept.

    High ratio = deep analysis (few concepts, well explained)
    Low ratio = shallow/buzzword soup (many concepts, little explanation)

    Returns:
        Words per concept, or 100.0 if no Marxist concepts found
    """
    words = len(text.split())
    concepts = _count_unique_marxist_concepts(text)

    if concepts == 0:
        return 100.0  # No Marxist concepts = neutral (not shallow)

    return words / concepts


def _count_hollow_buzzwords(text: str) -> int:
    """Count hollow buzzwords in text."""
    text_lower = text.lower()
    count = 0
    for buzzword in HOLLOW_BUZZWORDS:
        if buzzword in text_lower:
            count += 1
    return count


def _count_depth_markers(text: str) -> int:
    """Count analytical depth markers in text."""
    text_lower = text.lower()
    count = 0
    for marker in DEPTH_MARKERS:
        if marker in text_lower:
            count += 1
    return count


def _count_explanatory_phrases(text: str) -> int:
    """Count explanatory phrases in text."""
    text_lower = text.lower()
    count = 0
    for phrase in EXPLANATORY_PHRASES:
        if phrase in text_lower:
            count += 1
    return count


def _concepts_have_explanations(text: str) -> tuple[int, int]:
    """
    Check if introduced concepts have nearby explanations.

    Returns:
        Tuple of (explained_count, unexplained_count)
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)

    # Get sentences
    sentences = [sent.text.lower() for sent in doc.sents]

    explained = 0
    unexplained = 0

    for i, sent in enumerate(sentences):
        # Check which Marxist concepts appear in this sentence
        concepts_in_sent = [t for t in MARXIST_TERMS if t in sent]

        for _concept in concepts_in_sent:
            # Check if explanatory phrase appears in same or adjacent sentence
            has_explanation = False

            # Check current sentence
            for phrase in EXPLANATORY_PHRASES:
                if phrase in sent:
                    has_explanation = True
                    break

            # Check next sentence if exists
            if not has_explanation and i + 1 < len(sentences):
                next_sent = sentences[i + 1]
                for phrase in EXPLANATORY_PHRASES:
                    if phrase in next_sent:
                        has_explanation = True
                        break

            if has_explanation:
                explained += 1
            else:
                unexplained += 1

    return explained, unexplained


def interconnection_depth_reward(
    completions: Sequence[Sequence[dict[str, str]]], **kwargs: object
) -> list[float]:
    """
    Reward deep, meaningful interconnections; penalize buzzword salad.

    This reward distinguishes between:
    - GOOD: "Surplus value relates to imperialism BECAUSE capital export..."
    - BAD: "Surplus value intersects with imperialism, colonialism, patriarchy..."

    Signals:
    1. Depth ratio: words per unique Marxist concept
       - High (>15): Deep analysis, concepts well-explained
       - Low (<5): Shallow buzzword soup (many concepts crammed together)
    2. Hollow buzzword density: activist jargon without substance
    3. Depth markers: citations, examples, historical specificity
    4. Explanation ratio: concepts with nearby explanatory phrases

    Scoring:
        Depth ratio > 20: +1.0 (deep analysis)
        Depth ratio 10-20: +0.5 (adequate depth)
        Depth ratio < 5: -1.5 (severe buzzword soup)
        Depth ratio 5-10: -0.5 (shallow)
        Hollow buzzwords > 2: -0.3 each additional
        Depth markers present: +0.3 each (max +1.5)
        Good explanation ratio: +0.5
        Low explanation ratio with many concepts: -0.5

    Total range: approximately -2.5 to +3.0
    """
    scores: list[float] = []

    for completion in completions:
        response = completion[0]["content"]

        # Extract answer part after </think>
        if REASONING_END in response:
            answer_part = response.split(REASONING_END, 1)[1].strip()
        else:
            answer_part = response

        # Skip very short responses (handled by completeness_reward)
        word_count = len(answer_part.split())
        if word_count < 20:
            scores.append(0.0)
            continue

        score = 0.0
        concept_count = _count_unique_marxist_concepts(answer_part)

        # Signal 1: Depth ratio (words per concept)
        # Only penalize if there are concepts to evaluate
        if concept_count > 0:
            depth_ratio = word_count / concept_count
            if depth_ratio > 20:
                score += 1.0  # Deep analysis
            elif depth_ratio > 10:
                score += 0.5  # Adequate depth
            elif depth_ratio < 5:
                score -= 1.5  # Severe buzzword soup (many concepts, few words)
            elif depth_ratio < 10:
                score -= 0.5  # Shallow

        # Signal 2: Hollow buzzword penalty
        hollow_count = _count_hollow_buzzwords(answer_part)
        if hollow_count > 2:
            # Penalize excess hollow buzzwords
            penalty = 0.3 * (hollow_count - 2)
            score -= min(penalty, 1.5)  # Cap penalty at -1.5

        # Signal 3: Depth markers bonus
        depth_marker_count = _count_depth_markers(answer_part)
        score += min(depth_marker_count * 0.3, 1.5)

        # Signal 4: Explanation ratio
        explanatory_count = _count_explanatory_phrases(answer_part)

        if concept_count > 0:
            explanation_ratio = explanatory_count / concept_count
            if explanation_ratio >= 0.5:
                score += 0.5  # Good: at least 1 explanation per 2 concepts
            elif explanation_ratio < 0.1 and concept_count > 5:
                score -= 0.5  # Bad: many concepts, almost no explanations

        # Clamp final score
        scores.append(max(min(score, 3.0), -2.5))

    return scores


# =============================================================================
# RESPONSE COMPLETENESS REWARD
# =============================================================================


def completeness_reward(
    completions: Sequence[Sequence[dict[str, str]]],
    answer: Sequence[str],
    **kwargs: object,
) -> list[float]:
    """
    Reward thorough, detailed responses.

    Compares response length to ground truth length.

    Scoring:
        50-150% of target length: +2.0
        30-200% of target length: +1.0
        < 20% (too short): -2.0
        > 200% (too verbose): -0.5
    """
    scores: list[float] = []

    for completion, true_answer in zip(completions, answer, strict=False):
        response = completion[0]["content"]

        # Extract answer after </think>
        if REASONING_END in response:
            answer_part = response.split(REASONING_END, 1)[1].strip()
        else:
            answer_part = response

        answer_len = len(answer_part.split())
        true_len = len(true_answer.split())

        # Avoid division by zero
        if true_len == 0:
            scores.append(0.0)
            continue

        # Reward responses that are 50-150% of target length
        ratio = answer_len / true_len

        if 0.5 <= ratio <= 1.5:
            score = 2.0
        elif 0.3 <= ratio <= 2.0:
            score = 1.0
        elif ratio < 0.2:  # Too short
            score = -2.0
        else:  # Too long (verbose)
            score = -0.5

        scores.append(score)

    return scores


# =============================================================================
# DEBUG REWARD (for monitoring during training)
# =============================================================================

# Global counter for printing samples
_PRINT_COUNTER = 0
_PRINT_EVERY = 10


def debug_print_reward(
    prompts: Sequence[Sequence[dict[str, str]]],
    completions: Sequence[Sequence[dict[str, str]]],
    answer: Sequence[str],
    **kwargs: object,
) -> list[float]:
    """
    Print sample outputs periodically for monitoring.

    Returns 0.0 (no effect on training).
    """
    global _PRINT_COUNTER

    if _PRINT_COUNTER % _PRINT_EVERY == 0:
        question = prompts[0][-1]["content"]
        response = completions[0][0]["content"]
        true_answer = answer[0]

        print("=" * 60)
        print(f"Step {_PRINT_COUNTER}")
        print(f"Question: {question[:100]}...")
        print(f"Response: {response[:200]}...")
        print(f"Expected: {true_answer[:100]}...")
        print("=" * 60)

    _PRINT_COUNTER += 1

    return [0.0] * len(completions)
