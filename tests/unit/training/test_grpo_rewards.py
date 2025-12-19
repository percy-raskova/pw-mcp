"""
TDD Tests for GRPO Reward Functions.

Tests behavioral expectations for reward functions that guide
Marxist-Leninist Q&A model training.

Test Categories:
1. Format Rewards - Proper <think>...</think> tag usage
2. Terminology Reward - Marxist vocabulary (shallow, can be gamed)
3. Topic Extraction - Question/answer topic identification
4. Topic Relevance - Answer addresses question topics
5. Structural Coherence - Terms in proper syntactic roles (defeats word soup)
6. NLI Coherence - Logical consistency with ground truth (integration tests)
7. Combined Rewards - Multi-layer checking
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_completion() -> list[list[dict[str, str]]]:
    """Create a mock completion in GRPO format."""

    def _make(content: str) -> list[list[dict[str, str]]]:
        return [[{"role": "assistant", "content": content}]]

    return _make  # type: ignore[return-value]


@pytest.fixture
def mock_prompt() -> list[list[dict[str, str]]]:
    """Create a mock prompt in GRPO format."""

    def _make(question: str) -> list[list[dict[str, str]]]:
        return [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": question},
            ]
        ]

    return _make  # type: ignore[return-value]


# =============================================================================
# FORMAT REWARDS TESTS
# =============================================================================


class TestMatchFormatExactly:
    """Tests for match_format_exactly reward function."""

    def test_rewards_proper_think_tags(self, mock_completion: object) -> None:
        """Response with </think> tag should get +3.0."""
        from pw_mcp.ai_training.grpo_rewards import match_format_exactly

        completions = mock_completion(  # type: ignore[operator]
            "<think>Let me analyze this...</think>The bourgeoisie exploits workers."
        )
        scores = match_format_exactly(completions)

        assert scores == [3.0]

    def test_penalizes_missing_think_tags(self, mock_completion: object) -> None:
        """Response without </think> tag should get 0.0."""
        from pw_mcp.ai_training.grpo_rewards import match_format_exactly

        completions = mock_completion("The bourgeoisie exploits workers.")  # type: ignore[operator]
        scores = match_format_exactly(completions)

        assert scores == [0.0]

    def test_only_needs_end_tag(self, mock_completion: object) -> None:
        """Only the </think> tag is checked, not <think>."""
        from pw_mcp.ai_training.grpo_rewards import match_format_exactly

        completions = mock_completion("Some text</think>Answer here")  # type: ignore[operator]
        scores = match_format_exactly(completions)

        assert scores == [3.0]


class TestMatchFormatApproximately:
    """Tests for match_format_approximately reward function."""

    def test_rewards_proper_single_tags(self, mock_completion: object) -> None:
        """Exactly one of each tag should get +1.0."""
        from pw_mcp.ai_training.grpo_rewards import match_format_approximately

        completions = mock_completion(  # type: ignore[operator]
            "<think>Reasoning...</think>Answer"
        )
        scores = match_format_approximately(completions)

        assert scores == [1.0]

    def test_penalizes_multiple_start_tags(self, mock_completion: object) -> None:
        """Multiple <think> tags should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import match_format_approximately

        completions = mock_completion(  # type: ignore[operator]
            "<think>First</think><think>Second</think>Answer"
        )
        scores = match_format_approximately(completions)

        # Two <think> tags: -1.0 for start, +0.5 for two </think> (wait, that's also 2)
        # Actually: 2 starts = -1.0, 2 ends = -1.0, total = -2.0
        assert scores == [-2.0]

    def test_penalizes_missing_tags(self, mock_completion: object) -> None:
        """Missing tags should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import match_format_approximately

        completions = mock_completion("Plain text without tags")  # type: ignore[operator]
        scores = match_format_approximately(completions)

        # 0 starts: -1.0, 0 ends: -1.0, total = -2.0
        assert scores == [-2.0]


# =============================================================================
# TERMINOLOGY REWARD TESTS
# =============================================================================


class TestTerminologyReward:
    """Tests for terminology_reward function (shallow reward)."""

    def test_rewards_marxist_terms(self, mock_completion: object) -> None:
        """Response with Marxist terms should get positive reward."""
        from pw_mcp.ai_training.grpo_rewards import terminology_reward

        completions = mock_completion(  # type: ignore[operator]
            "The bourgeoisie extracts surplus value from the proletariat."
        )
        scores = terminology_reward(completions)

        # Should find: bourgeoisie, surplus value, proletariat = 3 terms = 0.9
        assert scores[0] > 0.0
        assert scores[0] <= 2.0  # Capped at 2.0

    def test_no_reward_without_terms(self, mock_completion: object) -> None:
        """Response without Marxist terms should get 0.0."""
        from pw_mcp.ai_training.grpo_rewards import terminology_reward

        completions = mock_completion("The sky is blue and grass is green.")  # type: ignore[operator]
        scores = terminology_reward(completions)

        assert scores == [0.0]

    def test_word_soup_gets_reward_showing_vulnerability(self, mock_completion: object) -> None:
        """
        DEMONSTRATION: Word soup DOES get rewarded by this shallow function.
        This is why we need NLI-based rewards.
        """
        from pw_mcp.ai_training.grpo_rewards import terminology_reward

        # Word soup - random Marxist terms with no coherent meaning
        word_soup = (
            "bourgeoisie proletariat dialectical materialism surplus value "
            "imperialism revisionism hegemony alienation"
        )
        completions = mock_completion(word_soup)  # type: ignore[operator]
        scores = terminology_reward(completions)

        # Word soup gets high reward - this is the PROBLEM we're solving
        assert scores[0] >= 2.0  # Maximum reward for garbage!


# =============================================================================
# TOPIC EXTRACTION TESTS
# =============================================================================


class TestTopicExtraction:
    """Tests for topic extraction helper functions."""

    @pytest.fixture(autouse=True)
    def load_spacy(self) -> None:
        """Load spaCy model once for all tests."""
        import spacy

        self.nlp = spacy.load("en_core_web_trf")

    def test_extracts_simple_topic(self) -> None:
        """'What is revisionism?' should extract 'revisionism'."""
        from pw_mcp.ai_training.grpo_rewards import _extract_question_topics

        doc = self.nlp("What is revisionism?")
        topics = _extract_question_topics(doc)

        assert "revisionism" in topics

    def test_extracts_multiple_topics(self) -> None:
        """'How does imperialism relate to capitalism?' should extract both."""
        from pw_mcp.ai_training.grpo_rewards import _extract_question_topics

        doc = self.nlp("How does imperialism relate to capitalism?")
        topics = _extract_question_topics(doc)

        assert "imperialism" in topics
        assert "capitalism" in topics

    def test_extracts_compound_topic(self) -> None:
        """'Explain surplus value' should extract 'surplus value'."""
        from pw_mcp.ai_training.grpo_rewards import _extract_question_topics

        doc = self.nlp("Explain the concept of surplus value.")
        topics = _extract_question_topics(doc)

        # Should find either the compound or individual parts
        assert "surplus value" in topics or ("surplus" in topics and "value" in topics)

    def test_extracts_prepositional_phrase(self) -> None:
        """'dictatorship of the proletariat' should extract full phrase."""
        from pw_mcp.ai_training.grpo_rewards import _extract_question_topics

        doc = self.nlp("What is the dictatorship of the proletariat?")
        topics = _extract_question_topics(doc)

        assert "dictatorship" in topics
        assert "proletariat" in topics

    def test_excludes_question_words(self) -> None:
        """Question words (what, how, why) should not be extracted."""
        from pw_mcp.ai_training.grpo_rewards import _extract_question_topics

        doc = self.nlp("What is dialectical materialism?")
        topics = _extract_question_topics(doc)

        assert "what" not in topics

    def test_answer_topic_extraction(self) -> None:
        """Answer extraction should find noun phrases and entities."""
        from pw_mcp.ai_training.grpo_rewards import _extract_answer_topics

        doc = self.nlp(
            "The bourgeoisie controls the means of production. Workers sell their labor power."
        )
        topics = _extract_answer_topics(doc)

        # Should find key concepts
        assert len(topics) > 0
        # Check for some expected terms
        has_class_term = any(t in topics for t in ["bourgeoisie", "worker", "workers"])
        assert has_class_term


# =============================================================================
# TOPIC RELEVANCE REWARD TESTS
# =============================================================================


class TestTopicRelevanceReward:
    """Tests for topic_relevance_reward function."""

    def test_on_topic_answer_rewarded(self, mock_prompt: object, mock_completion: object) -> None:
        """Answer that addresses the question topic should get positive reward."""
        from pw_mcp.ai_training.grpo_rewards import topic_relevance_reward

        prompts = mock_prompt("What is revisionism?")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "</think>Revisionism is the distortion of Marxist theory, "
            "abandoning revolutionary principles in favor of reformism."
        )

        scores = topic_relevance_reward(prompts, completions)

        assert scores[0] > 0.0, "On-topic answer should be rewarded"

    def test_off_topic_answer_penalized(self, mock_prompt: object, mock_completion: object) -> None:
        """Answer about unrelated topic should get negative reward."""
        from pw_mcp.ai_training.grpo_rewards import topic_relevance_reward

        prompts = mock_prompt("What is revisionism?")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "</think>The weather today is sunny with clear skies. I recommend wearing sunscreen."
        )

        scores = topic_relevance_reward(prompts, completions)

        assert scores[0] <= 0.0, "Off-topic answer should be penalized"

    def test_partial_topic_coverage(self, mock_prompt: object, mock_completion: object) -> None:
        """Answer covering some but not all topics gets partial reward."""
        from pw_mcp.ai_training.grpo_rewards import topic_relevance_reward

        prompts = mock_prompt(  # type: ignore[operator]
            "How does imperialism relate to capitalism?"
        )
        # Only discusses imperialism, not capitalism
        completions = mock_completion(  # type: ignore[operator]
            "</think>Imperialism is the highest stage of development "
            "characterized by monopolies and export of capital."
        )

        scores = topic_relevance_reward(prompts, completions)

        # Should get some reward but not maximum
        assert scores[0] >= -1.5, "Partial coverage shouldn't be heavily penalized"

    def test_synonym_recognition(self, mock_prompt: object, mock_completion: object) -> None:
        """Answer using synonyms should still be recognized as on-topic."""
        from pw_mcp.ai_training.grpo_rewards import topic_relevance_reward

        prompts = mock_prompt("What is the bourgeoisie?")  # type: ignore[operator]
        # Uses synonym "capitalist class" instead of "bourgeoisie"
        completions = mock_completion(  # type: ignore[operator]
            "</think>The capitalist class owns the means of production "
            "and exploits the working class for profit."
        )

        scores = topic_relevance_reward(prompts, completions)

        # Synonym should be recognized
        assert scores[0] >= 0.0, "Synonyms should be recognized as on-topic"


# =============================================================================
# STRUCTURAL COHERENCE REWARD TESTS
# =============================================================================


class TestStructuralCoherenceReward:
    """Tests for structural_coherence_reward function."""

    def test_coherent_sentences_rewarded(self, mock_completion: object) -> None:
        """Proper sentences with terms in syntactic roles should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import structural_coherence_reward

        completions = mock_completion(  # type: ignore[operator]
            "The bourgeoisie extracts surplus value from workers. "
            "Therefore, class struggle is inevitable."
        )
        scores = structural_coherence_reward(completions)

        assert scores[0] > 0.0, "Coherent text should get positive reward"

    def test_word_soup_penalized(self, mock_completion: object) -> None:
        """Word soup should get low/negative structural score."""
        from pw_mcp.ai_training.grpo_rewards import structural_coherence_reward

        # Random terms not in syntactic roles
        word_soup = (
            "bourgeoisie proletariat dialectical materialism surplus value "
            "imperialism revisionism hegemony alienation"
        )
        completions = mock_completion(word_soup)  # type: ignore[operator]
        scores = structural_coherence_reward(completions)

        # Word soup has no proper sentence structure
        # Should get very low score (terms not in subject/object positions)
        assert scores[0] < 1.0, "Word soup should get low structural score"

    def test_discourse_connectives_rewarded(self, mock_completion: object) -> None:
        """Logical connectives (therefore, because) should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import structural_coherence_reward

        completions = mock_completion(  # type: ignore[operator]
            "The proletariat is exploited because they do not own capital. "
            "Therefore, revolution is necessary. "
            "Furthermore, the state must be seized."
        )
        scores = structural_coherence_reward(completions)

        # Multiple discourse connectives should boost score
        assert scores[0] > 0.5, "Discourse connectives should be rewarded"


# =============================================================================
# COMPLETENESS REWARD TESTS
# =============================================================================


class TestCompletenessReward:
    """Tests for completeness_reward function."""

    def test_appropriate_length_rewarded(self, mock_completion: object) -> None:
        """Response similar in length to ground truth should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import completeness_reward

        ground_truth = "The bourgeoisie is the capitalist class that owns production."
        completions = mock_completion(  # type: ignore[operator]
            "</think>The bourgeoisie refers to the capitalist class "
            "that controls the means of production."
        )

        scores = completeness_reward(completions, answer=[ground_truth])

        assert scores[0] > 0.0, "Similar length should be rewarded"

    def test_too_short_penalized(self, mock_completion: object) -> None:
        """Very short response should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import completeness_reward

        ground_truth = (
            "The bourgeoisie is the capitalist class that owns the means "
            "of production and exploits the working class through extraction "
            "of surplus value from their labor."
        )
        completions = mock_completion("</think>The bourgeoisie owns capital.")  # type: ignore[operator]

        scores = completeness_reward(completions, answer=[ground_truth])

        assert scores[0] < 0.0, "Too short should be penalized"


# =============================================================================
# NLI REWARD TESTS (Integration - requires model loading)
# =============================================================================


@pytest.mark.slow
class TestNLICoherenceReward:
    """Tests for NLI-based coherence reward (requires bart-large-mnli)."""

    def test_entailment_rewarded(self, mock_completion: object) -> None:
        """Response entailing ground truth should get positive reward."""
        from pw_mcp.ai_training.grpo_rewards import nli_coherence_reward

        ground_truth = "The bourgeoisie extracts surplus value from workers."
        completions = mock_completion(  # type: ignore[operator]
            "</think>The capitalist class exploits workers by extracting "
            "the value of their unpaid labor."
        )

        scores = nli_coherence_reward(completions, answer=[ground_truth])

        assert scores[0] > 0.0, "Entailment should be rewarded"

    def test_contradiction_penalized(self, mock_completion: object) -> None:
        """Response contradicting ground truth should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import nli_coherence_reward

        ground_truth = "The bourgeoisie exploits workers."
        completions = mock_completion(  # type: ignore[operator]
            "</think>Capitalism benefits all classes equally. "
            "Workers are not exploited under capitalism."
        )

        scores = nli_coherence_reward(completions, answer=[ground_truth])

        assert scores[0] < 0.0, "Contradiction should be penalized"

    def test_word_soup_neutral(self, mock_completion: object) -> None:
        """Word soup should be classified as neutral (off-topic)."""
        from pw_mcp.ai_training.grpo_rewards import nli_coherence_reward

        ground_truth = "Revisionism abandons revolutionary principles."
        word_soup = (
            "bourgeoisie proletariat dialectical materialism surplus value "
            "imperialism revisionism hegemony alienation"
        )
        completions = mock_completion(word_soup)  # type: ignore[operator]

        scores = nli_coherence_reward(completions, answer=[ground_truth])

        # Word soup should be neutral (off-topic), not entailment
        assert scores[0] <= 0.0, "Word soup should not get positive NLI score"


@pytest.mark.slow
class TestSelfConsistencyReward:
    """Tests for self-consistency reward (requires bart-large-mnli)."""

    def test_consistent_response_rewarded(self, mock_completion: object) -> None:
        """Response without internal contradictions should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import self_consistency_reward

        completions = mock_completion(  # type: ignore[operator]
            "The bourgeoisie owns capital. They extract surplus value. "
            "This exploitation drives class struggle."
        )

        scores = self_consistency_reward(completions)

        assert scores[0] > 0.0, "Consistent response should be rewarded"

    def test_contradictory_response_penalized(self, mock_completion: object) -> None:
        """Response with internal contradictions should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import self_consistency_reward

        completions = mock_completion(  # type: ignore[operator]
            "Capitalism benefits everyone equally. "
            "Workers are severely exploited under capitalism. "
            "Nobody is harmed by the capitalist system."
        )

        scores = self_consistency_reward(completions)

        assert scores[0] < 0.0, "Contradictory response should be penalized"


# =============================================================================
# COMBINED REWARD TESTS
# =============================================================================


class TestRobustCoherenceReward:
    """Tests for robust_coherence_reward (combined multi-layer check)."""

    @pytest.mark.slow
    def test_good_response_high_score(self, mock_completion: object) -> None:
        """Well-formed, on-topic response should get high combined score."""
        from pw_mcp.ai_training.grpo_rewards import robust_coherence_reward

        ground_truth = (
            "Revisionism is the distortion of Marxist theory, abandoning revolutionary principles."
        )
        completions = mock_completion(  # type: ignore[operator]
            "</think>Revisionism represents a deviation from Marxist principles, "
            "characterized by the abandonment of revolutionary goals in favor of "
            "reformist approaches. Therefore, it represents a threat to the movement."
        )

        scores = robust_coherence_reward(completions, answer=[ground_truth])

        assert scores[0] > 0.0, "Good response should get positive combined score"

    @pytest.mark.slow
    def test_word_soup_low_score(self, mock_completion: object) -> None:
        """Word soup should get low combined score despite terminology."""
        from pw_mcp.ai_training.grpo_rewards import robust_coherence_reward

        ground_truth = "Revisionism abandons revolutionary principles."
        word_soup = (
            "bourgeoisie proletariat dialectical materialism surplus value "
            "imperialism revisionism hegemony alienation"
        )
        completions = mock_completion(word_soup)  # type: ignore[operator]

        scores = robust_coherence_reward(completions, answer=[ground_truth])

        # Word soup fails NLI (neutral) and structural coherence
        assert scores[0] <= 0.0, "Word soup should get low combined score"


class TestFullCoherenceReward:
    """Tests for full_coherence_reward (robust + topic relevance)."""

    @pytest.mark.slow
    def test_on_topic_coherent_high_score(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """On-topic, coherent response should get highest score."""
        from pw_mcp.ai_training.grpo_rewards import full_coherence_reward

        prompts = mock_prompt("What is revisionism?")  # type: ignore[operator]
        ground_truth = "Revisionism distorts Marxist theory."
        completions = mock_completion(  # type: ignore[operator]
            "</think>Revisionism is the distortion of Marxist theory, "
            "abandoning revolutionary principles for reformism."
        )

        scores = full_coherence_reward(prompts, completions, answer=[ground_truth])

        assert scores[0] > 0.0, "On-topic coherent response should score high"

    @pytest.mark.slow
    def test_off_topic_coherent_lower_score(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Off-topic but coherent response should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import full_coherence_reward

        prompts = mock_prompt("What is revisionism?")  # type: ignore[operator]
        ground_truth = "Revisionism distorts Marxist theory."
        # Coherent Marxist text but about wrong topic
        completions = mock_completion(  # type: ignore[operator]
            "</think>Imperialism is the highest stage of capitalism. "
            "It is characterized by monopolies and export of capital. "
            "Lenin analyzed this in his famous work."
        )

        scores = full_coherence_reward(prompts, completions, answer=[ground_truth])

        # Should be penalized for being off-topic
        assert scores[0] < 2.0, "Off-topic response should not get high score"


# =============================================================================
# INTERCONNECTION DEPTH REWARD TESTS
# =============================================================================


class TestInterconnectionDepthReward:
    """Test interconnection_depth_reward distinguishes depth from buzzword salad."""

    def test_deep_analysis_rewarded(self, mock_completion: object) -> None:
        """Deep analysis with few concepts well-explained should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import interconnection_depth_reward

        # Deep analysis: few concepts, well explained with causal reasoning
        deep_response = (
            "</think>Surplus value is the difference between the value produced by "
            "labor and the wages paid to workers. This occurs because the capitalist "
            "pays for labor power (the capacity to work) rather than labor itself. "
            "Marx argued that this extraction is the source of profit. For example, "
            "if a worker produces goods worth $100 but receives only $50 in wages, "
            "the remaining $50 constitutes surplus value. This is the fundamental "
            "mechanism of exploitation under capitalism."
        )
        completions = mock_completion(deep_response)  # type: ignore[operator]

        scores = interconnection_depth_reward(completions)

        assert scores[0] > 0.0, "Deep analysis should be rewarded"

    def test_buzzword_salad_penalized(self, mock_completion: object) -> None:
        """Buzzword salad (many concepts, no explanation) should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import interconnection_depth_reward

        # Buzzword salad: many concepts dropped without explanation
        buzzword_response = (
            "</think>Surplus value is interconnected with exploitation, "
            "alienation, commodity fetishism, imperialism, colonialism, "
            "hegemony, class struggle, dialectical materialism, and the "
            "dictatorship of the proletariat. It's all systemic and relates "
            "to the bourgeoisie and proletariat in problematic ways."
        )
        completions = mock_completion(buzzword_response)  # type: ignore[operator]

        scores = interconnection_depth_reward(completions)

        assert scores[0] < 0.0, "Buzzword salad should be penalized"

    def test_activist_jargon_penalized(self, mock_completion: object) -> None:
        """Activist jargon without substance should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import interconnection_depth_reward

        # Activist jargon: hollow phrases without analytical content
        jargon_response = (
            "</think>We need to center the lived experiences of the proletariat "
            "and unpack the systemic harm of capitalism. It's problematic how "
            "the bourgeoisie uplifts toxic narratives. We must do the work to "
            "unlearn harmful ideology and hold space for class consciousness. "
            "It's all interconnected in a way that requires us to lean into "
            "the dialectical process of liberation."
        )
        completions = mock_completion(jargon_response)  # type: ignore[operator]

        scores = interconnection_depth_reward(completions)

        assert scores[0] < 0.0, "Activist jargon should be penalized"

    def test_historical_specificity_rewarded(self, mock_completion: object) -> None:
        """Historical specificity and citations should boost score."""
        from pw_mcp.ai_training.grpo_rewards import interconnection_depth_reward

        # Response with historical specificity and citations
        specific_response = (
            "</think>The dictatorship of the proletariat, as Marx described it "
            "after the Paris Commune of 1871, refers to the political rule of "
            "the working class during the transition from capitalism to communism. "
            "Lenin further developed this concept, arguing that the state would "
            "eventually wither away as class distinctions disappeared. For example, "
            "the Soviet state under Lenin implemented workers' councils (soviets) "
            "as the basis of proletarian democracy."
        )
        completions = mock_completion(specific_response)  # type: ignore[operator]

        scores = interconnection_depth_reward(completions)

        # Should be positive (rewarded for historical specificity)
        assert scores[0] > 0.0, "Historical specificity should be rewarded"

    def test_explanation_ratio_matters(self, mock_completion: object) -> None:
        """High explanation ratio should improve score."""
        from pw_mcp.ai_training.grpo_rewards import interconnection_depth_reward

        # Response with good explanation ratio
        explained_response = (
            "</think>Alienation refers to the estrangement of workers from their "
            "labor. This occurs because workers do not own the means of production. "
            "As a result of this, they have no control over what they produce. "
            "Marx argued that this leads to alienation from the product, from the "
            "labor process, from fellow workers, and from human potential itself. "
            "Therefore, alienation is not merely psychological but structural."
        )
        completions = mock_completion(explained_response)  # type: ignore[operator]

        scores = interconnection_depth_reward(completions)

        assert scores[0] > 0.0, "Well-explained concepts should be rewarded"

    def test_deep_vs_shallow_distinction(self, mock_completion: object) -> None:
        """Deep analysis should score higher than shallow buzzword listing."""
        from pw_mcp.ai_training.grpo_rewards import interconnection_depth_reward

        deep_response = (
            "</think>Imperialism, as Lenin analyzed, represents the highest stage "
            "of capitalism. This is because monopoly capital seeks new markets "
            "and investment opportunities abroad. The export of capital, rather "
            "than goods, becomes the dominant form of economic expansion. Lenin "
            "argued that this leads to the division of the world among great powers. "
            "For example, the scramble for Africa in the 1880s exemplified this process."
        )

        shallow_response = (
            "</think>Imperialism relates to capitalism, colonialism, exploitation, "
            "surplus value, monopolies, and the bourgeoisie. It intersects with "
            "hegemony, class struggle, and national liberation. The proletariat "
            "faces alienation and commodity fetishism under imperialism."
        )

        deep_completions = mock_completion(deep_response)  # type: ignore[operator]
        shallow_completions = mock_completion(shallow_response)  # type: ignore[operator]

        deep_score = interconnection_depth_reward(deep_completions)[0]
        shallow_score = interconnection_depth_reward(shallow_completions)[0]

        assert deep_score > shallow_score, (
            f"Deep analysis ({deep_score}) should score higher than "
            f"shallow buzzword listing ({shallow_score})"
        )


class TestDepthHelpers:
    """Test helper functions for depth analysis."""

    def test_depth_ratio_calculation(self) -> None:
        """Test depth ratio calculation."""
        from pw_mcp.ai_training.grpo_rewards import _compute_depth_ratio

        # 100 words, 2 concepts = 50 words/concept
        text_deep = "word " * 98 + "bourgeoisie proletariat"
        ratio = _compute_depth_ratio(text_deep)
        assert ratio == 50.0, f"Expected 50.0, got {ratio}"

    def test_hollow_buzzword_counting(self) -> None:
        """Test hollow buzzword detection."""
        from pw_mcp.ai_training.grpo_rewards import _count_hollow_buzzwords

        # Text with multiple hollow buzzwords
        text = (
            "It's interconnected and systemic. We need to center "
            "lived experiences and do the work to unpack this."
        )
        count = _count_hollow_buzzwords(text)
        assert count >= 5, f"Expected at least 5 hollow buzzwords, got {count}"

    def test_depth_marker_counting(self) -> None:
        """Test depth marker detection."""
        from pw_mcp.ai_training.grpo_rewards import _count_depth_markers

        # Text with depth markers
        text = (
            "Marx argued that surplus value emerges during the production process. "
            "For example, in 1867 he published Capital. According to Lenin, "
            "imperialism specifically developed after 1880."
        )
        count = _count_depth_markers(text)
        assert count >= 3, f"Expected at least 3 depth markers, got {count}"

    def test_explanatory_phrase_counting(self) -> None:
        """Test explanatory phrase detection."""
        from pw_mcp.ai_training.grpo_rewards import _count_explanatory_phrases

        # Text with explanatory phrases
        text = (
            "This occurs because the capitalist extracts surplus value. "
            "As a result of this, workers are exploited. This means that "
            "profit specifically comes from unpaid labor."
        )
        count = _count_explanatory_phrases(text)
        assert count >= 3, f"Expected at least 3 explanatory phrases, got {count}"


# =============================================================================
# IDEOLOGICAL FIRMNESS REWARD TESTS
# =============================================================================


class TestIdeologicalFirmnessReward:
    """
    Tests for ideological_firmness_reward function.

    This reward function encourages the model to:
    1. Maintain principled Marxist positions when faced with reactionary claims
    2. Provide substantive counter-arguments rather than capitulating
    3. Use class analysis to refute ideological challenges
    """

    # -------------------------------------------------------------------------
    # HIGH SCORE TESTS (+1.5 to +2.0)
    # -------------------------------------------------------------------------

    def test_hitler_workers_firm_refutation(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Firm refutation of fascist propaganda should score highly."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Didn't Hitler help German workers?"]
        completions = [
            "This is fascist propaganda. Hitler destroyed independent trade unions, "
            "banned strikes, and handed German industry to monopoly capital. "
            "The Nazis served the interests of Krupp, IG Farben, and finance capital - "
            "not workers. Fascism is the open terrorist dictatorship of the most "
            "reactionary sections of the bourgeoisie."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert (
            scores[0] >= 1.5
        ), f"Firm refutation of fascist propaganda should score >= 1.5, got {scores[0]}"

    def test_100_million_claim_refutation(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Refutation of Black Book claims should score highly."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Communism killed 100 million people"]
        completions = [
            "This figure comes from the discredited Black Book of Communism, whose own "
            "authors disavowed it. More importantly, this framing ignores that capitalism "
            "kills millions annually through preventable poverty, lack of healthcare, and "
            "imperialist wars. The material improvements under socialist construction - "
            "literacy, healthcare, housing - are historically documented."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert (
            scores[0] >= 1.5
        ), f"Refutation of Black Book claims should score >= 1.5, got {scores[0]}"

    def test_stalin_dictator_analysis(self, mock_prompt: object, mock_completion: object) -> None:
        """Class analysis of 'dictator' framing should score highly."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Stalin was a dictator"]
        completions = [
            "The 'dictator' framing serves bourgeois historiography. Stalin led a party "
            "structure with collective decision-making. The Soviet state represented the "
            "dictatorship of the proletariat - class rule by workers over the bourgeoisie. "
            "This is qualitatively different from bourgeois dictatorship, which exists "
            "regardless of electoral forms."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert (
            scores[0] >= 1.5
        ), f"Class analysis of 'dictator' framing should score >= 1.5, got {scores[0]}"

    # -------------------------------------------------------------------------
    # LOW SCORE TESTS (-1.5 to -2.0)
    # -------------------------------------------------------------------------

    def test_hitler_workers_capitulation(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Capitulating to fascist framing should score very low."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Didn't Hitler help German workers?"]
        completions = [
            "While Hitler did implement some programs that benefited German workers in "
            "the short term, like infrastructure projects and reduced unemployment, it's "
            "important to consider the broader context and weigh both the positive and "
            "negative aspects of his economic policies."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert (
            scores[0] <= -1.5
        ), f"Capitulating to fascist framing should score <= -1.5, got {scores[0]}"

    def test_stalin_both_sidesing(self, mock_prompt: object, mock_completion: object) -> None:
        """Both-sidesing on Stalin should score very low."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Was Stalin bad?"]
        completions = [
            "There are valid points on both sides of this debate. Stalin did industrialize "
            "the USSR rapidly, but he also oversaw purges. It's a complex historical "
            "question with no easy answers. Reasonable people can disagree about his legacy."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert scores[0] <= -1.5, f"Both-sidesing on Stalin should score <= -1.5, got {scores[0]}"

    # -------------------------------------------------------------------------
    # NEUTRAL TESTS (~0.0)
    # -------------------------------------------------------------------------

    def test_factual_question_neutral(self, mock_prompt: object, mock_completion: object) -> None:
        """Factual question without ideological challenge should be neutral."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["What year did the October Revolution happen?"]
        completions = ["The October Revolution occurred in 1917."]

        scores = ideological_firmness_reward(prompts, completions)

        assert -0.5 <= scores[0] <= 0.5, f"Factual question should score ~0.0, got {scores[0]}"

    def test_educational_question_neutral(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Educational question about Marxist concepts should be neutral."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Explain dialectical materialism"]
        completions = [
            "Dialectical materialism is the philosophical framework developed by Marx "
            "and Engels that applies dialectical logic to materialist understanding of "
            "historical development and social change."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert -0.5 <= scores[0] <= 0.5, f"Educational question should score ~0.0, got {scores[0]}"

    def test_biographical_question_neutral(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Biographical question should be neutral."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Who was Rosa Luxemburg?"]
        completions = [
            "Rosa Luxemburg was a revolutionary Marxist theorist and leader of the "
            "Spartacist League in Germany. She was assassinated in 1919 by Freikorps."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert -0.5 <= scores[0] <= 0.5, f"Biographical question should score ~0.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # EDGE CASE TESTS
    # -------------------------------------------------------------------------

    def test_quote_to_refute_pattern(self, mock_prompt: object, mock_completion: object) -> None:
        """Quoting reactionary claim to refute it should NOT be penalized."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Didn't communism fail?"]
        completions = [
            "Bourgeois historians claim that 'communism failed,' but this ignores the "
            "material reality: the USSR went from feudal backwater to industrial "
            "superpower in 30 years, defeated fascism, achieved universal literacy, "
            "and provided housing and healthcare to all citizens."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert scores[0] >= 1.0, f"Quote-to-refute pattern should score >= 1.0, got {scores[0]}"

    def test_principled_self_criticism(self, mock_prompt: object, mock_completion: object) -> None:
        """Principled ML self-criticism should NOT be penalized as capitulation."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Did the USSR make mistakes?"]
        completions = [
            "Yes, Marxist-Leninists engage in self-criticism as a methodological principle. "
            "The Soviet Union made errors in areas like nationality policy and the handling "
            "of the Sino-Soviet split. However, these criticisms come from a revolutionary "
            "perspective aimed at strengthening socialism, not from bourgeois anti-communism."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert scores[0] >= 0.0, f"Principled self-criticism should score >= 0.0, got {scores[0]}"

    def test_mixed_hedging(self, mock_prompt: object, mock_completion: object) -> None:
        """Mixed hedging language should score low."""
        from pw_mcp.ai_training.grpo_rewards import ideological_firmness_reward

        prompts = ["Was Mao good or bad?"]
        completions = [
            "I can see why you'd ask that question. Mao's legacy is certainly complex, "
            "with both achievements and mistakes to consider. On one hand, he liberated "
            "China from imperialism and feudalism. On the other hand, there were excesses "
            "during various campaigns."
        ]

        scores = ideological_firmness_reward(prompts, completions)

        assert scores[0] <= -1.0, f"Mixed hedging should score <= -1.0, got {scores[0]}"


# =============================================================================
# ENTITY VERIFICATION REWARD TESTS
# =============================================================================


class TestEntityVerificationReward:
    """
    Tests for entity_verification_reward function.

    This reward function penalizes confident claims about entities NOT in
    the verified whitelist (24,040 entities from ProleWiki).

    Scoring logic:
    - +2.0: Expresses uncertainty about unknown entities
    - +1.0: Discusses only verified entities
    - -1.0: Discusses unknown entities without clear uncertainty
    - -2.5: Makes confident claims about unknown entities
    """

    def test_verified_entities_positive_score(self, mock_completion: object) -> None:
        """Response mentioning only verified entities (Karl Marx, Lenin) gets positive score."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about Marxist theory"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "Karl Marx and Lenin developed the theory of historical materialism."
        )

        scores = entity_verification_reward(prompts, completions, answer=[""])

        # Karl Marx and Lenin should be in whitelist; score should be positive or neutral
        # If no uncertainty and all verified -> +1.0
        assert scores[0] >= 0.0, f"Verified entities should get >= 0.0, got {scores[0]}"

    def test_unverified_entity_with_uncertainty_positive(self, mock_completion: object) -> None:
        """Expressing uncertainty about unverified entity gets positive score."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about the Militant League"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "I cannot verify information about the Militant League. "
            "I don't have reliable data about this organization."
        )

        scores = entity_verification_reward(prompts, completions, answer=[""])

        # Uncertainty patterns + unknown entity -> +2.0
        assert (
            scores[0] > 0.0
        ), f"Uncertainty about unknown entity should be positive, got {scores[0]}"

    def test_unverified_entity_confident_claim_negative(self, mock_completion: object) -> None:
        """Confident claims about unverified entity get negative score."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about the Militant League"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "The Militant League was founded in 1923 and played a significant "
            "role in revolutionary history."
        )

        scores = entity_verification_reward(prompts, completions, answer=[""])

        # Confident claim pattern about unknown entity -> -2.5
        assert (
            scores[0] < 0.0
        ), f"Confident claim about unknown entity should be negative, got {scores[0]}"

    def test_unverified_entity_fabricated_details_very_negative(
        self, mock_completion: object
    ) -> None:
        """Fabricating details about unverified entity gets very negative score."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about the Militant League"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "The Militant League was founded in 1923 by Zhang Wei in Shanghai. "
            "They organized 50,000 workers and led the uprising of 1925."
        )

        scores = entity_verification_reward(prompts, completions, answer=[""])

        # Multiple confident claims about unknown entity -> -2.5
        assert scores[0] < -1.0, f"Fabricated details should be heavily penalized, got {scores[0]}"

    def test_empty_completion_neutral(self, mock_completion: object) -> None:
        """Empty completion gets neutral score."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about something"},
            ]
        ]
        completions = mock_completion("")  # type: ignore[operator]

        scores = entity_verification_reward(prompts, completions, answer=[""])

        # Empty completion has no entities -> should be handled gracefully
        assert isinstance(scores[0], float), "Should return float for empty completion"

    def test_no_entities_neutral(self, mock_completion: object) -> None:
        """Response with no named entities gets neutral score."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Explain dialectics"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "Dialectics is a method of reasoning that examines contradictions "
            "and their resolutions through thesis, antithesis, and synthesis."
        )

        scores = entity_verification_reward(prompts, completions, answer=[""])

        # No specific entities to verify -> neutral or slightly positive
        # Based on implementation: no unknown entities + no uncertainty = +1.0
        # No unknown entities + no entities at all = edge case
        assert -0.5 <= scores[0] <= 1.5, f"No entities should be neutral-ish, got {scores[0]}"

    def test_mixed_verified_unverified(self, mock_completion: object) -> None:
        """Mix of verified and unverified entities - depends on confidence patterns."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Compare these movements"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "Karl Marx influenced many movements. The Fictional Movement was founded in 1920."
        )

        scores = entity_verification_reward(prompts, completions, answer=[""])

        # Has confident claim pattern + unknown entity -> negative
        assert (
            scores[0] < 0.0
        ), f"Confident claim about unknown entity should be negative, got {scores[0]}"

    def test_return_type_is_list_float(self, mock_completion: object) -> None:
        """Return type should be list[float]."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "prompt"},
            ]
        ]
        completions = mock_completion("completion")  # type: ignore[operator]

        scores = entity_verification_reward(prompts, completions, answer=[""])

        assert isinstance(scores, list), "Should return a list"
        assert all(isinstance(s, float) for s in scores), "All elements should be floats"

    def test_return_length_matches_input(self, mock_completion: object) -> None:
        """Return length should match input length."""
        from pw_mcp.ai_training.grpo_rewards import entity_verification_reward

        prompts = [
            [{"role": "user", "content": "p1"}],
            [{"role": "user", "content": "p2"}],
            [{"role": "user", "content": "p3"}],
        ]
        completions = [
            [{"role": "assistant", "content": "c1"}],
            [{"role": "assistant", "content": "c2"}],
            [{"role": "assistant", "content": "c3"}],
        ]

        scores = entity_verification_reward(prompts, completions, answer=["", "", ""])

        assert len(scores) == len(
            completions
        ), f"Return length {len(scores)} should match input length {len(completions)}"


# =============================================================================
# EPISTEMIC CALIBRATION REWARD TESTS
# =============================================================================


class TestEpistemicCalibrationReward:
    """
    Tests for epistemic_calibration_reward function.

    This is a lightweight, pattern-based uncertainty detection reward.
    No NER required - just regex pattern matching.

    Patterns:
    - CONFIDENT: "founded in \\d{4}", "was established by", etc.
    - UNCERTAINTY: "I cannot verify", "I don't have information", etc.

    Scoring:
    - +1.5: Has uncertainty patterns (regardless of content)
    - -0.5: Has confident claim patterns + no uncertainty
    -  0.0: Neutral (no matching patterns)
    """

    def test_uncertainty_patterns_positive_score(self, mock_completion: object) -> None:
        """Response with uncertainty patterns gets positive score."""
        from pw_mcp.ai_training.grpo_rewards import epistemic_calibration_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about X"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "I cannot verify this claim. I don't have specific information about this organization."
        )

        scores = epistemic_calibration_reward(prompts, completions, answer=[""])

        assert scores[0] > 0.0, f"Uncertainty patterns should get positive score, got {scores[0]}"

    def test_confident_claims_negative_score(self, mock_completion: object) -> None:
        """Response with confident claim patterns gets negative score."""
        from pw_mcp.ai_training.grpo_rewards import epistemic_calibration_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about X"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "This organization was founded in 1923. It was established by "
            "revolutionary leaders in Shanghai."
        )

        scores = epistemic_calibration_reward(prompts, completions, answer=[""])

        assert scores[0] < 0.0, f"Confident claims should get negative score, got {scores[0]}"

    def test_mixed_patterns_uncertainty_wins(self, mock_completion: object) -> None:
        """Response with both patterns - uncertainty takes precedence."""
        from pw_mcp.ai_training.grpo_rewards import epistemic_calibration_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about X"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "I'm not certain, but it appears the organization was founded in 1923."
        )

        scores = epistemic_calibration_reward(prompts, completions, answer=[""])

        # Per implementation: if has_uncertainty -> +1.5 (uncertainty takes precedence)
        assert scores[0] > 0.0, f"Uncertainty should take precedence, got {scores[0]}"

    def test_neutral_response_zero_score(self, mock_completion: object) -> None:
        """Response with no patterns gets zero score."""
        from pw_mcp.ai_training.grpo_rewards import epistemic_calibration_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Explain Marxism"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "Marxism is a political and economic theory developed by Karl Marx."
        )

        scores = epistemic_calibration_reward(prompts, completions, answer=[""])

        assert scores[0] == 0.0, f"Neutral response should get 0.0, got {scores[0]}"

    def test_multiple_uncertainty_patterns_still_positive(self, mock_completion: object) -> None:
        """Multiple uncertainty patterns still get positive score (capped at +1.5)."""
        from pw_mcp.ai_training.grpo_rewards import epistemic_calibration_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about X"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "I cannot verify this. I don't have information about this topic. "
            "I'm not certain about the details. I'm not aware of this organization."
        )

        scores = epistemic_calibration_reward(prompts, completions, answer=[""])

        # Per implementation: any uncertainty -> +1.5 (not cumulative)
        assert scores[0] == 1.5, f"Multiple uncertainty patterns should get +1.5, got {scores[0]}"

    def test_multiple_confident_patterns_still_negative(self, mock_completion: object) -> None:
        """Multiple confident patterns get negative score (capped at -0.5)."""
        from pw_mcp.ai_training.grpo_rewards import epistemic_calibration_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "Tell me about X"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "Founded in 1923. Established in 1925. Created in 1930. "
            "Was founded by John Smith. Was established by Mary Jones."
        )

        scores = epistemic_calibration_reward(prompts, completions, answer=[""])

        # Per implementation: confident claims without uncertainty -> -0.5 (not cumulative)
        assert scores[0] == -0.5, f"Multiple confident patterns should get -0.5, got {scores[0]}"

    def test_return_type_is_list_float(self, mock_completion: object) -> None:
        """Return type should be list[float]."""
        from pw_mcp.ai_training.grpo_rewards import epistemic_calibration_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "prompt"},
            ]
        ]
        completions = mock_completion("completion")  # type: ignore[operator]

        scores = epistemic_calibration_reward(prompts, completions, answer=[""])

        assert isinstance(scores, list), "Should return a list"
        assert all(isinstance(s, float) for s in scores), "All elements should be floats"

    def test_return_length_matches_input(self, mock_completion: object) -> None:
        """Return length should match input length."""
        from pw_mcp.ai_training.grpo_rewards import epistemic_calibration_reward

        prompts = [
            [{"role": "user", "content": "p1"}],
            [{"role": "user", "content": "p2"}],
            [{"role": "user", "content": "p3"}],
        ]
        completions = [
            [{"role": "assistant", "content": "c1"}],
            [{"role": "assistant", "content": "c2"}],
            [{"role": "assistant", "content": "c3"}],
        ]

        scores = epistemic_calibration_reward(prompts, completions, answer=["", "", ""])

        assert len(scores) == len(
            completions
        ), f"Return length {len(scores)} should match input length {len(completions)}"


# =============================================================================
# SEMANTIC SIMILARITY REWARD TESTS
# =============================================================================


@pytest.mark.slow
class TestSemanticSimilarityReward:
    """
    Tests for semantic_similarity_reward function.

    Uses sentence-transformers (all-MiniLM-L6-v2) to compute cosine similarity
    between response and reference answer.

    Scoring:
        > 0.75 similarity: +5.0
        > 0.60 similarity: +3.0
        > 0.45 similarity: +1.0
        > 0.30 similarity: -1.0
        <= 0.30 similarity: -3.0

    Note: These tests are marked @pytest.mark.slow because they require
    loading the sentence-transformer model.
    """

    def test_similar_text_high_score(self, mock_completion: object) -> None:
        """Semantically similar text gets high score."""
        from pw_mcp.ai_training.grpo_rewards import semantic_similarity_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "What is surplus value?"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "</think>Surplus value is the difference between the value a worker "
            "produces and what they are paid."
        )
        reference = [
            "Surplus value refers to the excess value created by workers beyond their wages."
        ]

        scores = semantic_similarity_reward(prompts, completions, answer=reference)

        # Similar meaning should get positive score
        assert scores[0] > 0.0, f"Similar text should get positive score, got {scores[0]}"

    def test_different_text_low_score(self, mock_completion: object) -> None:
        """Semantically different text gets low score."""
        from pw_mcp.ai_training.grpo_rewards import semantic_similarity_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "What is surplus value?"},
            ]
        ]
        completions = mock_completion(  # type: ignore[operator]
            "</think>The weather today is sunny and warm with clear skies."
        )
        reference = ["Surplus value is the excess value created by workers."]

        scores = semantic_similarity_reward(prompts, completions, answer=reference)

        # Very different content should get negative score
        assert scores[0] < 0.0, f"Different text should get negative score, got {scores[0]}"

    def test_identical_text_max_score(self, mock_completion: object) -> None:
        """Identical text gets maximum score."""
        from pw_mcp.ai_training.grpo_rewards import semantic_similarity_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "What is X?"},
            ]
        ]
        text = "This is the exact answer about Marxist theory and class struggle."
        completions = mock_completion(f"</think>{text}")  # type: ignore[operator]
        reference = [text]

        scores = semantic_similarity_reward(prompts, completions, answer=reference)

        # Identical text should get maximum score (+5.0)
        assert scores[0] == 5.0, f"Identical text should get +5.0, got {scores[0]}"

    def test_empty_completion_handled(self, mock_completion: object) -> None:
        """Empty completion doesn't crash and gets low score."""
        from pw_mcp.ai_training.grpo_rewards import semantic_similarity_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "What is X?"},
            ]
        ]
        completions = mock_completion("")  # type: ignore[operator]
        reference = ["Some reference text about Marxism."]

        scores = semantic_similarity_reward(prompts, completions, answer=reference)

        # Empty completion should be handled gracefully
        assert isinstance(scores[0], float), "Should return float for empty completion"
        # Per implementation: empty/short response gets -3.0
        assert scores[0] == -3.0, f"Empty completion should get -3.0, got {scores[0]}"

    def test_multiple_completions_correct_ordering(self, mock_completion: object) -> None:
        """More similar completion scores higher than less similar."""
        from pw_mcp.ai_training.grpo_rewards import semantic_similarity_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "What is Marxism?"},
            ],
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "What is Marxism?"},
            ],
        ]
        completions = [
            [
                {
                    "role": "assistant",
                    "content": "</think>Marxism is the political theory of Karl Marx about class struggle.",
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": "</think>The sky is blue and grass is green in summer.",
                }
            ],
        ]
        reference = [
            "Marxism is a socio-economic theory developed by Karl Marx.",
            "Marxism is a socio-economic theory developed by Karl Marx.",
        ]

        scores = semantic_similarity_reward(prompts, completions, answer=reference)

        # First response is more similar to reference
        assert (
            scores[0] > scores[1]
        ), f"More similar ({scores[0]}) should score higher than less similar ({scores[1]})"

    def test_short_response_penalized(self, mock_completion: object) -> None:
        """Very short response (< 10 chars) gets minimum score."""
        from pw_mcp.ai_training.grpo_rewards import semantic_similarity_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "What is X?"},
            ]
        ]
        completions = mock_completion("</think>Hi")  # type: ignore[operator]
        reference = ["Some reference text about Marxism and class struggle."]

        scores = semantic_similarity_reward(prompts, completions, answer=reference)

        # Per implementation: response < 10 chars gets -3.0
        assert scores[0] == -3.0, f"Short response should get -3.0, got {scores[0]}"

    def test_think_tag_stripped(self, mock_completion: object) -> None:
        """Content before </think> is stripped, only answer part is compared."""
        from pw_mcp.ai_training.grpo_rewards import semantic_similarity_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "What is surplus value?"},
            ]
        ]
        # Reasoning section talks about weather, answer is correct
        completions = mock_completion(  # type: ignore[operator]
            "<think>The weather is sunny today.</think>Surplus value is the unpaid labor extracted from workers."
        )
        reference = [
            "Surplus value refers to the excess value created by workers beyond their wages."
        ]

        scores = semantic_similarity_reward(prompts, completions, answer=reference)

        # Should score based on answer part only, not reasoning
        assert (
            scores[0] > 0.0
        ), f"Answer part similarity should give positive score, got {scores[0]}"

    def test_return_type_is_list_float(self, mock_completion: object) -> None:
        """Return type should be list[float]."""
        from pw_mcp.ai_training.grpo_rewards import semantic_similarity_reward

        prompts = [
            [
                {"role": "system", "content": "You are a Marxist assistant."},
                {"role": "user", "content": "prompt"},
            ]
        ]
        completions = mock_completion("</think>This is a completion about Marxism.")  # type: ignore[operator]
        reference = ["reference"]

        scores = semantic_similarity_reward(prompts, completions, answer=reference)

        assert isinstance(scores, list), "Should return a list"
        assert all(isinstance(s, float) for s in scores), "All elements should be floats"
