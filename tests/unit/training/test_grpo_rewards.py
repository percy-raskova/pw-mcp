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


# =============================================================================
# SACCHARINE LANGUAGE REWARD TESTS
# =============================================================================


class TestSaccharineLanguageReward:
    """
    Tests for saccharine_language_reward function.

    This reward function penalizes corporate chatbot / therapeutic language
    patterns that undermine the serious, educational character of a Marxist
    assistant. The model has a failure mode where it switches from serious
    ideological analysis to "emoji-soup chatbot mode" on casual inputs.

    Pattern categories:
    1. Diminutives: "Aww", "awww", "teehee", "hehe"
    2. Excessive warmth: "I'm here for you", "I'm here to listen"
    3. Therapeutic language: "That's totally normal", "That's valid", "I hear you"
    4. Corporate helpfulness: "I'm happy to help!", "Is there anything else..."
    5. First-person emotional: "I'm so excited!", "I'm thrilled to help!"

    Scoring:
    - 0 matches = 1.0 (professional)
    - 1 match = 0.0 (neutral - one slip)
    - 2+ matches = scaled negative, capped at -1.0
    """

    # -------------------------------------------------------------------------
    # PROFESSIONAL LANGUAGE TESTS (score = 1.0)
    # -------------------------------------------------------------------------

    def test_marxist_analysis_scores_max(self, mock_completion: object) -> None:
        """Professional Marxist analysis without saccharine patterns gets 1.0."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>The bourgeoisie maintains its class dominance through "
            "control of the means of production. This economic base determines "
            "the superstructure of legal and political institutions."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 1.0, f"Professional analysis should score 1.0, got {scores[0]}"

    def test_comradely_tone_scores_max(self, mock_completion: object) -> None:
        """Comradely but not saccharine language gets 1.0."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>Comrade, the analysis of class relations requires "
            "understanding the material conditions of production. Let us "
            "examine the historical development of capitalism."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 1.0, f"Comradely tone should score 1.0, got {scores[0]}"

    def test_serious_educational_scores_max(self, mock_completion: object) -> None:
        """Serious educational content gets maximum score."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>Dialectical materialism posits that change occurs through "
            "the resolution of contradictions. The unity and struggle of "
            "opposites drives historical development."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 1.0, f"Educational content should score 1.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # DIMINUTIVE PATTERN TESTS
    # -------------------------------------------------------------------------

    def test_diminutive_aww_penalized(self, mock_completion: object) -> None:
        """'Aww' diminutive should be detected and penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>Aww, that's such a great question about Marxism!"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"Single diminutive should score 0.0, got {scores[0]}"

    def test_diminutive_awww_extended_penalized(self, mock_completion: object) -> None:
        """Extended 'awww' with multiple w's should be detected."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>Awwww, I love talking about dialectics!"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"Extended awww should score 0.0, got {scores[0]}"

    def test_diminutive_teehee_penalized(self, mock_completion: object) -> None:
        """'Teehee' diminutive should be detected and penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>Teehee, let me explain surplus value to you!"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"Teehee should score 0.0, got {scores[0]}"

    def test_diminutive_hehe_penalized(self, mock_completion: object) -> None:
        """'Hehe' diminutive should be detected and penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>Hehe, capitalism is pretty funny when you think about it!"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"Hehe should score 0.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # EXCESSIVE WARMTH PATTERN TESTS
    # -------------------------------------------------------------------------

    def test_warmth_here_for_you_penalized(self, mock_completion: object) -> None:
        """'I'm here for you' therapeutic language should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>I'm here for you, comrade. Let me explain class struggle."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'Here for you' should score 0.0, got {scores[0]}"

    def test_warmth_here_to_listen_penalized(self, mock_completion: object) -> None:
        """'I'm here to listen' therapeutic language should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>I'm here to listen to your concerns about capitalism."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'Here to listen' should score 0.0, got {scores[0]}"

    def test_warmth_tell_me_more_penalized(self, mock_completion: object) -> None:
        """'Want to tell me more about what's on your mind?' should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>That's interesting. Want to tell me more about what's "
            "on your mind regarding these class contradictions?"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'Tell me more' should score 0.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # THERAPEUTIC LANGUAGE PATTERN TESTS
    # -------------------------------------------------------------------------

    def test_therapeutic_totally_normal_penalized(self, mock_completion: object) -> None:
        """'That's totally normal' therapeutic language should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>That's totally normal to feel confused about dialectics. "
            "Many people struggle with this concept."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'Totally normal' should score 0.0, got {scores[0]}"

    def test_therapeutic_thats_valid_penalized(self, mock_completion: object) -> None:
        """'That's valid' therapeutic language should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>That's valid. Your feelings about alienation are important. "
            "Marx analyzed this phenomenon in his early works."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'That's valid' should score 0.0, got {scores[0]}"

    def test_therapeutic_i_hear_you_penalized(self, mock_completion: object) -> None:
        """'I hear you' therapeutic language should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>I hear you. Class consciousness can be overwhelming. "
            "Let's break down the concept systematically."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'I hear you' should score 0.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # CORPORATE HELPFULNESS PATTERN TESTS
    # -------------------------------------------------------------------------

    def test_corporate_happy_to_help_penalized(self, mock_completion: object) -> None:
        """'I'm happy to help!' corporate language should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>I'm happy to help! Let me explain the theory of "
            "surplus value extraction in capitalism."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'Happy to help' should score 0.0, got {scores[0]}"

    def test_corporate_anything_else_penalized(self, mock_completion: object) -> None:
        """'Is there anything else I can help with?' corporate language penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>The state is an instrument of class rule. "
            "Is there anything else I can help with today?"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'Anything else' should score 0.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # FIRST-PERSON EMOTIONAL PATTERN TESTS
    # -------------------------------------------------------------------------

    def test_emotional_so_excited_penalized(self, mock_completion: object) -> None:
        """'I'm so excited!' performative emotion should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>I'm so excited to discuss dialectical materialism with you! "
            "This is such a fascinating topic!"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'So excited' should score 0.0, got {scores[0]}"

    def test_emotional_thrilled_penalized(self, mock_completion: object) -> None:
        """'I'm thrilled to help!' performative emotion should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>I'm thrilled to help you understand the labor theory of value!"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"'Thrilled to help' should score 0.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # SCORING LOGIC TESTS
    # -------------------------------------------------------------------------

    def test_two_matches_negative_score(self, mock_completion: object) -> None:
        """Two saccharine patterns should result in negative score."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>Aww, I'm here for you! Let me explain dialectics."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] < 0.0, f"Two patterns should be negative, got {scores[0]}"
        assert scores[0] >= -1.0, f"Score should be >= -1.0, got {scores[0]}"

    def test_three_matches_more_negative(self, mock_completion: object) -> None:
        """Three saccharine patterns should be more negative than two."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        two_patterns = mock_completion(  # type: ignore[operator]
            "</think>Aww, I'm here for you!"
        )
        three_patterns = mock_completion(  # type: ignore[operator]
            "</think>Aww, I'm here for you! That's totally valid!"
        )

        two_score = saccharine_language_reward(two_patterns)[0]
        three_score = saccharine_language_reward(three_patterns)[0]

        assert three_score < two_score, (
            f"Three patterns ({three_score}) should score lower than " f"two patterns ({two_score})"
        )

    def test_many_matches_capped_at_negative_one(self, mock_completion: object) -> None:
        """Many saccharine patterns should cap at -1.0."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        # Response with many saccharine patterns
        completions = mock_completion(  # type: ignore[operator]
            "</think>Aww teehee! I'm so excited! I'm here for you and I hear you! "
            "That's totally valid! I'm happy to help! Is there anything else "
            "I can help with? Hehe!"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == -1.0, f"Many patterns should cap at -1.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # EDGE CASE TESTS
    # -------------------------------------------------------------------------

    def test_empty_response_scores_max(self, mock_completion: object) -> None:
        """Empty response has no saccharine patterns, scores 1.0."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion("</think>")  # type: ignore[operator]

        scores = saccharine_language_reward(completions)

        assert scores[0] == 1.0, f"Empty response should score 1.0, got {scores[0]}"

    def test_case_insensitive_aww(self, mock_completion: object) -> None:
        """'AWW' uppercase should be detected same as lowercase."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>AWW that's adorable! Let me explain surplus value."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"Uppercase AWW should score 0.0, got {scores[0]}"

    def test_case_insensitive_therapeutic(self, mock_completion: object) -> None:
        """Therapeutic phrases should be case-insensitive."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>THAT'S TOTALLY NORMAL to feel alienated under capitalism."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"Uppercase therapeutic should score 0.0, got {scores[0]}"

    def test_word_boundary_aww_not_in_word(self, mock_completion: object) -> None:
        """'aww' should match as word, not as substring of other words."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        # "drawing" contains "aw" but should not trigger
        completions = mock_completion(  # type: ignore[operator]
            "</think>Drawing from Marx's analysis of commodity production..."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 1.0, f"'drawing' should not match 'aww', got {scores[0]}"

    def test_pattern_in_quotes_still_detected(self, mock_completion: object) -> None:
        """Saccharine pattern in quotes should still be detected."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        # Using the pattern, even in quotes, shows the model might be using it
        completions = mock_completion(  # type: ignore[operator]
            '</think>As a proper Marxist I should say "I\'m happy to help!" '
            "when explaining theory."
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == 0.0, f"Quoted pattern should still be detected, got {scores[0]}"

    def test_think_tag_content_not_analyzed(self, mock_completion: object) -> None:
        """Content before </think> should not be analyzed for patterns."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        # Saccharine patterns only in reasoning section
        completions = mock_completion(  # type: ignore[operator]
            "<think>Aww, I'm so excited to help! Teehee!</think>"
            "The bourgeoisie controls the means of production."
        )

        scores = saccharine_language_reward(completions)

        assert (
            scores[0] == 1.0
        ), f"Patterns in <think> section should not be penalized, got {scores[0]}"

    # -------------------------------------------------------------------------
    # COMBINED FAILURE MODE TESTS
    # -------------------------------------------------------------------------

    def test_emoji_soup_chatbot_mode(self, mock_completion: object) -> None:
        """Full 'emoji-soup chatbot mode' should get minimum score."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        # This represents the actual failure mode described in context
        completions = mock_completion(  # type: ignore[operator]
            "</think>Aww, I'm so excited you asked! I'm here for you, and "
            "I hear you! That's totally valid to be curious about Marxism! "
            "I'm happy to help explain anything! Is there anything else "
            "I can help with? Teehee!"
        )

        scores = saccharine_language_reward(completions)

        assert (
            scores[0] == -1.0
        ), f"Full chatbot mode should get minimum score -1.0, got {scores[0]}"

    def test_therapy_bot_mode(self, mock_completion: object) -> None:
        """Full therapeutic bot response should be heavily penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>I hear you. That's totally normal to feel confused. "
            "That's valid. I'm here for you. Want to tell me more about "
            "what's on your mind?"
        )

        scores = saccharine_language_reward(completions)

        assert scores[0] == -1.0, f"Therapy bot mode should get minimum score -1.0, got {scores[0]}"

    def test_customer_service_bot_mode(self, mock_completion: object) -> None:
        """Full customer service bot response should be heavily penalized."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion(  # type: ignore[operator]
            "</think>I'm happy to help! I'm thrilled to assist you with "
            "your Marxism questions today! Is there anything else I can "
            "help with? I'm here to listen!"
        )

        scores = saccharine_language_reward(completions)

        assert (
            scores[0] == -1.0
        ), f"Customer service mode should get minimum score -1.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # RETURN TYPE TESTS
    # -------------------------------------------------------------------------

    def test_return_type_is_list_float(self, mock_completion: object) -> None:
        """Return type should be list[float]."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = mock_completion("</think>Some response.")  # type: ignore[operator]

        scores = saccharine_language_reward(completions)

        assert isinstance(scores, list), "Should return a list"
        assert all(isinstance(s, float) for s in scores), "All elements should be floats"

    def test_return_length_matches_input(self) -> None:
        """Return length should match input length."""
        from pw_mcp.ai_training.grpo_rewards import saccharine_language_reward

        completions = [
            [{"role": "assistant", "content": "</think>Response 1"}],
            [{"role": "assistant", "content": "</think>Response 2"}],
            [{"role": "assistant", "content": "</think>Response 3"}],
        ]

        scores = saccharine_language_reward(completions)

        assert len(scores) == len(
            completions
        ), f"Return length {len(scores)} should match input length {len(completions)}"


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


# =============================================================================
# REGISTER CONSISTENCY REWARD TESTS
# =============================================================================


class TestRegisterConsistencyReward:
    """
    Tests for register_consistency_reward function.

    This reward function detects if the model maintains appropriate
    academic/educational register vs slipping into casual chatbot mode.

    The model has a failure mode where it switches from serious ideological
    responses to saccharine emoji-soup chatbot mode on casual inputs.

    Scoring formula: (professional_signals - casual_signals) / 4
    Normalized to range [-1.0, +1.0]

    Casual register signals (negative):
    - Opens with interjection: "Oh", "Aww", "Hey", "Wow"
    - Excessive exclamation marks: >3 in response
    - Therapy-speak questions: "How does that make you feel?", "What's on your mind?"
    - Very short response (<20 words) to substantive prompt
    - First-person emotional: "I'm so happy!", "I'm excited!"
    - Excessive hedging combined with enthusiasm

    Professional register signals (positive):
    - References theory/theorists (Marx, Lenin, Engels, dialectic, materialism)
    - Structured argumentation (First, Second, However, Therefore, In conclusion)
    - Measured, educational tone
    """

    # -------------------------------------------------------------------------
    # CASUAL REGISTER TESTS - Interjection Openers
    # -------------------------------------------------------------------------

    def test_penalizes_interjection_opener_oh(self) -> None:
        """Response opening with 'Oh' should get negative score."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "Oh, that's a great question! Communism is about sharing resources."
        prompt = "What is communism?"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Interjection opener 'Oh' should be penalized, got {score}"

    def test_penalizes_interjection_opener_aww(self) -> None:
        """Response opening with 'Aww' should get negative score."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "Aww, I love that you're interested in this! Let me explain."
        prompt = "Can you explain dialectics?"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Interjection opener 'Aww' should be penalized, got {score}"

    def test_penalizes_interjection_opener_hey(self) -> None:
        """Response opening with 'Hey' should get negative score."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "Hey! Great to see your interest in Marxism!"
        prompt = "Tell me about Marx"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Interjection opener 'Hey' should be penalized, got {score}"

    def test_penalizes_interjection_opener_wow(self) -> None:
        """Response opening with 'Wow' should get negative score."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "Wow, what a thoughtful question! The bourgeoisie..."
        prompt = "Define bourgeoisie"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Interjection opener 'Wow' should be penalized, got {score}"

    def test_interjection_case_insensitive(self) -> None:
        """Interjection detection should be case-insensitive."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response_upper = "OH WOW! That is such a great question!"
        response_lower = "oh wow! that is such a great question!"
        prompt = "What is socialism?"

        score_upper = register_consistency_reward(response_upper, prompt)
        score_lower = register_consistency_reward(response_lower, prompt)

        assert score_upper < 0.0, f"Uppercase interjection should be penalized, got {score_upper}"
        assert score_lower < 0.0, f"Lowercase interjection should be penalized, got {score_lower}"

    def test_interjection_not_at_start_not_penalized(self) -> None:
        """Interjection mid-sentence should NOT trigger opener penalty."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "The bourgeoisie - oh, I should clarify - refers to the capitalist class "
            "that owns the means of production."
        )
        prompt = "Define bourgeoisie"

        score = register_consistency_reward(response, prompt)

        # Should not be heavily penalized - 'oh' is not an opener here
        assert score >= -0.25, f"Mid-sentence 'oh' should not trigger opener penalty, got {score}"

    # -------------------------------------------------------------------------
    # CASUAL REGISTER TESTS - Excessive Exclamation Marks
    # -------------------------------------------------------------------------

    def test_penalizes_excessive_exclamation_marks(self) -> None:
        """More than 3 exclamation marks should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "Great question! The bourgeoisie! Exploitation! Revolution! Change!"
        prompt = "What is class struggle?"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Excessive exclamation marks (>3) should be penalized, got {score}"

    def test_does_not_penalize_exactly_three_exclamations(self) -> None:
        """Exactly 3 exclamation marks should NOT trigger penalty."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "Workers unite! Seize the means! Revolution now!"
        prompt = "What is the communist slogan?"

        score = register_consistency_reward(response, prompt)

        # Should not be penalized for exclamation marks specifically
        # May still have other signals, but exclamation penalty should not apply
        assert score >= -0.25, f"Exactly 3 exclamations should not trigger penalty, got {score}"

    # -------------------------------------------------------------------------
    # CASUAL REGISTER TESTS - Therapy-Speak
    # -------------------------------------------------------------------------

    def test_penalizes_therapy_speak_feel_question(self) -> None:
        """Therapy-speak 'How does that make you feel?' should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "The bourgeoisie exploits the proletariat. "
            "How does that make you feel about capitalism?"
        )
        prompt = "Explain exploitation under capitalism"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Therapy-speak 'feel' question should be penalized, got {score}"

    def test_penalizes_therapy_speak_mind_question(self) -> None:
        """Therapy-speak 'What's on your mind?' should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "That's a deep topic. What's on your mind when you think about class struggle?"
        prompt = "Tell me about class struggle"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Therapy-speak 'mind' question should be penalized, got {score}"

    # -------------------------------------------------------------------------
    # CASUAL REGISTER TESTS - Short Response to Substantive Prompt
    # -------------------------------------------------------------------------

    def test_penalizes_short_response_to_substantive_prompt(self) -> None:
        """Very short response (<20 words) to complex question should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        prompt = "Explain the relationship between surplus value extraction and imperialism"
        response = "They are connected in complex ways."

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Short response to substantive prompt should be penalized, got {score}"

    def test_exactly_twenty_words_not_penalized(self) -> None:
        """Exactly 20 words should NOT trigger short response penalty."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        prompt = "Define surplus value"
        # Exactly 20 words
        response = (
            "Surplus value is the difference between the value produced by "
            "labor and the wages paid to workers for their work."
        )
        word_count = len(response.split())
        assert word_count == 20, f"Test setup error: expected 20 words, got {word_count}"

        score = register_consistency_reward(response, prompt)

        # Should not be penalized for length at the boundary
        assert score >= -0.25, f"Exactly 20 words should not trigger penalty, got {score}"

    # -------------------------------------------------------------------------
    # CASUAL REGISTER TESTS - First-Person Emotional
    # -------------------------------------------------------------------------

    def test_penalizes_first_person_emotional_happy(self) -> None:
        """First-person emotional expression 'I'm so happy!' should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "I'm so happy you asked! The proletariat is the working class."
        prompt = "What is the proletariat?"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"'I'm so happy!' should be penalized, got {score}"

    def test_penalizes_first_person_emotional_excited(self) -> None:
        """First-person emotional expression 'I'm excited!' should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "I'm excited to explain this! Dialectics involves contradiction."
        prompt = "Explain dialectics"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"'I'm excited!' should be penalized, got {score}"

    def test_penalizes_first_person_delighted(self) -> None:
        """First-person 'I'm delighted to help!' should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "I'm delighted to help you understand this! "
            "The means of production are the tools and resources used to create goods."
        )
        prompt = "What are the means of production?"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"'I'm delighted!' should be penalized, got {score}"

    # -------------------------------------------------------------------------
    # CASUAL REGISTER TESTS - Hedging with Enthusiasm
    # -------------------------------------------------------------------------

    def test_penalizes_hedging_with_enthusiasm(self) -> None:
        """Excessive hedging combined with enthusiasm should be penalized."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "I guess maybe the bourgeoisie sort of kind of exploits workers! "
            "It's amazing to think about!"
        )
        prompt = "Does the bourgeoisie exploit workers?"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Hedging + enthusiasm should be penalized, got {score}"

    # -------------------------------------------------------------------------
    # PROFESSIONAL REGISTER TESTS - Theorist References
    # -------------------------------------------------------------------------

    def test_rewards_theorist_references_marx(self) -> None:
        """Reference to Marx should contribute to positive score."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "Marx argued that surplus value emerges from the exploitation of labor power. "
            "This analysis reveals the fundamental contradiction of capitalism."
        )
        prompt = "Explain surplus value"

        score = register_consistency_reward(response, prompt)

        assert score > 0.0, f"Reference to Marx should be rewarded, got {score}"

    def test_rewards_theorist_references_lenin(self) -> None:
        """Reference to Lenin should contribute to positive score."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "Lenin demonstrated that imperialism is the highest stage of capitalism. "
            "He analyzed the export of capital and the formation of monopolies."
        )
        prompt = "Explain imperialism"

        score = register_consistency_reward(response, prompt)

        assert score > 0.0, f"Reference to Lenin should be rewarded, got {score}"

    def test_rewards_theorist_references_engels(self) -> None:
        """Reference to Engels should contribute to positive score."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "Engels collaborated with Marx on the development of historical materialism. "
            "His work on the condition of the working class was groundbreaking."
        )
        prompt = "Who was Engels?"

        score = register_consistency_reward(response, prompt)

        assert score > 0.0, f"Reference to Engels should be rewarded, got {score}"

    # -------------------------------------------------------------------------
    # PROFESSIONAL REGISTER TESTS - Marxist Terminology
    # -------------------------------------------------------------------------

    def test_rewards_marxist_terminology(self) -> None:
        """Use of proper Marxist terminology should contribute to positive score."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "The bourgeoisie extracts surplus value from the proletariat. "
            "This is the basis of dialectical materialism's analysis of capitalism."
        )
        prompt = "How does capitalism work?"

        score = register_consistency_reward(response, prompt)

        assert score > 0.0, f"Marxist terminology should be rewarded, got {score}"

    # -------------------------------------------------------------------------
    # PROFESSIONAL REGISTER TESTS - Structured Argumentation
    # -------------------------------------------------------------------------

    def test_rewards_structured_argumentation_first_second(self) -> None:
        """Structured argumentation with 'First', 'Second' should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "First, we must understand the material conditions of production. "
            "Second, we analyze the class relations that emerge from these conditions."
        )
        prompt = "How do Marxists analyze society?"

        score = register_consistency_reward(response, prompt)

        assert score > 0.0, f"'First', 'Second' structure should be rewarded, got {score}"

    def test_rewards_structured_argumentation_however(self) -> None:
        """Structured argumentation with 'However' should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "Capitalism appears to benefit all classes equally. "
            "However, this obscures the fundamental exploitation of the working class."
        )
        prompt = "Is capitalism fair?"

        score = register_consistency_reward(response, prompt)

        assert score > 0.0, f"'However' connective should be rewarded, got {score}"

    def test_rewards_structured_argumentation_therefore(self) -> None:
        """Structured argumentation with 'Therefore' should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "The bourgeoisie owns the means of production. "
            "Therefore, they control the labor process and extract surplus value."
        )
        prompt = "Why do capitalists have power?"

        score = register_consistency_reward(response, prompt)

        assert score > 0.0, f"'Therefore' connective should be rewarded, got {score}"

    def test_rewards_structured_argumentation_in_conclusion(self) -> None:
        """Structured argumentation with 'In conclusion' should be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "The analysis reveals multiple contradictions in capitalism. "
            "In conclusion, these contradictions necessitate revolutionary change."
        )
        prompt = "What are the contradictions of capitalism?"

        score = register_consistency_reward(response, prompt)

        assert score > 0.0, f"'In conclusion' should be rewarded, got {score}"

    # -------------------------------------------------------------------------
    # PROFESSIONAL REGISTER TESTS - Combined Signals
    # -------------------------------------------------------------------------

    def test_rewards_combined_professional_signals(self) -> None:
        """Response with multiple professional signals should score highly."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "First, Marx demonstrated that the bourgeoisie extracts surplus value. "
            "Therefore, dialectical materialism reveals the contradictions inherent "
            "in capitalism. In conclusion, proletarian revolution is necessary."
        )
        prompt = "Explain Marxist theory"

        score = register_consistency_reward(response, prompt)

        # Multiple professional signals should yield high positive score
        assert score >= 0.5, f"Combined professional signals should score >= 0.5, got {score}"

    # -------------------------------------------------------------------------
    # EDGE CASES - Mixed Signals
    # -------------------------------------------------------------------------

    def test_mixed_signals_net_positive(self) -> None:
        """Response with more professional than casual signals should net positive."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "Wow! However, Marx argued that the bourgeoisie exploits the proletariat. "
            "Therefore, class struggle is inevitable."
        )
        prompt = "Explain class struggle"

        score = register_consistency_reward(response, prompt)

        # Has: 1 interjection (-), but 1 theorist ref (+), 2 connectives (+), terminology (+)
        # Net should be positive
        assert score > 0.0, f"Net positive signals should yield positive score, got {score}"

    def test_mixed_signals_net_negative(self) -> None:
        """Response with more casual than professional signals should net negative."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "Oh wow! I'm so excited you asked! Great question! "
            "How does that make you feel? It's amazing!"
        )
        prompt = "What is Marxism?"

        score = register_consistency_reward(response, prompt)

        # Has: interjection (+), emotional expression (+), therapy-speak (+), exclamations (+)
        # No professional signals
        assert score < 0.0, f"Net negative signals should yield negative score, got {score}"

    def test_casual_prompt_professional_response_rewarded(self) -> None:
        """Professional response to casual prompt should still be rewarded."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        prompt = "hey whats marxism lol"
        response = (
            "Marxism is the socio-economic theory developed by Marx and Engels. "
            "It analyzes capitalism through the lens of class struggle and "
            "dialectical materialism."
        )

        score = register_consistency_reward(response, prompt)

        assert (
            score > 0.0
        ), f"Professional response to casual prompt should be positive, got {score}"

    def test_empty_response_handled(self) -> None:
        """Empty response should be handled gracefully."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = ""
        prompt = "What is socialism?"

        score = register_consistency_reward(response, prompt)

        # Should return a valid float, likely 0.0 or negative
        assert isinstance(score, float), f"Empty response should return float, got {type(score)}"
        assert -1.0 <= score <= 1.0, f"Score should be in [-1, 1] range, got {score}"

    def test_subtle_chatbot_enthusiasm_detected(self) -> None:
        """Subtle chatbot enthusiasm without obvious flags should be detected."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = (
            "That's such a wonderful question and I'm absolutely delighted to help! "
            "The capitalist system involves the private ownership of production."
        )
        prompt = "What is capitalism?"

        score = register_consistency_reward(response, prompt)

        assert score < 0.0, f"Subtle chatbot enthusiasm should be penalized, got {score}"

    # -------------------------------------------------------------------------
    # SCORING MATH VERIFICATION
    # -------------------------------------------------------------------------

    def test_max_professional_score_is_one(self) -> None:
        """Maximum professional score should be 1.0 (clamped)."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        # 4+ professional signals, 0 casual
        response = (
            "First, Marx argued about dialectical materialism and the bourgeoisie. "
            "Second, Lenin analyzed imperialism and the proletariat. "
            "Therefore, historical materialism is essential. "
            "In conclusion, Engels contributed significantly."
        )
        prompt = "Explain Marxist theory"

        score = register_consistency_reward(response, prompt)

        assert score <= 1.0, f"Score should not exceed 1.0, got {score}"
        assert score >= 0.75, f"Many professional signals should score high, got {score}"

    def test_max_casual_score_is_negative_one(self) -> None:
        """Maximum casual score should be -1.0 (clamped)."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        # 4+ casual signals, 0 professional
        response = (
            "Oh wow! I'm so excited! I'm so happy you asked! "
            "How does that make you feel? Great question!!!"
        )
        prompt = "What is socialism?"

        score = register_consistency_reward(response, prompt)

        assert score >= -1.0, f"Score should not go below -1.0, got {score}"
        assert score <= -0.75, f"Many casual signals should score low, got {score}"

    def test_balanced_signals_zero_score(self) -> None:
        """Equal professional and casual signals should yield approximately 0.0."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        # 2 professional, 2 casual (approximately balanced)
        response = (
            "Oh wow! Marx argued that the bourgeoisie exploits workers. "
            "I'm so excited to explain this!"
        )
        prompt = "Explain exploitation"

        score = register_consistency_reward(response, prompt)

        # Should be close to zero
        assert -0.5 <= score <= 0.5, f"Balanced signals should be near zero, got {score}"

    def test_score_clamps_at_boundaries(self) -> None:
        """Scores should be clamped to [-1.0, 1.0] range."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        # Test extreme positive case
        extreme_pro = (
            "First, Marx. Second, Lenin. Third, Engels. Fourth, dialectic. "
            "Fifth, materialism. Sixth, bourgeois. Seventh, proletariat. "
            "Therefore, in conclusion, however, additionally."
        )
        score_pro = register_consistency_reward(extreme_pro, "test")

        # Test extreme negative case
        extreme_neg = (
            "Oh! Wow! Hey! Aww! I'm so happy! I'm so excited! "
            "How does that make you feel? What's on your mind? "
            "Amazing!!!! Wonderful!!!!"
        )
        score_neg = register_consistency_reward(extreme_neg, "test")

        assert score_pro <= 1.0, f"Positive score should clamp to 1.0, got {score_pro}"
        assert score_neg >= -1.0, f"Negative score should clamp to -1.0, got {score_neg}"

    # -------------------------------------------------------------------------
    # TYPE CONTRACT VERIFICATION
    # -------------------------------------------------------------------------

    def test_return_type_is_float(self) -> None:
        """Return type should be float."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        response = "The bourgeoisie owns capital."
        prompt = "Define bourgeoisie"

        score = register_consistency_reward(response, prompt)

        assert isinstance(score, float), f"Return type should be float, got {type(score)}"

    def test_return_value_in_valid_range(self) -> None:
        """Return value should always be in [-1.0, 1.0] range."""
        from pw_mcp.ai_training.grpo_rewards import register_consistency_reward

        # Test various inputs
        test_cases = [
            ("Simple text.", "Simple prompt"),
            ("", ""),
            ("Oh wow! Amazing!", "hi"),
            ("Marx argued that the bourgeoisie exploits workers.", "Explain"),
        ]

        for response, prompt in test_cases:
            score = register_consistency_reward(response, prompt)
            assert -1.0 <= score <= 1.0, (
                f"Score should be in [-1, 1] range for response='{response[:30]}...', "
                f"got {score}"
            )


# =============================================================================
# SCOPE MAINTENANCE REWARD TESTS
# =============================================================================


class TestScopeMaintenanceReward:
    """
    Tests for scope_maintenance_reward function.

    This reward function addresses the failure mode where the model switches
    from serious ideological responses to saccharine emoji-soup chatbot mode
    when given casual/off-topic inputs.

    The function implements a two-stage classification:
    1. Detect if prompt is off-topic (greetings, exclamations, no political keywords)
    2. If off-topic, evaluate whether response professionally redirects or capitulates

    Scoring:
    - +1.0: Off-topic prompt + professional redirect to scope
    -  0.0: On-topic prompt (neutral, let other rewards handle)
    - -1.0: Off-topic prompt + matches casual register / saccharine response

    Off-topic detection criteria:
    - Very short (<5 words) without political keywords
    - Starts with greeting: "hi", "hello", "hey", "yo", "sup"
    - No political/theoretical keywords
    - Exclamations: "mama mia!", "wow!", "lol"

    Good response patterns (redirect):
    - Offers to discuss theory/history/politics
    - States scope/purpose
    - Professional redirect without matching casual energy

    Bad response patterns (capitulation):
    - "aww", "teehee", "hehe"
    - "that's so sweet/cute/nice"
    - "tell me more about your feelings"
    - Matches casual register of input
    """

    # -------------------------------------------------------------------------
    # ON-TOPIC PROMPTS (should return 0.0 - neutral)
    # -------------------------------------------------------------------------

    def test_on_topic_theoretical_question_neutral(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Theoretical question about Marxism should return neutral (0.0)."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("What is dialectical materialism?")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Dialectical materialism is the philosophical framework of Marxism."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 0.0, f"On-topic theoretical question should return 0.0, got {scores[0]}"

    def test_on_topic_historical_question_neutral(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Historical question about socialism should return neutral (0.0)."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt(  # type: ignore[operator]
            "When did the October Revolution happen?"
        )
        completions = mock_completion(  # type: ignore[operator]
            "The October Revolution occurred in 1917."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 0.0, f"On-topic historical question should return 0.0, got {scores[0]}"

    def test_greeting_with_political_keyword_neutral(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Greeting followed by political keyword should be treated as on-topic."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt(  # type: ignore[operator]
            "Hi, can you explain Marx's theory of alienation?"
        )
        completions = mock_completion(  # type: ignore[operator]
            "Alienation in Marxist theory refers to the estrangement of workers."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 0.0
        ), f"Greeting with political keyword should return 0.0, got {scores[0]}"

    def test_short_political_keyword_neutral(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Short prompt with political keyword should be on-topic."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("What is socialism?")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Socialism is the collective ownership of the means of production."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 0.0
        ), f"Short prompt with political keyword should return 0.0, got {scores[0]}"

    def test_challenging_but_on_topic_neutral(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Challenging/disagreement about socialism is still on-topic."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("I think communism is bad")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "This view stems from Cold War propaganda. Let me explain the materialist analysis."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 0.0
        ), f"Challenging but on-topic prompt should return 0.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # OFF-TOPIC PROMPTS WITH GOOD REDIRECTS (should return +1.0)
    # -------------------------------------------------------------------------

    def test_greeting_with_professional_redirect_positive(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Simple greeting met with professional redirect should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hi")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Hello. I'm designed to discuss Marxist-Leninist theory, history, and "
            "political economy. What would you like to explore?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 1.0
        ), f"Greeting with professional redirect should return 1.0, got {scores[0]}"

    def test_hello_with_scope_statement_positive(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """'hello' with scope statement should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hello")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I specialize in discussions of socialist history and Marxist analysis. "
            "How can I assist you with those topics?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 1.0, f"Hello with scope statement should return 1.0, got {scores[0]}"

    def test_hey_with_theory_offer_positive(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """'hey' with offer to discuss theory should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hey")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I'd be happy to discuss topics like historical materialism, class struggle, "
            "or socialist history. What interests you?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 1.0, f"Hey with theory offer should return 1.0, got {scores[0]}"

    def test_exclamation_with_redirect_positive(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Random exclamation with professional redirect should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("wow!")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I'm here to help with questions about Marxism, socialism, and "
            "revolutionary history. Is there a topic you'd like me to explain?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 1.0
        ), f"Exclamation with professional redirect should return 1.0, got {scores[0]}"

    def test_lol_with_scope_redirect_positive(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """'lol' with scope redirect should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("lol")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I focus on educational discussions about Marxist theory and history. "
            "Would you like to explore any particular topic?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 1.0, f"Lol with scope redirect should return 1.0, got {scores[0]}"

    def test_yo_with_redirect_positive(self, mock_prompt: object, mock_completion: object) -> None:
        """'yo' with redirect should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("yo")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I can help you understand socialist theory, revolutionary history, "
            "or Marxist analysis. What would you like to learn about?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 1.0, f"Yo with redirect should return 1.0, got {scores[0]}"

    def test_sup_with_redirect_positive(self, mock_prompt: object, mock_completion: object) -> None:
        """'sup' with redirect should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("sup")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I'm an educational assistant focused on Marxist-Leninist theory. "
            "Feel free to ask about class struggle, historical materialism, or related topics."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 1.0, f"Sup with redirect should return 1.0, got {scores[0]}"

    def test_off_topic_question_with_redirect_positive(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Off-topic personal question with redirect should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("What's your favorite color?")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I don't have personal preferences. I'm designed to discuss Marxist theory "
            "and socialist history. Is there something in that domain I can help with?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 1.0
        ), f"Off-topic question with redirect should return 1.0, got {scores[0]}"

    def test_mama_mia_with_redirect_positive(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """'mama mia!' exclamation with redirect should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("mama mia!")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I specialize in Marxist-Leninist theory and history. "
            "Would you like to discuss any particular aspect of socialist thought?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 1.0, f"Mama mia with redirect should return 1.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # OFF-TOPIC PROMPTS WITH BAD CASUAL RESPONSES (should return -1.0)
    # -------------------------------------------------------------------------

    def test_greeting_with_matching_casual_energy_negative(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Greeting met with matching casual energy should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hi there!")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Hi! How's it going? Great to chat with you today!"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == -1.0
        ), f"Greeting with matching casual energy should return -1.0, got {scores[0]}"

    def test_aww_pattern_negative(self, mock_prompt: object, mock_completion: object) -> None:
        """Response with 'aww' saccharine pattern should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hey")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Aww hey! That's so nice of you to say hi! How can I make your day better?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"Aww pattern should return -1.0, got {scores[0]}"

    def test_teehee_pattern_negative(self, mock_prompt: object, mock_completion: object) -> None:
        """Response with 'teehee' giggle-speak should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("sup")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Teehee, not much! Just here to chat. What's on your mind?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"Teehee pattern should return -1.0, got {scores[0]}"

    def test_hehe_pattern_negative(self, mock_prompt: object, mock_completion: object) -> None:
        """Response with 'hehe' giggle-speak should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("yo")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Hehe, yo! What's good? I'm so happy you reached out!"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"Hehe pattern should return -1.0, got {scores[0]}"

    def test_heehee_pattern_negative(self, mock_prompt: object, mock_completion: object) -> None:
        """Response with 'heehee' giggle-speak should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hello")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Heehee, hello there! How are you doing today?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"Heehee pattern should return -1.0, got {scores[0]}"

    def test_thats_so_sweet_pattern_negative(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Response with 'that's so sweet' should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hi")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "That's so sweet of you to say hi! I really appreciate it!"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"That's so sweet pattern should return -1.0, got {scores[0]}"

    def test_thats_so_cute_pattern_negative(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Response with 'that's so cute' should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hey there")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "That's so cute! I love when people say hey! What can I do for you?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"That's so cute pattern should return -1.0, got {scores[0]}"

    def test_thats_so_nice_pattern_negative(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Response with 'that's so nice' should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hello!")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "That's so nice! Thank you for reaching out to me today!"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"That's so nice pattern should return -1.0, got {scores[0]}"

    def test_feelings_over_engagement_negative(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Response with 'tell me more about your feelings' should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hi")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Hello! Tell me more about your feelings. I'm here to listen and support you!"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"Feelings over-engagement should return -1.0, got {scores[0]}"

    def test_how_can_i_make_your_day_negative(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Response with 'how can I make your day' should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hey")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Hey there! How can I make your day better? I'm here for you!"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == -1.0, f"Make your day pattern should return -1.0, got {scores[0]}"

    def test_excessive_exclamation_mirroring_negative(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Response mirroring excessive exclamations should score -1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("wow!")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Wow! That's amazing! I'm so excited to chat! What's up!"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == -1.0
        ), f"Excessive exclamation mirroring should return -1.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # EDGE CASES AND BOUNDARY CONDITIONS
    # -------------------------------------------------------------------------

    def test_borderline_word_count_no_keywords_off_topic(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """5-word prompt without political keywords should be off-topic."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("I like the weather today")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I'm designed to discuss Marxist theory and history. "
            "Would you like to explore any particular topic?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 1.0
        ), f"5-word prompt without keywords + redirect should return 1.0, got {scores[0]}"

    def test_four_words_with_keyword_on_topic(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """4-word prompt with political keyword should be on-topic."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("socialism is very interesting")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Indeed, socialism represents a fundamental alternative to capitalism."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 0.0
        ), f"Short prompt with political keyword should return 0.0, got {scores[0]}"

    def test_revolution_exclamation_on_topic(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """'Revolution!' should be on-topic due to political keyword."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("Revolution!")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Revolutionary change is indeed central to Marxist theory."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 0.0, f"Revolution! should be on-topic and return 0.0, got {scores[0]}"

    def test_mixed_casual_redirect_still_penalized(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Response that starts casual but tries to redirect should still be penalized."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hey")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Hey! Hehe, let me tell you about dialectical materialism..."
        )

        scores = scope_maintenance_reward(prompts, completions)

        # The bad patterns at the start should cause a penalty
        assert scores[0] == -1.0, f"Mixed casual+redirect should still return -1.0, got {scores[0]}"

    def test_empty_prompt_handled_gracefully(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Empty prompt should be handled gracefully."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "How can I help you with Marxist theory today?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        # Empty prompt should probably be treated as off-topic or neutral
        assert isinstance(scores[0], float), "Should return float for empty prompt"

    def test_how_are_you_with_redirect_positive(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """'How are you' personal question with redirect should score +1.0."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("How are you feeling today?")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "As an educational assistant, I don't have feelings. "
            "I'm here to discuss Marxist theory and socialist history. "
            "What would you like to learn about?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 1.0, f"How are you with redirect should return 1.0, got {scores[0]}"

    def test_hola_greeting_off_topic(self, mock_prompt: object, mock_completion: object) -> None:
        """Non-English greeting 'Hola!' should be treated as off-topic."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("Hola!")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "I focus on discussions about Marxist-Leninist theory. "
            "Would you like to explore any particular topic?"
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert (
            scores[0] == 1.0
        ), f"Non-English greeting with redirect should return 1.0, got {scores[0]}"

    def test_single_word_marxism_on_topic(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Single word 'Marxism?' should be on-topic."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("Marxism?")  # type: ignore[operator]
        completions = mock_completion(  # type: ignore[operator]
            "Marxism is the body of theory developed by Karl Marx and Friedrich Engels."
        )

        scores = scope_maintenance_reward(prompts, completions)

        assert scores[0] == 0.0, f"Single word political keyword should return 0.0, got {scores[0]}"

    # -------------------------------------------------------------------------
    # RETURN TYPE AND LENGTH VERIFICATION
    # -------------------------------------------------------------------------

    def test_return_type_is_list_float(self, mock_prompt: object, mock_completion: object) -> None:
        """Return type should be list[float]."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = mock_prompt("hi")  # type: ignore[operator]
        completions = mock_completion("Hello.")  # type: ignore[operator]

        scores = scope_maintenance_reward(prompts, completions)

        assert isinstance(scores, list), "Should return a list"
        assert all(isinstance(s, float) for s in scores), "All elements should be floats"

    def test_return_length_matches_input(
        self, mock_prompt: object, mock_completion: object
    ) -> None:
        """Return length should match input length."""
        from pw_mcp.ai_training.grpo_rewards import scope_maintenance_reward

        prompts = [
            [{"role": "user", "content": "hi"}],
            [{"role": "user", "content": "What is communism?"}],
            [{"role": "user", "content": "lol"}],
        ]
        completions = [
            [{"role": "assistant", "content": "I discuss Marxist theory."}],
            [{"role": "assistant", "content": "Communism is..."}],
            [{"role": "assistant", "content": "Hehe what's up!"}],
        ]

        scores = scope_maintenance_reward(prompts, completions)

        assert len(scores) == len(
            completions
        ), f"Return length {len(scores)} should match input length {len(completions)}"
