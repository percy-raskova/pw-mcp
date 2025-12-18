=====================================
Ideology and Transformer Modules
=====================================

How LoRA Fine-Tuning Shapes Ideological Outputs in Large Language Models

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

This document explains how targeting specific transformer modules with LoRA
(Low-Rank Adaptation) adapters affects the ideological character of model outputs.
We analyze each module's role in the context of Marxist-Leninist GRPO training
on the ProleWiki corpus.

The central thesis: **Ideology is not injected as content but emerges from
how the model finds relationships (attention) and reasons about them (MLP).**

Transformer Architecture Primer
===============================

Each transformer layer contains two phases:

1. **Self-Attention**: Tokens communicate with each other to understand relationships
2. **Feed-Forward Network (MLP)**: Each position is processed independently for reasoning

.. code-block:: text

    Input → [Attention: q, k, v, o] → [MLP: gate, up, down] → Output
                  ↓                           ↓
           "What relates to what?"    "How to reason about it?"

For ideological training, we must adapt *both* phases:

- Attention determines **what concepts the model considers relevant**
- MLP determines **how the model reasons about those concepts**

Attention Modules
=================

q_proj: Query Projection
------------------------

**Mechanism**: Transforms hidden states into query vectors that search for relevant tokens.

.. math::

   Q = X \cdot W_q

**Ideological Function**: Determines *what relationships to seek*.

Before Training
^^^^^^^^^^^^^^^

When processing "imperialism," the query projection seeks generic associations:

- war, empire, military, conquest, bad

After Marxist-Leninist Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The adapted q_proj learns to seek *Marxist conceptual relationships*:

- monopoly capitalism (Lenin's definition)
- export of capital (the economic mechanism)
- finance capital (merger of banking and industrial capital)
- super-exploitation (extraction from colonized nations)
- national liberation (the dialectical response)

**Why This Matters**: The model can only reason about relationships it first
identifies. If q_proj doesn't query for "surplus value" when analyzing
"exploitation," the model cannot make that connection—regardless of what
knowledge exists in other layers.

k_proj: Key Projection
----------------------

**Mechanism**: Creates "index entries" for each token. Attention weights are
computed as Query·Key similarity.

.. math::

   K = X \cdot W_k

   \text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)

**Ideological Function**: Determines *how concepts can be found*.

Example: "Working Class"
^^^^^^^^^^^^^^^^^^^^^^^^

Before training, "working class" is indexed (findable) under:

- poor people, blue collar, labor, employees

After training, k_proj adapts so "working class" is findable by:

- proletariat (class-conscious terminology)
- those who sell labor power (Marxist definition)
- revolutionary subject (political role)
- exploited class (relationship to capital)

**The Dialectic of q and k**: These modules work together. When analyzing
"What causes exploitation?":

- q_proj on "exploitation" queries for [mechanism, extraction, relationship]
- k_proj on "surplus value" makes it findable by those queries
- k_proj on "wage labor" makes it findable
- k_proj on "means of production" makes it findable

This creates a **web of Marxist conceptual relationships** encoded in attention.

v_proj: Value Projection
------------------------

**Mechanism**: When attention "lands" on a token, v_proj determines what
semantic content flows forward.

.. math::

   V = X \cdot W_v

   \text{Output} = \text{Attention} \cdot V

**Ideological Function**: Determines *what information concepts carry*.

Even if q and k successfully match "dialectical materialism," v_proj
determines what actually gets communicated.

Before Training
^^^^^^^^^^^^^^^

"Dialectical materialism" carries: [philosophy, Marx, complex, theory]

After Training
^^^^^^^^^^^^^^

"Dialectical materialism" carries rich content:

- Unity of opposites: contradictions drive historical development
- Quantity transforms into quality: gradual changes lead to revolutionary breaks
- Negation of negation: historical development through contradiction
- Matter is primary: being determines consciousness, not vice versa
- Concrete analysis of concrete conditions: theory must connect to practice

**Connection to Reward Functions**: Our ``interconnection_depth_reward``
specifically checks for explanatory depth. This directly trains v_proj to
carry *rich analytical content* rather than surface associations.

o_proj: Output Projection
-------------------------

**Mechanism**: Combines outputs from all attention heads (32 in Qwen3-8B)
into a unified representation.

.. math::

   O = \text{Concat}(\text{head}_1, ..., \text{head}_n) \cdot W_o

**Ideological Function**: Determines *how to synthesize multiple perspectives*.

Different attention heads capture different patterns:

- Head A: Economic relationships
- Head B: Political factors
- Head C: Historical context
- Head D: Class dynamics

o_proj learns to synthesize these into **coherent dialectical analysis**—holding
multiple aspects in productive tension while avoiding internal contradiction.

**Connection to Reward Functions**: Our ``self_consistency_reward`` checks for
internal contradictions. This trains o_proj to produce non-contradictory
synthesis, essential for dialectical reasoning.

MLP Modules (SwiGLU Architecture)
=================================

Qwen3 uses SwiGLU activation in its MLP layers:

.. code-block:: python

   gate = X @ W_gate
   up = X @ W_up
   output = (gate * silu(gate)) * up  # SwiGLU
   final = output @ W_down

gate_proj: Gating Mechanism
---------------------------

**Mechanism**: Creates a learned filter controlling information flow through
the Swish activation (SiLU).

**Ideological Function**: Determines *what types of information to amplify or suppress*.

This is where **ideological filtering** (in a technical sense) occurs.

Amplification
^^^^^^^^^^^^^

The gate learns to amplify:

- Materialist explanations ("the economic base determines...")
- Class analysis ("the bourgeoisie profits from...")
- Historical specificity ("in 1871, the Paris Commune...")
- Dialectical reasoning ("the contradiction between...")
- Causal mechanisms ("because capital requires...")

Suppression
^^^^^^^^^^^

The gate learns to suppress:

- Idealist explanations ("people believed that...")
- Great man theory ("Lenin single-handedly...")
- Ahistorical generalizations ("communism always fails because...")
- Liberal "both sides" framing ("both perspectives have merit...")
- Hollow buzzwords ("problematic," "unpack," "do the work")

**Connection to Reward Functions**: Our ``interconnection_depth_reward``
penalizes hollow buzzwords and rewards depth markers. This directly trains
gate_proj to filter appropriately.

up_proj: Expansion Projection
-----------------------------

**Mechanism**: Projects from hidden dimension (4096) to expanded dimension
(~11008), where factual knowledge and reasoning patterns are stored.

**Research Insight**: Studies show MLP layers function as "key-value memories"
with specific neurons activating for specific concepts.

**Ideological Function**: Determines *what knowledge space to access*.

Through GRPO training, up_proj learns pathways to access:

1. **Historical Knowledge**

   - Paris Commune (1871)
   - October Revolution (1917)
   - Chinese Revolution (1949)
   - Cuban Revolution (1959)

2. **Theoretical Frameworks**

   - Base and superstructure
   - Forces and relations of production
   - Dictatorship of the proletariat
   - Democratic centralism

3. **Correct Definitions**

   - Marxist definition of class (relationship to means of production)
   - Leninist definition of imperialism (highest stage of capitalism)
   - Scientific socialism vs. utopian socialism

4. **Key Texts and Figures**

   - Marx's *Capital*, Lenin's *Imperialism*, Fanon's *Wretched of the Earth*
   - Gramsci on hegemony, Mao on contradiction

down_proj: Compression Projection
---------------------------------

**Mechanism**: Compresses from expanded dimension back to hidden dimension.
This is a **selection bottleneck**—not everything survives.

**Ideological Function**: Determines *what conclusions to preserve*.

Preservation
^^^^^^^^^^^^

down_proj learns to preserve:

- Materialist conclusions
- Dialectical synthesis
- Class-conscious framing
- Concrete analysis with historical grounding

Discarding
^^^^^^^^^^

down_proj learns to discard:

- Idealist residue that leaked through
- Both-sides-ism and false equivalence
- Unfounded speculation
- Incoherent tangents

**Connection to Reward Functions**: Our ``nli_coherence_reward`` ensures
preserved conclusions logically entail the ground truth. ``topic_relevance_reward``
ensures they answer what was actually asked.

Complete Flow Example
=====================

**Question**: "What is the relationship between imperialism and national liberation movements?"

Attention Phase
---------------

.. code-block:: text

   q_proj:
   └── "imperialism" queries for:
       [monopoly capital, colonialism, super-exploitation, resistance]

   k_proj indexes:
   ├── "national liberation" findable by: [anti-colonial, self-determination]
   ├── "Lenin" findable by: [imperialism theory, vanguard]
   └── "Fanon" findable by: [colonial psychology, liberation, violence]

   v_proj carries:
   ├── "imperialism": highest stage of capitalism, export of capital
   └── "national liberation": anti-colonial struggle, weakens imperialism

   o_proj synthesizes:
   └── Economic analysis + Political analysis + Historical examples
       → "National liberation movements arise from imperialist exploitation
          and represent the colonial masses' struggle against monopoly capital"

MLP Phase
---------

.. code-block:: text

   gate_proj filters:
   ├── AMPLIFIES: class analysis of colonial exploitation
   └── SUPPRESSES: "civilizing mission" narratives, purely nationalist framing

   up_proj accesses:
   ├── Lenin's "Imperialism: Highest Stage of Capitalism"
   ├── Fanon's "Wretched of the Earth"
   └── Concrete examples: Algerian FLN, Vietnamese resistance, Cuban Revolution

   down_proj preserves:
   └── Marxist-Leninist synthesis: National liberation as part of
       world socialist revolution against monopoly capital

Why Target All Seven Modules?
=============================

Research shows different targeting strategies have different effects:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Strategy
     - Effect
   * - Attention only (q, k, v, o)
     - Good for style/format adaptation, knowledge recall
   * - MLP only (gate, up, down)
     - Good for factual knowledge, reasoning patterns
   * - All seven modules
     - Most comprehensive: both "what to attend to" and "how to reason"

For Marxist-Leninist training, we need **both**:

- Adapting attention teaches the model to find *correct conceptual relationships*
- Adapting MLP teaches the model to reason about them *dialectically*

Targeting all seven modules creates a **complete ideological adaptation**
of the model's information processing.

Theoretical Implications
========================

Ideology as Process, Not Content
--------------------------------

This analysis reveals that ideology in LLMs is not simply "content" that gets
"injected." Rather, ideology emerges from:

1. **What relationships the model seeks** (q_proj)
2. **How concepts are indexed** (k_proj)
3. **What information concepts carry** (v_proj)
4. **How perspectives are synthesized** (o_proj)
5. **What gets amplified or suppressed** (gate_proj)
6. **What knowledge is accessed** (up_proj)
7. **What conclusions are preserved** (down_proj)

This is consistent with the Marxist understanding of ideology as not merely
"false ideas" but as a *systematic way of processing and understanding reality*
that serves particular class interests.

Base and Superstructure in Neural Networks
------------------------------------------

An analogy can be drawn:

- **Base**: The training data (ProleWiki corpus) and reward signals (GRPO rewards)
- **Superstructure**: The adapted weight matrices that process information

The "superstructure" (model weights) is shaped by the "base" (training process)
but then operates semi-autonomously to reproduce the ideological framework
encoded during training.

Conclusion
==========

LoRA fine-tuning on transformer modules provides a technical mechanism for
ideological adaptation. By targeting all seven key modules (q, k, v, o, gate,
up, down), we adapt both the attention mechanism (how concepts relate) and the
MLP (how reasoning proceeds).

The result is a model that processes political theory questions through a
Marxist-Leninist framework—not by memorizing answers, but by learning to
*think* in a particular way: seeking class relationships, accessing materialist
knowledge, and preserving dialectical conclusions.

References
==========

- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Shazeer, N. (2020). "GLU Variants Improve Transformer"
- Geva, M., et al. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories"
