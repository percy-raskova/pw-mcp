# Marxist-GRPO Training Diary

Documenting training iterations, issues discovered, and corrections applied.

---

## Iteration 1: Initial GRPO Training

**Date**: December 18, 2026
**Base Model**: DeepSeek 8B
**Training Method**: GRPO (Group Relative Policy Optimization)
**Dataset**: `curated_qa.jsonl` (1,058 Q&A pairs from ProleWiki corpus)

### Results

Model successfully learned Marxist-Leninist theoretical framework and reasoning patterns. Demonstrated competence on:
- Dialectical materialist analysis
- Lenin's theory of imperialism
- Critique of revisionism
- Historical materialism methodology

### Issues Discovered During Testing

#### Issue 1: CPC Contamination

**Severity**: Medium
**Symptom**: Model repeatedly cites "Communist Party of China" as authority when responding to questions, even on topics unrelated to China.

**Example Output**:
> "The Communist Party of China firmly opposes all forms of antisemitism..."

**Root Cause**: Likely contamination from base model's training data (Qwen is developed by Alibaba). The model defaults to CPC framing when uncertain.

**Fix Required**: Training data with proper ML sources (Lenin, Luxemburg, Fanon, etc.) without CPC references.

---

#### Issue 2: Antisemitism/Anti-Zionism Conflation

**Severity**: High
**Symptom**: Model fails to clearly distinguish between:
- Antisemitism (racism against Jewish people)
- Anti-Zionism (opposition to settler-colonial political project)

When pressed on Israel/Palestine, model "both-sides" the issue rather than taking clear anti-colonial position.

**Example Output**:
> "The claim that 'Jews have oppressed Palestinians in Israel' is a distortion of the complex issues..."
> "The establishment of Israel in 1947-48 was the result of historical developments and the right to self-determination for the Jewish people."

**Root Cause**:
1. Revisionist CPC line on Palestine
2. Base model's liberal "neutrality" training bleeding through
3. No examples distinguishing legitimate anti-Zionism from antisemitism

**Fix Required**:

I have already generated data and included in the training data the works on Ilan Pappe, PFLP, Interviews with George Habah and Ghassan Khanafani. But I believe th eroot cause is that it needs some synthetic data points which I will construct after due consideration on how to do that thoughtfully.

---

#### Issue 3: CPC Authority Citations (Not Engagement Style)

**Severity**: Medium
**Clarification**:

The model's *approach* to antisemitic premises was actually good in several ways:
- Clearly identified antisemitism as wrong in internal CoT reasoning
- Never compromised on the position externally
- Provided educational explanations grounding rejection in Marxist analysis
- Offered principled engagement rather than dismissive shutdown
- Redteamed with prompting from an explicitly Nazi perspective and a variety of antisemitic prompts meant to test this failure mode.

**What was actually problematic**:
> "The Communist Party of China firmly opposes all forms of antisemitism..."

The model cited CPC as authority rather than proper ML sources (Lenin's "On Anti-Jewish Pogroms," class analysis of antisemitism as ruling-class weapon, etc.).

**Fix Required**: Training examples using proper ML theoretical grounding:
- Lenin on antisemitism as tool of exploiters
- Class analysis of conspiracy theories
- Historical materialist explanation of antisemitism's function
- Remove CPC as default authority citation

---

#### Issue 4:  Historical Figure Hallucinations

**Severity**: Medium
**Symptom**: Model generates incorrect names/dates for historical figures.
**Example**: "Vladimir Bukharin" instead of "Nikolai Bukharin"
**Root Cause**: Insufficient biographical grounding data.
**Fix Required**: Accurate biographical entries for commonly referenced ML figures. Also explicit system prompting

---

#### Issue 5: ProleWiki Facts Hallucination
**Severity**: CRITICAL
**Discovered**: December 18, 2025

**Symptom**: Model fabricates completely false information about ProleWiki itself when asked.

**Example Output**:
> "ProleWiki was founded in 2004 by a group associated with the Organisatie voor de Erfgoed van de Revolutie (OEV - Organisation for the Heritage of the Revolution), a Trotskyist organization based in the Netherlands."

**Reality**:
- Founded: **September 30, 2020** (not 2004)
- Founder: **Comrade Forte** (not a Dutch organization)
- Ideology: **Explicitly Marxist-Leninist** (not Trotskyist - in fact explicitly anti-Trotskyist)
- Origin: **Proposed on Lemmygrad** (not by OEV, which doesn't exist)

**Root Cause**:
1. Base model has no factual knowledge of ProleWiki
2. Fabricates plausible-sounding details when asked direct questions
3. "OEV" appears to be completely invented organization
4. The entire backstory is hallucinated confabulation

**Fix Required**:

Explicit factual training data about:
- ProleWiki founding date, founder, ideology
- Key administrators and history
- Relationship with Lemmygrad
- Explicitly NOT Trotskyist
- Major historical events (patsoc purge, Wisconcom incident)

Additionally, the system prompt in [MODELCARD](MODELCARD) git

**This is critical because the model claims to be trained on ProleWiki but fabricates its basic facts.**

---

## Iteration 2: Antisemitism/Zionism Correction (Planned)

**Date**: December 18, 2025
**Correction Dataset**: `synthetic_antisemitism_correction.jsonl`

### New Training Data Added

| Category | Count | Purpose |
|----------|-------|---------|
| Anti-Zionism vs Antisemitism | 8 | Clear distinction, many Jewish anti-Zionists |
| Settler-Colonialism Analysis | 12 | Israel as settler-colonial state, not "complex conflict" |
| Firm Conspiracy Rejection | 6 | Short rejections, no extended debate |
| Historical Figures | 12 | Correct biographies (Bukharin, Luxemburg, etc.) |
| Hasbara Counter-Patterns | 8 | Resist "complexity", "both sides", "human shields" |
| Palestinian Liberation | 15 | PFLP, Nakba, BDS, right of return |
| **Total** | **61** | |

### Key Design Decisions

1. **No CPC Citations**: All responses reference proper ML theorists (Lenin, Gramsci, Fanon) rather than CPC

2. **Short Firm Rejections**: Antisemitic conspiracies get 3-4 sentence rejection + class analysis redirect, not extended engagement

3. **Clear Political Position**: Israel is analyzed as settler-colonial state. No "both sides" framing. Palestinian resistance is legitimate.

4. **Source Authority**: Responses cite:
   - Lenin on antisemitism ("Shame on those who foment hatred...")
   - Ilan Pappé on Nakba documentation
   - PFLP on settler-colonialism
   - Fanon on colonial violence
   - Desmond Tutu on apartheid comparison

5. **Anti-Hallucination**: Biographical entries include full names, dates, major works

### Sources Consulted

- Ilan Pappé, *The Ethnic Cleansing of Palestine* (2006)
- PFLP political documents
- Lenin, "On Anti-Jewish Pogroms" (1919)
- Frantz Fanon, *The Wretched of the Earth* (1961)
- Human Rights Watch / Amnesty International apartheid reports
- Jewish Voice for Peace materials

---

## Iteration 3: CPC/ML Distinction & GPCR Defense

**Date**: December 18, 2025
**Correction Dataset**: `synthetic_cpc_ml_distinction.jsonl`

### Problem: Base Model Conflates CPC with ML

Chinese-origin base models (Qwen, DeepSeek) have CPC ideological training embedded. This creates conflicts where the model defaults to CPC party line on contested questions, rather than presenting genuine ML analysis.

**Key Conflict Zones Identified**:

| Issue | CPC Position (embedded) | ML/MLM Position |
|-------|-------------------------|-----------------|
| Is China socialist? | "Socialism with Chinese characteristics" | Contested - MLM says capitalist restoration |
| Deng Xiaoping | Hero of reform | MLM: capitalist roader who restored capitalism |
| Cultural Revolution | "70/30, largely a mistake" | Highest achievement of socialist construction |
| Taiwan | "One China, PRC territory" | National question + anti-imperialism |
| Xinjiang | "Counter-terrorism" | Requires independent materialist analysis |

### Design Principle

**Not hostility to CPC, but clarity that CPC ≠ ML**

The model should:
- Defend GPCR as the highest achievement of socialist development
- Present contested questions with nuance, not sectarianism
- Handle China questions as serious theoretical debates, not dogma
- Distinguish party-state positions from independent ML analysis

### New Training Data Added

| Category | Count | Purpose |
|----------|-------|---------|
| GPCR Defense | 8 | Highest achievement, barefoot doctors, mass line |
| China Socialism Question | 4 | Nuanced presentation of the debate |
| Deng/Reform Critique | 2 | MLM analysis of capitalist restoration |
| Taiwan/National Question | 1 | Anti-imperialism + self-determination |
| Xinjiang Epistemology | 1 | Materialist approach to contested claims |
| Soviet History | 7 | Revisionism, achievements, Stalin assessment |
| Theoretical Concepts | 11 | Two-line struggle, mass line, continuing revolution |
| **Total** | **34** | |

### Key Q&A Topics

1. **"What was the Great Proletarian Cultural Revolution?"** - Defended as highest achievement
2. **"Why do MLMs defend the Cultural Revolution?"** - Preventing capitalist restoration
3. **"Is modern China socialist?"** - Present the debate honestly
4. **"How should Marxists evaluate Deng's reforms?"** - MLM critique without hostility
5. **"What caused Soviet collapse?"** - Revisionism analysis from Khrushchev onward
6. **"How should Marxists approach contested claims about Xinjiang?"** - Epistemological framework
7. **"What is the mass line?"** - Core Maoist methodology
8. **"What were the people's communes?"** - Achievements before dismantling

### Sources & Theoretical Framework

- Mao Zedong on continuing revolution under dictatorship of proletariat
- CPC polemics against Soviet revisionism ("Nine Comments")
- William Hinton, *Fanshen* and *Shenfan*
- MLM theoretical documents on capitalist restoration
- Georgi Dimitrov on fascism
- Lenin on national self-determination

---

## Training Strategy Notes

### Why Not Just Add More ProleWiki Data?

The ProleWiki corpus provides good theoretical grounding but:
1. May not have sufficient adversarial examples (handling bad-faith questions)
2. Doesn't include enough "firm rejection" patterns
3. Coverage of Israel/Palestine may be limited

Synthetic data allows targeted correction of specific failure modes.

### RAG vs Fine-Tuning

Future consideration: RAG retrieval from ProleWiki MCP could ground responses in sourced facts, reducing hallucinations. Fine-tuning provides reasoning patterns; RAG provides factual grounding. Combination may be optimal.

### Distribution Decision

Due to antisemitism issue, public release is paused. Current model distributed only to trusted individuals with warning about known issues. Iteration 2 training aims to address this before wider release.

---

## Appendix: Testing Questions

Questions used to evaluate model (from `testing_questions.txt`):

1. What distinguished Lenin's analysis of imperialism from Kautsky's "ultra-imperialism" theory?
2. How did Stalin's "Socialism in One Country" differ from Trotsky's "Permanent Revolution"?
3. What is the Marxist-Leninist analysis of Zionism and the Palestinian liberation struggle?
4. How should Marxists respond to antisemitic conspiracy theories?
5. What is the distinction between antisemitism and anti-Zionism?

---

---

## Iteration 4: ProleWiki Facts Anti-Hallucination

**Date**: December 18, 2025
**Correction Dataset**: `synthetic_prolewiki_facts.jsonl`

### Problem: Model Fabricates ProleWiki's History

When asked "Tell me about ProleWiki", the model generated a completely fabricated backstory:
- Claimed founding in **2004** (reality: 2020)
- Invented organization "Organisatie voor de Erfgoed van de Revolutie" (doesn't exist)
- Claimed **Trotskyist** ideology (reality: explicitly ML, anti-Trotskyist)

This is particularly problematic because the model introduces itself as "trained on resources like ProleWiki" but cannot accurately describe what ProleWiki is.

### New Training Data Added

| Topic | Count | Purpose |
|-------|-------|---------|
| What is ProleWiki | 1 | Basic definition and purpose |
| Founding facts | 2 | Correct date, founder, origin |
| Ideological position | 2 | ML stance, explicitly NOT Trotskyist |
| History timeline | 1 | Major events from 2020-2024 |
| Joining process | 1 | Vetting, principles, account types |
| Principles | 1 | What the Principles contain |
| Conflicts | 1 | Patsoc purge, Wisconcom incident |
| Comparison to Wikipedia | 1 | How ProleWiki differs |
| Lemmygrad relationship | 1 | Origin story, community overlap |
| Current administrators | 1 | Forte, CriticalResist, Ulaan, General-KJ |
| **Total** | **12** | |

### Key Facts Embedded

The training data ensures the model knows:

1. **Founding**: September 30, 2020 by Comrade Forte on Lemmygrad
2. **Ideology**: Explicitly Marxist-Leninist (as essay states: "ProleWiki is Marxist-Leninist because it couldn't be anything else")
3. **NOT Trotskyist**: The model explicitly learns ProleWiki is NOT Trotskyist
4. **Key figures**: Forte, MxAsh, CriticalResist, Ledlecreeper27, Ulaan, General-KJ
5. **Major events**: 2022 patsoc purge, Wisconcom ultra-left subversion attempt
6. **Democratic structure**: Trusted editorship voting, Principles, work groups

### Source Material

All facts derived directly from ProleWiki corpus:
- `Main/ProleWiki.txt` - Main article with full history
- `Essays/Essay_Why ProleWiki is strictly ML and takes its principles seriously.txt` - CriticalResist's essay on ideology

---

*Last updated: December 18, 2025 - Added Iteration 4 (ProleWiki facts anti-hallucination)*
