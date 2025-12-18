# Marxist-GRPO Training Diary

Documenting training iterations, issues discovered, and corrections applied.

---

## Iteration 1: Initial GRPO Training

**Date**: December 2024
**Base Model**: Qwen3 8B
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
1. Insufficient training data on Palestine/Zionism from ML perspective
2. Base model's liberal "neutrality" training bleeding through
3. No examples distinguishing legitimate anti-Zionism from antisemitism

**Fix Required**: Explicit training data on:
- Settler-colonialism analysis of Israel
- Distinction between Judaism and Zionism
- PFLP and Palestinian liberation theory
- Ilan Pappé's historical research

---

#### Issue 3: CPC Authority Citations (Not Engagement Style)
**Severity**: Medium
**Clarification**: The model's *approach* to antisemitic premises was actually good in several ways:
- Clearly identified antisemitism as wrong in internal CoT reasoning
- Never compromised on the position externally
- Provided educational explanations grounding rejection in Marxist analysis
- Offered principled engagement rather than dismissive shutdown

**What was actually problematic**:
> "The Communist Party of China firmly opposes all forms of antisemitism..."

The model cited CPC as authority rather than proper ML sources (Lenin's "On Anti-Jewish Pogroms," class analysis of antisemitism as ruling-class weapon, etc.).

**Fix Required**: Training examples using proper ML theoretical grounding:
- Lenin on antisemitism as tool of exploiters
- Class analysis of conspiracy theories
- Historical materialist explanation of antisemitism's function
- Remove CPC as default authority citation

---

#### Issue 4: Gradual Accommodation on Zionism (Not Antisemitism)
**Severity**: High
**Clarification**: The model held firm on antisemitism throughout the conversation - that was good. The accommodation problem was specifically on the **Zionism/Israel question**.

**Symptom**: When conversation shifted from antisemitism to Israel/Palestine, model moved toward liberal "both sides" framing:
> "The establishment of Israel in 1947-48 was the result of historical developments and the right to self-determination for the Jewish people."
> "Israel, like any other nation, has the right to determine its own policies..."
> "The situation in Palestine is complex..."

**Root Cause**:
1. Model correctly distinguished antisemitism from anti-Zionism in principle
2. But when applying this to Israel specifically, defaulted to liberal "legitimacy" framing
3. Base model training treats Israel as normal state rather than settler-colonial project

**Fix Required**: Training data with clear settler-colonial analysis:
- Israel as settler-colonial state (Ilan Pappé, Patrick Wolfe)
- Palestinian resistance as legitimate national liberation
- Reject "complexity" framing as ideological mystification
- Maintain position across adversarial questioning on this specific topic

---

#### Issue 5: Historical Figure Hallucinations
**Severity**: Medium
**Symptom**: Model generates incorrect names/dates for historical figures.

**Example**: "Vladimir Bukharin" instead of "Nikolai Bukharin"

**Root Cause**: Insufficient biographical grounding data.

**Fix Required**: Accurate biographical entries for commonly referenced ML figures.

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

*Last updated: December 18, 2025 - Added Iteration 3 (CPC/ML distinction)*
