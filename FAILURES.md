# Failure Analysis & System Reflection
**Project:** Agentic RAG Pipeline for AI Regulation

This document provides a deep-dive analysis of the critical **agentic logic failures** encountered during development. Unlike standard software bugs, these failures represent the "reasoning gaps" where the AI agent's decision-making logic failed to align with factual reality or user intent.

---

## 1. Algorithmic Failure: The "Similarity Dead Zone"
**Observed Problem:** The system initially struggled to reject queries like *"What AI regulations exist in Brazil or India?"* (Out-of-Scope) while simultaneously accepting niche facts like *"Frontier model thresholds"* (Factual).

**Root Cause:** 
The embedding model (`all-MiniLM-L6-v2`) produced similarity scores for "junk" queries (~0.56) that were actually **higher** than those for specific technical facts (~0.45). This occurred because the junk query matched frequent boilerplate terms ("AI," "Regulation," "Policy") found in every document, creating a high-similarity background "noise" that a static threshold could not filter.

**The Agentic Pivot (Sanitization):**
We implemented a **multi-stage gate**:
1.  **Score Dispersion:** If retrieval scores are too "flat," the system suspects background noise.
2.  **Agentic Sanitization:** For marginal scores, a `llama-3.1-8b` agent verifies domain relevance (EU, US, China, UK). This added a "reasoning layer" that math alone couldn't provide.

---

## 2. Logic Failure: The "Dominant Doc" (Factual-Synthesis Misrouting)
**Observed Problem:** Queries explicitly asking for cross-document comparisons (e.g., *"What penalty figures are cited across the documents?"*) were incorrectly routed to the **Factual** strategy.

**Root Cause:** 
The router's "Confidence-Gap" logic had a flaw: if one specific document chunk had a very high similarity score (e.g., >0.65), the system assumed it had found a "perfect match" and defaulted to the Factual route. This "hijacked" the synthesis request, causing the system to ignore other relevant documents and providing a single-source answer that missed the requested comparison.

**The Agentic Pivot:**
Priority was shifted to **Linguistic Signals**. We now prioritize synthesis keywords (e.g., "across," "compare," "both") over raw similarity scores. Even if one document is "loud," an explicit request for multi-source reasoning will now force the Synthesis route.

---

## 3. Reasoning Failure: Synthesis Noise & "Information Dilution"
**Observed Problem:** In complex synthesis queries, the system would sometimes provide "generic" or "vague" answers, even when the data was present.

**Root Cause:** 
By increasing the context window to **Top-7 chunks** for the Synthesis route, we inadvertently introduced "contextual noise." Legal documents contain heavy boilerplate. When 7 chunks were injected into the prompt, the specific answer was often "diluted" by surrounding legal definitions, leading to a "Lost in the Middle" syndrome where the model focused on the most prominent (but irrelevant) text.

**The Agentic Pivot:**
We implemented **Source Anchoring** in the synthesis prompt. The system is now explicitly instructed to map claims to specific chunks immediately. This forces the model to treat the 7 chunks as discrete evidence points rather than a single block of text.

---

## 4. Sensitivity Failure: Sanitization Over-Rejection
**Observed Problem:** The system would refuse to answer valid queries like *"List the AI guidelines for developers"* even though the documents contained them.

**Root Cause:** 
The **Agentic Sanitizer** was originally too strict. The prompt instructed the LLM to only answer YES if the question mentioned "Regulations" or "Acts." Since guidelines and treaties aren't technically "Acts," the sanitizer triggered a hard refusal. This created a "Brittle Agent" that lacked the semantic flexibility to understand that "Guidelines" are part of the "Regulation" domain.

**The Agentic Pivot:**
The Sanitizer prompt was refactored to focus on **Area of Interest** (Domain) rather than **Document Type**. This allows the agent to be "Semantic" rather than "Literal" in its gatekeeping.

---

## 5. Ongoing Limitation: Contextual Contradiction Anchoring
**Reflection:** 
The system still struggles to reconcile conflicting values when they are not physically close in the retrieved text. For example, if one document cites a fine of "30 Million" and another "35 Million," the system only catches this if *both* chunks make it into the Top-K. 

**Future Work:** 
The next evolution would be **Iterative Refining**—where the agent generates a draft answer, identifies an ambiguity (like a missing fine amount), and performs a *targeted* second-pass search for that specific fact. This "multi-turn" retrieval was out of scope for the current architecture but is the logical next step for 100% accuracy.

---

## 6. Structural Failure: Chunk Boundary Loss (Semantic Truncation)
**Observed Problem:** The system occasionally provided technically correct but "dangerously incomplete" answers regarding specific legal penalties.

**Root Cause:** 
Our ingestion pipeline uses a fixed chunk size of 400 tokens. In legal documents, complex clauses are often very long. If a penalty amount is mentioned in the first half of a clause and the *qualifying condition* for that penalty is in the second half, the splitter may cut the clause in two. If the vector search only retrieves the "amount" chunk, the Agent provides the figure without the critical legal context, leading to a loss of semantic integrity.

**The Agentic Pivot:**
While not fully implemented in this version, we attempted to mitigate this by increasing **Chunk Overlap (to 80 tokens)**. This ensures that the context at the "cut point" is preserved in both adjacent chunks, reducing the risk of a fact being stranded without its surrounding logic.

---
*End of Report*
