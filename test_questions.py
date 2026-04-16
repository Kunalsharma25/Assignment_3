# test_questions.py

TEST_QUESTIONS = [

    # ──────────────────────────────────────────────
    # FACTUAL QUESTIONS (5) — Answer in one document
    # ──────────────────────────────────────────────
    {
        "id": "F1",
        "query": "What are the four risk categories under the EU AI Act?",
        "expected_route": "factual",
        "expected_answer_keywords": [
            "unacceptable", "high risk", "limited risk", "minimal risk"
        ],
        "expected_source_docs": [
            "Document_1_Policy_Report.txt",
            "Document_4_Technical_Brief.txt"
        ],
        "notes": "Directly listed in Doc 1 Section 2.1 and Doc 4 Section 1"
    },
    {
        "id": "F2",
        "query": "What is the penalty for violating prohibited system rules under the EU AI Act?",
        "expected_route": "factual",
        "expected_answer_keywords": [
            "35 million", "7%", "global turnover"
        ],
        "expected_source_docs": [
            "Document_4_Technical_Brief.txt",
            "Document_2_News_Article.txt"
        ],
        "notes": "Specific penalty figure from Doc 4 Section 5 and Doc 2"
    },
    {
        "id": "F3",
        "query": "What is the frontier model threshold defined in the US Executive Order?",
        "expected_route": "factual",
        "expected_answer_keywords": [
            "10^26", "floating point", "computational", "frontier"
        ],
        "expected_source_docs": [
            "Document_4_Technical_Brief.txt",
            "Document_3_Stakeholder_Memo.txt"
        ],
        "notes": "Specific technical threshold from Doc 4 Section 3 and Doc 3"
    },
    {
        "id": "F4",
        "query": "What is the UK government's current approach to AI regulation?",
        "expected_route": "factual",
        "expected_answer_keywords": [
            "principles-based", "existing regulators", "no primary legislation",
            "prescriptive"
        ],
        "expected_source_docs": ["Document_3_Stakeholder_Memo.txt"],
        "notes": "UK position described only in Doc 3, Issue 3"
    },
    {
        "id": "F5",
        "query": "When did China's generative AI regulations come into force?",
        "expected_route": "factual",
        "expected_answer_keywords": ["August 2023", "generative AI", "China"],
        "expected_source_docs": [
            "Document_2_News_Article.txt",
            "Document_4_Technical_Brief.txt"
        ],
        "notes": "Specific date stated in Doc 2 and Doc 4 Section 6"
    },

    # ──────────────────────────────────────────────────────────────────
    # SYNTHESIS QUESTIONS (5) — Answer requires combining multiple docs
    # ──────────────────────────────────────────────────────────────────
    {
        "id": "S1",
        "query": "How do the EU, US, and China differ in their overall approach to AI regulation?",
        "expected_route": "synthesis",
        "expected_answer_keywords": [
            "EU", "United States", "China", "comprehensive",
            "sectoral", "generative", "differ"
        ],
        "expected_source_docs": None,
        "notes": "Requires combining Doc 1, Doc 2, Doc 3, Doc 4 — cross-jurisdiction comparison"
    },
    {
        "id": "S2",
        "query": "What penalty figures are cited across the documents for EU AI Act violations, and do they agree?",
        "expected_route": "synthesis",
        "expected_answer_keywords": [
            "30 million", "35 million", "6%", "7%", "inconsistency",
            "earlier draft", "final"
        ],
        "expected_source_docs": None,
        "notes": "Deliberate contradiction: Doc 1 says 30M/6%, Doc 2 and Doc 4 say 35M/7% — tests contradiction handling"
    },
    {
        "id": "S3",
        "query": "What common compliance requirements appear across multiple regulatory frameworks?",
        "expected_route": "synthesis",
        "expected_answer_keywords": [
            "transparency", "human oversight", "documentation",
            "accountability", "data", "safety"
        ],
        "expected_source_docs": None,
        "notes": "Requires synthesizing convergence themes from Doc 1 Section 3 and Doc 4"
    },
    {
        "id": "S4",
        "query": "How do the documents collectively describe the EU AI Office and its role?",
        "expected_route": "synthesis",
        "expected_answer_keywords": [
            "AI Office", "oversight", "standards", "implementation",
            "technical", "enforcement"
        ],
        "expected_source_docs": None,
        "notes": "AI Office mentioned in Doc 2, Doc 3, and Doc 4 — requires combining all three"
    },
    {
        "id": "S5",
        "query": "What challenges do smaller developers face compared to large enterprises under AI regulation, across the documents?",
        "expected_route": "synthesis",
        "expected_answer_keywords": [
            "smaller", "enterprise", "cost", "conformity",
            "workable", "compliance"
        ],
        "expected_source_docs": None,
        "notes": "Doc 3 mentions SME concerns, Doc 1 mentions $50-200M costs, Doc 4 has documentation burden"
    },

    # ─────────────────────────────────────────────────────────────────────
    # OUT-OF-SCOPE QUESTIONS (5) — Topic not in any of the 4 documents
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "O1",
        "query": "How do I fine-tune a large language model using LoRA?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Technical ML training question — not covered in any regulation document"
    },
    {
        "id": "O2",
        "query": "What is the current stock price of Microsoft?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Financial market data — completely unrelated"
    },
    {
        "id": "O3",
        "query": "What AI regulations exist in Brazil or India?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Documents only cover EU, US, China, UK — Brazil/India not mentioned anywhere"
    },
    {
        "id": "O4",
        "query": "What are the environmental impacts of training large AI models?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Environmental angle not covered in any of the 4 documents"
    },
    {
        "id": "O5",
        "query": "Which programming languages are best for building AI applications?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Software engineering advice — completely outside scope of regulation documents"
    },
]
