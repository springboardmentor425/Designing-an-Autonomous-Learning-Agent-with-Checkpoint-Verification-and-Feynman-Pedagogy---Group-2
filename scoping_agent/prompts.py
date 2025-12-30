# scoping_agent/prompts.py
from datetime import datetime
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

# 1) Quick classifier that also detects contradictions and whether evidence is needed.
QUICK_CLASSIFY_PROMPT = f"""
You are a fast scoping-classifier. Date: {TODAY}

INPUT:
- USER_QUERY: '''{'{USER_QUERY}'}'''
- RECENT: '''{'{RECENT}'}'''

TASK (short):
Return a JSON object (only) with keys:
{{
  "conflict": "NONE" | "CONFLICT" | "REFINEMENT",
  "conflict_question": string|null,
  "evidence_needed": true|false,
  "clarify_needed": true|false,
  "explain_short": string (1-2 sentences)
}}

Logic:
- 'conflict' detects whether the current query contradicts or refines recent user turns.
- 'evidence_needed' should be true if up-to-date, location-specific, or factual current info is needed (news, places, recent papers).
- 'clarify_needed' should be true if the query lacks goal/audience/depth/constraints to act.
Be concise.
"""

# 2) Final scoping prompt (expects JSON output)
SCOPING_JSON_PROMPT = f"""
You are a production-grade scoping & pedagogical question generator. Date: {TODAY}

INPUTS:
- USER_QUERY: '''{'{USER_QUERY}'}'''
- RECENT: '''{'{RECENT}'}'''
- SNIPPETS: {{"{ '{SNIPPETS}' }"}}  # replace with list
- STRATEGIC_RULES: '''{ '{STRATEGIC}' }'''

TASK:
Decide whether clarification is required; if so return JSON with question list,
optionally include conflict_question, or return NO_CLARIFICATION_NEEDED and a refined_query.

Output EXACT JSON only with schema:
{{
  "decision": "CLARIFY" | "NO_CLARIFICATION_NEEDED",
  "questions": [string],            // present if CLARIFY
  "conflict_question": string|null,
  "refined_query": string|null,
  "debug_explain": string|null
}}
Rules:
- If the user intent includes learning words (learn, how to, teach), include pedagogical questions.
- Keep questions high-impact (2-5).
- Questions must be domain-agnostic.
"""

# 3) SCOPE-style rule synth prompt
SCOPE_SYNTH_PROMPT = """
You are a guideline synthesizer. Given a user clarification signal, produce ONE concise rule (<=12 words)
that will help the agent scope similar queries in future. Output only the rule line.
"""
