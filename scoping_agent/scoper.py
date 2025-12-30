# scoping_agent/scoper.py
import json
from typing import Dict, Any, List, Optional
from .llm_tavily import GeminiWrapper, tavily_search_minimal
from .prompts import QUICK_CLASSIFY_PROMPT, SCOPING_JSON_PROMPT, SCOPE_SYNTH_PROMPT
from .memory import ScopingMemory

class Scoper:
    def __init__(self, llm: GeminiWrapper, memory: Optional[ScopingMemory] = None):
        self.llm = llm
        self.memory = memory or ScopingMemory()

    def _call_quick_classifier(self, user_query: str) -> Dict[str, Any]:
        prev = self.memory.recent()
        prompt = QUICK_CLASSIFY_PROMPT.replace("{USER_QUERY}", user_query).replace("{RECENT}", prev)
        raw = self.llm.generate(prompt)


        # try robust JSON parse
        try:
            # find JSON block
            s = raw.find("{")
            e = raw.rfind("}")
            js = raw[s:e+1] if s!=-1 and e!=-1 and e>s else raw
            return json.loads(js)
        except Exception:
            # fallback conservative decision: ask clarify and ask for evidence
            return {
                "conflict": "NONE",
                "conflict_question": None,
                "evidence_needed": True,
                "clarify_needed": True,
                "explain_short": "classifier parse failed; falling back to conservative behavior"
            }

    def scope(self, user_query: str) -> Dict[str, Any]:
        # 1) add to memory summary (not final until refined)
        prev_summary = self.memory.recent()
        # quick classifier LLM call (1 LLM call)
        cls = self._call_quick_classifier(user_query)

        # if classifier recommends evidence, fetch minimal tavily once
        snippets = []
        if cls.get("evidence_needed"):
            # check cache
            cached = self.memory.get_cached_snippets(user_query)
            if cached is not None:
                snippets = cached
            else:
                snippets = tavily_search_minimal(user_query, max_results=2)
                self.memory.cache_snippets(user_query, snippets)

        # If classifier says NO clarification and NO conflict, we can produce refined_query quickly
        if not cls.get("clarify_needed") and cls.get("conflict") == "NONE":
            # we still may want the final refined query; ask scoping prompt but only if snippets exist
            if snippets:
                # call final scoping LLM with snippets to get refined query (2nd LLM call)
                prompt = SCOPING_JSON_PROMPT.replace("{USER_QUERY}", user_query).replace("{RECENT}", prev_summary).replace("{STRATEGIC}", self.memory.get_rules_text())
                # add snippets into prompt text
                prompt = prompt.replace("{SNIPPETS}", json.dumps(snippets))
                raw = self.llm.generate(prompt, max_output_tokens=600)
                try:
                    s=raw.find("{"); e=raw.rfind("}")
                    data = json.loads(raw[s:e+1])
                    if data.get("decision","").upper() == "NO_CLARIFICATION_NEEDED":
                        refined = data.get("refined_query") or user_query
                        # store turn
                        self.memory.add_turn(user_query)
                        return {"need_clarify": False, "questions": [], "refined_query": refined, "debug": {"classifier": cls, "llm_raw": raw}}
                except Exception:
                    pass
            # no snippets or parse failed -> accept no-clarify
            self.memory.add_turn(user_query)
            return {"need_clarify": False, "questions": [], "refined_query": user_query, "debug": {"classifier": cls}}

        # If classifier says clarify_needed OR conflict detected -> produce final scoping questions (2nd LLM call)
        prompt = SCOPING_JSON_PROMPT.replace("{USER_QUERY}", user_query).replace("{RECENT}", prev_summary).replace("{STRATEGIC}", self.memory.get_rules_text())
        prompt = prompt.replace("{SNIPPETS}", json.dumps(snippets))
        raw = self.llm.generate(prompt, max_output_tokens=700)
        # parse JSON
        try:
            s=raw.find("{"); e=raw.rfind("}")
            data = json.loads(raw[s:e+1])
        except Exception as e:
            # fallback conservative questions
            return {"need_clarify": True, "questions": [
                "Please describe the specific goal and desired depth.",
                "Who is the audience and what is the intended outcome?"
            ], "refined_query": None, "debug": {"classifier": cls, "llm_raw": raw, "parse_error": str(e)}}

        # if conflict question set by classifier, prefer that first
        conflict_q = data.get("conflict_question")
        qs = data.get("questions", [])
        if conflict_q:
            qs = [conflict_q] + [q for q in qs if q != conflict_q]

        return {"need_clarify": data.get("decision","CLARIFY")=="CLARIFY", "questions": qs, "refined_query": data.get("refined_query"), "debug": {"classifier": cls, "llm_raw": raw}}

    def apply_user_answers(self, user_query: str, task_id: str, answers: Dict[str,str]) -> str:
        # combine answers into refined query
        parts = [f"Original: {user_query}"]
        for k,v in answers.items():
            parts.append(f"{k}:{v}")
        refined = " | ".join(parts)
        # Add refined as a turn
        self.memory.add_turn(f"REFINED[{task_id}]: {refined}")

        # SCOPE-style rule synthesis: 1 LLM call (optional; cheap)
        synth_prompt = SCOPE_SYNTH_PROMPT + "\n\nSignal:\n" + refined
        try:
            rule_raw = self.llm.generate(synth_prompt, max_output_tokens=80)
            rule = rule_raw.strip().splitlines()[0]
            if rule and len(rule) < 200:
                self.memory.add_rule(rule)
        except Exception:
            rule = None

        return refined
