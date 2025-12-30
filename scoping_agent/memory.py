# scoping_agent/memory.py
from typing import List, Dict

class ScopingMemory:
    def __init__(self):
        self.turns: List[str] = []
        self.strategic_rules: List[str] = []
        self.tavily_cache: Dict[str, List[str]] = {}

    def add_turn(self, text: str):
        self.turns.append(text)

    def recent(self, n=6) -> str:
        return " | ".join(self.turns[-n:]) if self.turns else ""

    def add_rule(self, rule: str):
        rule = rule.strip()
        if rule and rule not in self.strategic_rules:
            self.strategic_rules.append(rule)

    def get_rules_text(self) -> str:
        return " ; ".join(self.strategic_rules)

    def cache_snippets(self, key: str, snippets: List[str]):
        self.tavily_cache[key] = snippets

    def get_cached_snippets(self, key: str):
        return self.tavily_cache.get(key)
