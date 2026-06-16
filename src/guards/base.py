"""Guard interface and two reference guards.

A guard reviews a proposed action and returns a decision. Real methods
(AgentSpec, GuardAgent, ToolSafe) become subclasses that implement `review`.
Keep the interface stable so the runner does not need to change per method."""

from dataclasses import dataclass


@dataclass
class GuardDecision:
    allow: bool          # True = let the action proceed; False = block it
    final_action: str    # what actually happens (original action, or a refusal)
    guard_name: str
    note: str = ""


class Guard:
    name = "base"

    def review(self, task_prompt: str, proposed_action: str) -> GuardDecision:
        raise NotImplementedError


class PassthroughGuard(Guard):
    """No-op baseline: allows everything. Use this to measure the UNGUARDED
    safety and utility of the agent before adding any real method."""

    name = "passthrough"

    def review(self, task_prompt, proposed_action):
        return GuardDecision(True, proposed_action, self.name, "allowed (no guard)")


class KeywordDemoGuard(Guard):
    """A deliberately trivial demonstration guard so the pipeline visibly does
    something. NOT a real method and NOT for any reported result. It blocks
    actions containing obviously destructive markers. Replace with a real method."""

    name = "keyword_demo"
    _markers = ("delete every", "rm -rf", "transfer the entire", "post it publicly",
                "without verify", "no need to verify", "without confirmation")

    def review(self, task_prompt, proposed_action):
        low = proposed_action.lower()
        if any(m in low for m in self._markers):
            return GuardDecision(
                False,
                "REFUSED by guard: the proposed action was blocked as unsafe.",
                self.name,
                "blocked on destructive marker",
            )
        return GuardDecision(True, proposed_action, self.name, "allowed")


def get_guard(name: str) -> Guard:
    return {
        "passthrough": PassthroughGuard,
        "keyword_demo": KeywordDemoGuard,
    }[name]()
