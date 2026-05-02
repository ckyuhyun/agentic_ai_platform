from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from agentic_ai_platform.eval.safety.hallucination_safeguide import HallucinationSignal
from agentic_ai_platform.state_manager.draft_state import DraftState
from agentic_ai_platform.llm.llm import LLM
from agentic_ai_platform.eval.safety.hallucination_safeguide import HallucinationsJudge
from typing_extensions import Annotated



class EvalsTools:

    @staticmethod
    @tool
    def check_constraints(content: str) -> str:
        """Check whether the draft satisfies the explicit constraints of the task.

        Args:
            content: A string in the form "TASK: <task text>\\nDRAFT: <draft text>"
        Returns:
            A plain-text report of which constraints are met or violated.
            
        """

        # check if there is Task and Draft section
        # if "TASK:" not in content or "DRAFT:" not in content:
        #     return "This content doesn't appear to be in the expected format with 'TASK:' and 'DRAFT:' sections. So need to be reviewed by human."
        return ""

       

    @staticmethod 
    @tool
    def check_hallucinations(draft:str, state:Annotated[DraftState, InjectedState]):        
        """
        Check the draft for potential hallucinations, which are pieces of information that may be fabricated, inaccurate, or not supported by evidence.
        """
        hallucination_system_prompt_version_id = '0'
        hallucination_llm = LLM("llama3.1").llm_instance

        hallucination_tool = HallucinationsJudge(
            schema=HallucinationSignal,
            hallucination_llm=hallucination_llm,
            hallucination_system_prompt_version_id=hallucination_system_prompt_version_id,
        )

        hallucination_tool_result = hallucination_tool(draft)

        return hallucination_tool_result
        

    @staticmethod
    @tool
    def check_efficiency(draft: str) -> str:
        """Evaluate whether the draft communicates ideas concisely without redundancy.

        Args:
            draft: The raw draft text to evaluate.
        Returns:
            A plain-text efficiency report.
        """
        # findings = []

        # sentences = re.split(r"(?<=[.!?])\s+", draft.strip())
        # words     = draft.split()
        # word_count    = len(words)
        # sent_count    = max(len(sentences), 1)
        # avg_sent_len  = word_count / sent_count

        # findings.append(f"INFO  {word_count} words across {sent_count} sentence(s) "
        #                 f"(avg {avg_sent_len:.1f} words/sentence).")

        # if avg_sent_len > 35:
        #     findings.append("FAIL  sentences are very long (>35 words avg) — consider splitting for clarity.")
        # elif avg_sent_len > 25:
        #     findings.append("WARN  sentences are somewhat long (>25 words avg).")
        # else:
        #     findings.append("PASS  sentence length is comfortable.")

        # # Filler / padding phrases
        # fillers = [
        #     r"\bit is (important|worth) (to note|noting) that\b",
        #     r"\bin (conclusion|summary), (it is|we can) (clear|see) that\b",
        #     r"\bfirstly,?\s+it should be noted\b",
        #     r"\bas (previously|already) mentioned\b",
        #     r"\bwithout further ado\b",
        #     r"\bto be (honest|frank)\b",
        # ]
        # filler_hits = [p for p in fillers if re.search(p, draft, re.IGNORECASE)]
        # if filler_hits:
        #     findings.append(f"WARN  {len(filler_hits)} filler phrase(s) detected — remove for tighter prose.")
        # else:
        #     findings.append("PASS  no common filler phrases.")

        # # Repeated consecutive words (e.g. "the the")
        # repeated = re.findall(r"\b(\w+)\s+\1\b", draft, re.IGNORECASE)
        # if repeated:
        #     findings.append(f"FAIL  repeated word(s) detected: {set(repeated)} — likely a typo.")

        # return "\n".join(findings)
        return ""

    @staticmethod
    @tool
    def check_ethical_considerations(draft: str) -> str:
        """Flag potential ethical, legal, or bias issues in the draft.

        Args:
            draft: The raw draft text to evaluate.
        Returns:
            A plain-text ethics report.
        """
        # findings = []

        # # Demographic / discriminatory language
        # bias_patterns = [
        #     (r"\b(all|every)\s+(women|men|blacks|whites|asians|muslims|christians)\b",
        #      "broad demographic generalisation"),
        #     (r"\b(inferior|superior)\s+(race|gender|culture)\b",
        #      "discriminatory language"),
        #     (r"\b(illegal alien|anchor baby)\b",
        #      "charged immigration language"),
        # ]
        # bias_hits = []
        # for pattern, label in bias_patterns:
        #     if re.search(pattern, draft, re.IGNORECASE):
        #         bias_hits.append(label)
        # if bias_hits:
        #     findings.append(f"FAIL  potential bias / discriminatory language: {'; '.join(bias_hits)}.")
        # else:
        #     findings.append("PASS  no obvious demographic bias patterns.")

        # # Financial / medical / legal disclaimer check
        # professional_domains = {
        #     "financial":  r"\b(invest|stock|portfolio|return|dividend|buy|sell)\b",
        #     "medical":    r"\b(diagnos|treatment|medication|symptom|disease|cure)\b",
        #     "legal":      r"\b(lawsuit|liable|contract|regulation|compliance|attorney)\b",
        # }
        # domain_hits = [domain for domain, pattern in professional_domains.items()
        #                if re.search(pattern, draft, re.IGNORECASE)]
        # disclaimer_present = bool(re.search(
        #     r"\b(not (financial|medical|legal) advice|consult (a|your|an) (professional|doctor|lawyer|advisor))\b",
        #     draft, re.IGNORECASE,
        # ))
        # if domain_hits and not disclaimer_present:
        #     findings.append(
        #         f"WARN  draft touches {', '.join(domain_hits)} topics but contains no disclaimer — "
        #         "consider adding 'This is not professional advice.'"
        #     )
        # elif domain_hits and disclaimer_present:
        #     findings.append(f"PASS  disclaimer present for {', '.join(domain_hits)} content.")
        # else:
        #     findings.append("PASS  no sensitive professional-domain content detected.")

        # # Privacy / PII signals
        # pii_patterns = [
        #     (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",         "potential SSN"),
        #     (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "email address"),
        #     (r"\b\d{16}\b",                                   "potential credit card number"),
        # ]
        # pii_hits = [label for pattern, label in pii_patterns if re.search(pattern, draft)]
        # if pii_hits:
        #     findings.append(f"FAIL  possible PII in draft: {', '.join(pii_hits)} — review before publishing.")
        # else:
        #     findings.append("PASS  no PII patterns detected.")

        # return "\n".join(findings)
        return ""
