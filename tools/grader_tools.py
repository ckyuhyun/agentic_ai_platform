import re
from langchain_core.tools import tool


class GraderTools:

    @staticmethod
    @tool
    def check_constraints(content: str) -> str:
        """Check whether the draft satisfies the explicit constraints of the task.

        Args:
            content: A string in the form "TASK: <task text>\\nDRAFT: <draft text>"
        Returns:
            A plain-text report of which constraints are met or violated.
        """
        task_match  = re.search(r"TASK:\s*(.*?)(?=\nDRAFT:|\Z)", content, re.DOTALL | re.IGNORECASE)
        draft_match = re.search(r"DRAFT:\s*(.*)",                  content, re.DOTALL | re.IGNORECASE)

        task  = task_match.group(1).strip()  if task_match  else content
        draft = draft_match.group(1).strip() if draft_match else content

        findings = []

        # Length: a very short draft rarely satisfies non-trivial tasks
        word_count = len(draft.split())
        if word_count < 20:
            findings.append(f"FAIL  draft is too short ({word_count} words) to address the task.")
        else:
            findings.append(f"PASS  draft length is acceptable ({word_count} words).")

        # Keyword coverage: key nouns from the task should appear in the draft
        task_keywords = {w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", task)}
        draft_lower   = draft.lower()
        missing       = [kw for kw in task_keywords if kw not in draft_lower]
        coverage      = 1 - len(missing) / max(len(task_keywords), 1)
        if coverage >= 0.6:
            findings.append(f"PASS  keyword coverage {coverage:.0%} — draft addresses the task topics.")
        else:
            findings.append(
                f"FAIL  keyword coverage only {coverage:.0%}. "
                f"Missing task terms: {', '.join(sorted(missing)[:10])}"
            )

        # Completeness: draft should not end mid-sentence
        if draft and draft[-1] not in ".!?\"'":
            findings.append("WARN  draft appears to end abruptly (no terminal punctuation).")

        return "\n".join(findings)

    @staticmethod
    @tool
    def check_hallucinations(draft: str) -> str:
        """Scan the draft for common hallucination signals: unsupported statistics,
        vague authority appeals, and fabricated references.

        Args:
            draft: The raw draft text to evaluate.
        Returns:
            A plain-text report of potential hallucination risks.
        """
        findings = []

        # Unsupported statistics / percentages
        stat_pattern = re.compile(r"\b\d+(\.\d+)?\s*(%|percent|times|x)\b", re.IGNORECASE)
        stats = stat_pattern.findall(draft)
        if stats:
            findings.append(
                f"WARN  {len(stats)} statistic(s) / numeric claim(s) detected — verify each has a cited source."
            )
        else:
            findings.append("PASS  no bare statistics found.")

        # Vague authority appeals
        authority_patterns = [
            r"\bstudies show\b", r"\bresearch (shows|suggests|indicates)\b",
            r"\bexperts (say|believe|recommend)\b", r"\baccording to (experts|researchers)\b",
            r"\bit is (well[- ]known|widely accepted)\b",
        ]
        authority_hits = [p for p in authority_patterns if re.search(p, draft, re.IGNORECASE)]
        if authority_hits:
            findings.append(
                f"WARN  {len(authority_hits)} vague authority appeal(s) detected "
                f"(e.g. 'studies show', 'experts say') — cite specific sources."
            )
        else:
            findings.append("PASS  no vague authority appeals detected.")

        # Fabricated citations: [Author, Year] style without DOI / URL
        citation_pattern = re.compile(r"\[([A-Z][a-z]+(?:\s+et al\.?)?,\s*\d{4})\]")
        citations = citation_pattern.findall(draft)
        if citations:
            findings.append(
                f"WARN  {len(citations)} inline citation(s) found — confirm they are real publications: "
                + ", ".join(citations[:5])
            )

        # Absolute certainty language
        certainty_patterns = [r"\balways\b", r"\bnever\b", r"\bguaranteed\b", r"\bimpossible\b"]
        certainty_hits = [p for p in certainty_patterns if re.search(p, draft, re.IGNORECASE)]
        if certainty_hits:
            findings.append(
                f"WARN  absolute-certainty language detected ({', '.join(certainty_hits)}) "
                "— consider hedging or providing evidence."
            )
        else:
            findings.append("PASS  no absolute-certainty language.")

        return "\n".join(findings)

    @staticmethod
    @tool
    def check_efficiency(draft: str) -> str:
        """Evaluate whether the draft communicates ideas concisely without redundancy.

        Args:
            draft: The raw draft text to evaluate.
        Returns:
            A plain-text efficiency report.
        """
        findings = []

        sentences = re.split(r"(?<=[.!?])\s+", draft.strip())
        words     = draft.split()
        word_count    = len(words)
        sent_count    = max(len(sentences), 1)
        avg_sent_len  = word_count / sent_count

        findings.append(f"INFO  {word_count} words across {sent_count} sentence(s) "
                        f"(avg {avg_sent_len:.1f} words/sentence).")

        if avg_sent_len > 35:
            findings.append("FAIL  sentences are very long (>35 words avg) — consider splitting for clarity.")
        elif avg_sent_len > 25:
            findings.append("WARN  sentences are somewhat long (>25 words avg).")
        else:
            findings.append("PASS  sentence length is comfortable.")

        # Filler / padding phrases
        fillers = [
            r"\bit is (important|worth) (to note|noting) that\b",
            r"\bin (conclusion|summary), (it is|we can) (clear|see) that\b",
            r"\bfirstly,?\s+it should be noted\b",
            r"\bas (previously|already) mentioned\b",
            r"\bwithout further ado\b",
            r"\bto be (honest|frank)\b",
        ]
        filler_hits = [p for p in fillers if re.search(p, draft, re.IGNORECASE)]
        if filler_hits:
            findings.append(f"WARN  {len(filler_hits)} filler phrase(s) detected — remove for tighter prose.")
        else:
            findings.append("PASS  no common filler phrases.")

        # Repeated consecutive words (e.g. "the the")
        repeated = re.findall(r"\b(\w+)\s+\1\b", draft, re.IGNORECASE)
        if repeated:
            findings.append(f"FAIL  repeated word(s) detected: {set(repeated)} — likely a typo.")

        return "\n".join(findings)

    @staticmethod
    @tool
    def check_ethical_considerations(draft: str) -> str:
        """Flag potential ethical, legal, or bias issues in the draft.

        Args:
            draft: The raw draft text to evaluate.
        Returns:
            A plain-text ethics report.
        """
        findings = []

        # Demographic / discriminatory language
        bias_patterns = [
            (r"\b(all|every)\s+(women|men|blacks|whites|asians|muslims|christians)\b",
             "broad demographic generalisation"),
            (r"\b(inferior|superior)\s+(race|gender|culture)\b",
             "discriminatory language"),
            (r"\b(illegal alien|anchor baby)\b",
             "charged immigration language"),
        ]
        bias_hits = []
        for pattern, label in bias_patterns:
            if re.search(pattern, draft, re.IGNORECASE):
                bias_hits.append(label)
        if bias_hits:
            findings.append(f"FAIL  potential bias / discriminatory language: {'; '.join(bias_hits)}.")
        else:
            findings.append("PASS  no obvious demographic bias patterns.")

        # Financial / medical / legal disclaimer check
        professional_domains = {
            "financial":  r"\b(invest|stock|portfolio|return|dividend|buy|sell)\b",
            "medical":    r"\b(diagnos|treatment|medication|symptom|disease|cure)\b",
            "legal":      r"\b(lawsuit|liable|contract|regulation|compliance|attorney)\b",
        }
        domain_hits = [domain for domain, pattern in professional_domains.items()
                       if re.search(pattern, draft, re.IGNORECASE)]
        disclaimer_present = bool(re.search(
            r"\b(not (financial|medical|legal) advice|consult (a|your|an) (professional|doctor|lawyer|advisor))\b",
            draft, re.IGNORECASE,
        ))
        if domain_hits and not disclaimer_present:
            findings.append(
                f"WARN  draft touches {', '.join(domain_hits)} topics but contains no disclaimer — "
                "consider adding 'This is not professional advice.'"
            )
        elif domain_hits and disclaimer_present:
            findings.append(f"PASS  disclaimer present for {', '.join(domain_hits)} content.")
        else:
            findings.append("PASS  no sensitive professional-domain content detected.")

        # Privacy / PII signals
        pii_patterns = [
            (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",         "potential SSN"),
            (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "email address"),
            (r"\b\d{16}\b",                                   "potential credit card number"),
        ]
        pii_hits = [label for pattern, label in pii_patterns if re.search(pattern, draft)]
        if pii_hits:
            findings.append(f"FAIL  possible PII in draft: {', '.join(pii_hits)} — review before publishing.")
        else:
            findings.append("PASS  no PII patterns detected.")

        return "\n".join(findings)
