from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Type
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_ai_platform.llm.llm import LLM
from agentic_ai_platform.state_manager.draft_state import DraftState
from agentic_ai_platform.state_manager.hallucination_signal import HallucinationSignal, SafetyJudge
 

from agentic_ai_platform import prompt_registery




def HallucinationsJudge(schema : Type[BaseModel],
                        hallucination_llm : LLM,
                        hallucination_system_prompt_version_id : str):
    
        
    def hallucination_safeguide(_draft: str) -> HallucinationSignal:
        logs = []
        prompt  = prompt_registery.get_prompts_by_type_version_tags(
            prompt_type="hallucination_checker",
            version_id=hallucination_system_prompt_version_id
        )
        structed_model = hallucination_llm.with_structured_output(schema)

        if prompt:
            final_prompt = [SystemMessage(content=prompt) , 
                            HumanMessage(content=_draft)]

            hallucination_safeguide_result = structed_model.invoke(final_prompt) 

            _pattern = hallucination_safeguide_result.pattern
            _severity = hallucination_safeguide_result.severity
            _excerpt = hallucination_safeguide_result.excerpt
        else:
            logs.append(f"No system prompt found for hallucination checker with version {hallucination_system_prompt_version_id}. Skipping hallucination check.")
            _pattern = ""
            _severity = "WARN"
            _excerpt = ""
        
        
        # Placeholder for the actual implementation of the hallucination check
        # In a real implementation, this would analyze the agent's output text
        # and return any detected hallucination signals.
        return HallucinationSignal(
            pattern=_pattern,
            severity=_severity,
            excerpt=_excerpt,
            HallucinationSignal_logs=logs
        )
        


    def confidence_safeguide():
        return None
        
        

    def safety_safeguide():
       return None

    
    def evaluate() -> SafetyJudge:
        hallucination_signal_result = hallucination_safeguide()
        confidence_safeguide()
        safety_safeguide()

        return SafetyJudge(score=safety_safeguide.score,
                           hallucination_signal_result =hallucination_signal_result)



    return hallucination_safeguide


    