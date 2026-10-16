from typing import List



from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from agentic_ai_platform.data_class.tool_spec import ToolSpec
from agentic_ai_platform.states.tool_state import ToolState
from agentic_ai_platform.llm.llm import LLM


class EligibleTools(BaseModel):
    tool_names: List[str] = Field(description="Names of tools that can run this turn")


async def get_current_eligible_tools(llm:LLM,
                               tool_specs: List[ToolSpec],
                               tool_states: List[ToolState]) -> List[ToolSpec]:
    """
    Looking for tools that need to be run excluding the tools already succssfully ran or tried to 
    maximum times but failed.

    - succeeded_tools : listing of the tools already ran successfullly 
    - failed_tools : listing of tools of all failing up to given attempt trials

    Return 
    - Among passed tools as parameters, give the list of tools can run at this turn.

    """
    
    human_prompt = "\n".join(f"{spec.name} should run after {"".join(f"{required}," for required in spec.requires)}" for spec in tool_specs if spec.requires)

    prompt = ChatPromptTemplate.from_messages([("system", "Iterate a tool list. It should determine it taking account of a required tool"),
                                                ("user","{tool_info}")]).format_messages(tool_info=human_prompt)

    result = await llm.invoke_with_structured_llm(schema=EligibleTools, 
                                            prompt=prompt)

    

    eligible_tool_list : List[ToolSpec] = []
    for tool_name in result.tool_names:
        tool_spec = next(t for t in tool_specs if t.name == tool_name)
        eligible_tool_list.append(tool_spec)

    # succeeded_tools = {ts.tool_name for ts in tool_states if ts.status == "success"}

    # failed_tools = {ts.tool_name for ts in tool_states 
    #                 if ts.status == "failed" and ts.attempt >=  next(
    #                     (s.max_attempts for s in tool_specs if s.name == ts.tool_name), 1
    #                 )}

    # eligible_tool_list : List = []

    # for spec in tool_specs:
    #     if spec.name in succeeded_tools or spec.name in failed_tools:
    #         continue

    #     if not all(r in succeeded_tools for r in spec.requires):
    #         continue

    #     eligible_tool_list.append(spec)

    return eligible_tool_list

def  next_attempt_number(tool_name:str,
                            tool_states : List[ToolState]):
    """
    """
    prior = [ts.attempt for ts in tool_states if ts.tool_name == tool_name]
    return (max(prior) + 1) if prior else 1