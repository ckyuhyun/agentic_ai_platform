

from langchain.messages import ToolMessage
import ast
import json

from langchain_core.prompts import ChatPromptTemplate

from track_issue_system.State.filter_message_state import FilterMessageState

CASUAL_GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "yo"]
CASUAL_THANKS = ["thanks", "thank you", "thx", "awesome thanks", "perfect thank you"]
TEAMMEMBER_COMMON_COWORK = ["let's sync up", 
                            "let's discuss", 
                            "let's have a meeting", 
                            "can we talk about this?", 
                            "let's collaborate on this", 
                            "let's work together on this"]

TEAMMEMBER_COMMON_FEEDBACK = ["can you take a look at this?", 
                              "what do you think about this?", 
                              "do you have any feedback on this?", 
                              "can you review this?", 
                              "let me know your thoughts on this", 
                              "I'd appreciate your input on this"]

def create_message_filter_agent(node_llm, 
                                system_prompt: str):


    def message_filter_agent(state):
        """
        This function filters out unnecessary messages from the tool states in the given state.
        It checks each tool state and removes those that are deemed unnecessary based on certain criteria.
        The criteria for filtering can be defined as needed, such as removing tool states with empty results or those that do not contribute to the final output.
        """
        
        messages = state.messages[-1]
        if isinstance(messages, ToolMessage):
            messages = messages.content

        
        #joined_str = "\n".join(f"{i} : {m}" for i, m in enumerate(messages))

        prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        filtered_message =[]
        for i, m in enumerate(messages):
            prompt = prompt_template.format_messages(input=m)
            prompt[0].additional_kwargs["cache_control"] = {"type", "ephemeral"}

            #output = node_llm.llm_instance.invoke(prompt)
            output = node_llm.llm_instance.with_structured_output(FilterMessageState).invoke(prompt)
            filtered_message.append(output)
        
      
        black_list = CASUAL_GREETINGS + CASUAL_THANKS + TEAMMEMBER_COMMON_COWORK + TEAMMEMBER_COMMON_FEEDBACK

        messages = state.messages
        final_cleaned_messages :str = ""
        for msg in messages:
            if isinstance(msg, ToolMessage):
                if msg.content and "get_slack_message" in msg.name:
                    # try:
                    #     list_messages = ast.literal_eval(msg.content)
                    # except (ValueError, SyntaxError) as e:
                    #     raise ValueError(f"Invalid content in message: {msg.content}, error: {e}")

                    cleaned_messages : list[str]= []
                    for message in msg.content:
                        if not message in black_list:
                            cleaned_messages.append(message)

                    final_cleaned_messages = json.dumps(cleaned_messages)
                            
                    
        return state.model_copy(update={"messages": final_cleaned_messages})
    
    return message_filter_agent