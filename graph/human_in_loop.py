

class HITL:
    def __init__(self, *argc, prompt):
        self.human_input_function = argc
        self.prompt = prompt

    def get_human_input(self, prompt):
        
        return self.human_input_function(prompt)