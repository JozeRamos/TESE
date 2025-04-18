from View.view import ChatbotGUI
from Controller.controller import LLM

_chatbot_instance = None

class Chatbot:
    def __init__(self):
        self.stage = "Initial"
        self.llm = None
        self.view = None
    
    def set_view(self, view):
        self.view = view

    def set_llm(self, llm):
        self.llm = LLM('source\\Scenarios\\loop.json')

    def get_response(self, user_input):
        user = user_input.lower()

        if self.stage == "Initial":
            self.stage = "Options"
            return "Choose a scenario:\n1. Loop Scenario\n2. Create scenario\n"
        
        elif self.stage == "Options":
            if user == "1":
                self.stage = "loop"
                self.view.progress_bar_create()

                self.view.progress_bar_percentage(1, 95, "Loading loop scenario...")

                self.set_llm('source\\Scenarios\\loop.json')

                self.view.progress_bar_percentage(95, 101, "Finishing up...")
                self.view.progress_bar_delete()
                
                return "Loop scenario selected. Here is the scenario description:\n" + self.llm.get_stage_description()
            elif user == "2":
                return "Scenario creation selected."
            else:
                return "Invalid option. Please choose from the options provided.\n1. Loop Scenario\n2. Create scenario\n"
            
        elif self.stage == "loop":
            if user:
                self.view.progress_bar_create()
                message = self.llm.logic(user_input, self.view.progress_bar_percentage)
                self.view.progress_bar_delete()
                return message
            else:
                return "Bot -> Invalid command. Please enter 'hint', 'positive feedback', 'constructive feedback', or 'next stage'."
            

def main():
    global _chatbot_instance
    # Modify the code to create the Chatbot instance only once
    if _chatbot_instance is None:
        _chatbot_instance = Chatbot()

    chatbot = _chatbot_instance
    gui = ChatbotGUI(chatbot)
    if chatbot.view is None:
        chatbot.set_view(gui)
    gui.run()

if __name__ == "__main__":
    main()