from View.view import ChatbotGUI
from Controller.controller import LLM

class Chatbot:
    def __init__(self):
        self.stage = "Initial"
        self.llm = None

    def get_response(self, user_input):
        user = user_input.lower()
        
        if self.stage == "Initial":
            self.stage = "Options"
            return "Bot -> Choose a scenario:\n1. Loop Scenario 1\n2. Create scenario 2\n"
        
        elif self.stage == "Options":
            message = "You -> " + user_input + "\n"
            if user == "1":
                self.stage = "loop"
                self.llm = LLM('Scenarios\loop.json')
                
                return message + "Bot -> Loop scenario selected. Here is the scenario description:\n" + self.llm.get_stage_description()
            elif user == "2":
                self.stage = "create"
                return message + "Bot -> Scenario creation selected."
            else:
                return message + "Bot -> Invalid option. Please choose 1 or 2."
            
        elif self.stage == "loop":
            if user:
                message = "Your -> " + user_input + "\nBot -> " + self.llm.logic(user_input)
                return message
            else:
                return "Bot -> Invalid command. Please enter 'hint', 'positive feedback', 'constructive feedback', or 'next stage'."

def main():
    chatbot = Chatbot()
    gui = ChatbotGUI(chatbot.get_response)
    gui.run()

if __name__ == "__main__":
    main()