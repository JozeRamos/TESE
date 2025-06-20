from View.view import ChatbotGUI
from Controller.controller import LLM
import os

_chatbot_instance = None
_gui_instance = None

class Chatbot:
    def __init__(self):
        self.stage = "Initial"
        self.llm = None
        self.view = None

    def reset(self):
        self.stage = "Initial"
        self.llm = None
        self.view = None

    def set_view(self, view):
        self.view = view

    def set_llm(self, llm):
        self.llm = LLM(llm)
        # self.view.progress_bar_percentage(40, 60, "Loading RAG...")
        # LLM.build_index(self.llm)

    def get_response(self, user_input):
        user = user_input.lower()

        if self.stage == "Initial":
            self.stage = "Options"
            return "Choose a scenario:\n1. Calory Tracker\n2. Bug Hunt\n3. Ride Sharing\n4. Medicine"
        
        elif self.stage == "Options":
            if user == "1":
                self.stage = "loop"
                self.view.progress_bar_create()

                self.view.progress_bar_percentage(1, 30, "Calory Tracker scenario...")

                self.set_llm(os.path.join('source', 'Scenarios', 'calory_tracker.json'))

                self.view.progress_bar_percentage(95, 101, "Finishing up...")
                self.view.progress_bar_delete()
                
                return "Calory Tracker scenario selected.\n\n Here is the scenario description:\n\n" + self.llm.get_stage_description()
            elif user == "2":
                self.stage = "loop"
                self.view.progress_bar_create()

                self.view.progress_bar_percentage(1, 30, "Bug Hunt scenario...")

                self.set_llm(os.path.join('source', 'Scenarios', 'bug_hunt.json'))

                self.view.progress_bar_percentage(95, 101, "Finishing up...")
                self.view.progress_bar_delete()
                
                return "Bug Hunt scenario selected.\n\n Here is the scenario description:\n\n" + self.llm.get_stage_description()
            elif user == "3":
                self.stage = "loop"
                self.view.progress_bar_create()

                self.view.progress_bar_percentage(1, 30, "Ride Sharing scenario...")

                self.set_llm(os.path.join('source', 'Scenarios', 'ride_sharing.json'))

                self.view.progress_bar_percentage(95, 101, "Finishing up...")
                self.view.progress_bar_delete()
                
                return "Ride Sharing scenario selected.\n\n Here is the scenario description:\n\n" + self.llm.get_stage_description()
            elif user == "4":
                self.stage = "loop"
                self.view.progress_bar_create()

                self.view.progress_bar_percentage(1, 30, "Medicine scenario...")

                self.set_llm(os.path.join('source', 'Scenarios', 'medicine.json'))

                self.view.progress_bar_percentage(95, 101, "Finishing up...")
                self.view.progress_bar_delete()
                
                return "Medicine scenario selected.\n\n Here is the scenario description:\n\n" + self.llm.get_stage_description()
            else:
                return "Invalid option. Please choose from the options provided.\n1. Loop Scenario"
            
        elif self.stage == "loop":
            if user == "quit":
                self.stage = "Options"
                return "Loop scenario aborted.\n\nChoose a new scenario:\n1. Loop Scenario"
            if user:
                self.stage = "loop"
                self.view.progress_bar_create()
                message = self.llm.logic(user_input, self.view.progress_bar_percentage)
                self.view.progress_bar_delete()
                if message == "End":
                    self.stage = "Options"
                    self.llm = None
                    return "Congratulations you completed the scenario.\n\nChoose a new scenario:\n1. Calory Tracker\n2. Bug Hunt\n3. Ride Sharing\n4. Medicine"
                else:
                    self.view.progress_bar_delete()
                    return message
            else:
                return "Invalid command."

def main():
    global _chatbot_instance, _gui_instance

    # Ensure the Chatbot instance is created only once
    if _chatbot_instance is None:
        print("Creating Chatbot instance...")
        _chatbot_instance = Chatbot()

    chatbot = _chatbot_instance

    # Ensure the ChatbotGUI instance is created only once
    if _gui_instance is None:
        print("Creating ChatbotGUI instance...")
        _gui_instance = ChatbotGUI(chatbot)

    gui = _gui_instance

    # Set the view for the Chatbot only if it hasn't been set
    if chatbot.view is None:
        chatbot.set_view(gui)

    # Run the GUI
    gui.run()

if __name__ == "__main__":
    main()