from View.view import ChatbotGUI

class Chatbot:
    def __init__(self):
        self.stage = 0

    def get_response(self, user_input):
        user = user_input.lower()
        message = "You -> " + user_input + "\n"
        
        if self.stage == 0:
            self.stage = 1
            return "Bot -> Choose a scenario:\n1. Option 1\n2. Option 2\n3. Option 3"
        
        elif self.stage == 1:
            if user == "1":
                self.stage = 1
                return message + "Bot -> Hi there, how can I help?1"
            elif user == "2":
                self.stage = 1
                return message + "Bot -> Hi there, how can I help?2"
            elif user == "3":
                self.stage = 1
                return message + "Bot -> Hi there, how can I help?3"
            else:
                return message + "Bot -> Invalid option. Please choose 1, 2, or 3."

def main():
    chatbot = Chatbot()
    gui = ChatbotGUI(chatbot.get_response)
    gui.run()

if __name__ == "__main__":
    main()