# from tkinter import *
# # StreamLit
# class ChatbotGUI:
#     def __init__(self, response_function):
#         self.response_function = response_function
#         self.root = Tk()
#         self.root.title("Chatbot")

#         self.BG_GRAY = "#ABB2B9"
#         self.BG_COLOR = "#17202A"
#         self.TEXT_COLOR = "#EAECEE"

#         self.FONT = "Helvetica 14"
#         self.FONT_BOLD = "Helvetica 13 bold"

#         self.setup_gui()
#         self.display_initial_message()

#     def setup_gui(self):
#         self.txt = Text(self.root, bg=self.BG_COLOR, fg=self.TEXT_COLOR, font=self.FONT, width=60)
#         self.txt.grid(row=1, column=0, columnspan=2)

#         scrollbar = Scrollbar(self.txt)
#         scrollbar.place(relheight=1, relx=0.974)

#         self.e = Entry(self.root, bg="#2C3E50", fg=self.TEXT_COLOR, font=self.FONT, width=55)
#         self.e.grid(row=2, column=0)
#         self.e.bind("<Return>", self.on_enter_pressed)

#         send_button = Button(self.root, text="Send", font=self.FONT_BOLD, bg=self.BG_GRAY, command=self.send_message)
#         send_button.grid(row=2, column=1)

#     def display_initial_message(self):
#         initial_message = self.response_function("")
#         self.txt.insert(END, initial_message + "\n")
#         self.txt.see(END)

#     def send_message(self):
#         user_input = self.e.get()

#         response = self.response_function(user_input)
#         self.txt.insert(END, "\n" + response)
#         self.txt.see(END)

#         self.e.delete(0, END)

#     def on_enter_pressed(self, event):
#         self.send_message()

#     def run(self):
#         self.root.mainloop()

import streamlit as st
import time

class ChatbotGUI:
    def __init__(self, title="Chatbot"):
        # Initialize the Streamlit app
        st.title(title)
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize the session state for chat history."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        

    def display_chat_history(self):
        """Display the chat history from the session state."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self):
        """Handle user input and generate a response."""
        if prompt := st.chat_input("What is up?"):
            # Display user message
            self.add_message("user", prompt)
            # Generate a response
            response = self.generate_response(prompt)
            # Display assistant response
            self.add_message("assistant", response)

    def add_message(self, role, content):
        """Add a message to the chat history and display it."""
        with st.chat_message(role):
            st.markdown(content)
        st.session_state.messages.append({"role": role, "content": content})


    def generate_response(self, user_input):
        """Generate a response for the user input."""
        # Add a progress bar to simulate thinking
        progress_bar = st.progress(0)
        status_message = st.empty()  # Placeholder for status text

        for percent_complete in range(1, 101):
            time.sleep(0.1)  # Simulate progress
            progress_bar.progress(percent_complete)

            # Display "almost there" when progress reaches 50%
            if percent_complete == 50:
                status_message.text("Almost there...")

        time.sleep(0.5)  # Simulate progress
        # Remove the progress bar and status message
        progress_bar.empty()
        status_message.empty()

        # Generate the actual response
        response = f"{user_input}"
        return response

    def run(self):
        """Run the chatbot application."""
        self.display_chat_history()
        self.handle_user_input()


# Instantiate and run the chatbot
if __name__ == "__main__":
    chatbot = ChatbotGUI()
    chatbot.run()