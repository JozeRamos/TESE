import streamlit as st
import time
import os
import sys
os.environ["STREAMLIT_WATCH_FILE_CHANGES"] = "false"
sys.modules['torch.classes'] = None

class ChatbotGUI:
    def __init__(self, logic):
        self.logic = logic  # Assign the response function
        st.title("Chatbot")
        self.initialize_session_state(logic.get_response)
        self.progress_bar = None
        self.status_message = None

    def initialize_session_state(self, response_function):
        """Initialize the session state for chat history."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Call the response_function to get the initial message
            initial_message = response_function("")
            st.session_state.messages.append({
                "role": "assistant",
                "content": initial_message
            })

    def display_chat_history(self):
        if "messages" not in st.session_state:
            self.logic.reset()
            self.initialize_session_state(self.logic.get_response)
            self.progress_bar = None
            self.status_message = None

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
        """Generate a response using the model's get_response method."""
        return self.logic.get_response(user_input)

    def progress_bar_create(self):
        """Add a progress bar to simulate thinking"""
        self.progress_bar = st.progress(0)
        self.status_message = st.empty()

    def progress_bar_delete(self):
        """Delete the progress bar and status message."""
        if self.progress_bar is not None:
            self.progress_bar.empty()
            self.status_message.empty()

    def progress_bar_percentage(self, percent_before, percent_after, message):
        """Simulate progress bar percentage completion."""
        for percent_complete in range(percent_before, percent_after):
            time.sleep(0.01)  # Simulate progress
            self.progress_bar.progress(percent_complete)
            self.status_message.text(message)

            

    def run(self):
        """Run the chatbot application."""
        self.display_chat_history()
        self.handle_user_input()