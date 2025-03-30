from tkinter import *

class ChatbotGUI:
    def __init__(self, response_function):
        self.response_function = response_function
        self.root = Tk()
        self.root.title("Chatbot")

        self.BG_GRAY = "#ABB2B9"
        self.BG_COLOR = "#17202A"
        self.TEXT_COLOR = "#EAECEE"

        self.FONT = "Helvetica 14"
        self.FONT_BOLD = "Helvetica 13 bold"

        self.setup_gui()
        self.display_initial_message()

    def setup_gui(self):
        self.txt = Text(self.root, bg=self.BG_COLOR, fg=self.TEXT_COLOR, font=self.FONT, width=60)
        self.txt.grid(row=1, column=0, columnspan=2)

        scrollbar = Scrollbar(self.txt)
        scrollbar.place(relheight=1, relx=0.974)

        self.e = Entry(self.root, bg="#2C3E50", fg=self.TEXT_COLOR, font=self.FONT, width=55)
        self.e.grid(row=2, column=0)
        self.e.bind("<Return>", self.on_enter_pressed)

        send_button = Button(self.root, text="Send", font=self.FONT_BOLD, bg=self.BG_GRAY, command=self.send_message)
        send_button.grid(row=2, column=1)

    def display_initial_message(self):
        initial_message = self.response_function("")
        self.txt.insert(END, initial_message + "\n")
        self.txt.see(END)  # Auto-scroll to the end

    def send_message(self):
        user_input = self.e.get()

        response = self.response_function(user_input)
        self.txt.insert(END, "\n" + response)
        self.txt.see(END)  # Auto-scroll to the end

        self.e.delete(0, END)

    def on_enter_pressed(self, event):
        self.send_message()

    def run(self):
        self.root.mainloop()