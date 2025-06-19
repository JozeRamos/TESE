import unittest
from Controller.controller import LLM
import os
import time

class TestController(unittest.TestCase):
    def setUp(self):
        # Initialize the controller before each test
        self.controller = LLM(os.path.join('source', 'Scenarios', 'calory_tracker.json'))

        self.temp_text = self.controller.prepare_conversation_history()

        self.user_input = "Ask the user for the number of meals"
        self.output_file = "source/tests/test_results.txt"

    #tests for the first scenario in the calory tracker
    def test_Rag(self):
        with open(self.output_file, "w") as file:
            for i in range(5):
                start_time = time.time()
                self.controller.build_index()
                end_time = time.time()
                file.write(f"Run {i + 1}: test_RAG took {end_time - start_time:.4f} seconds\n")


    def test_LLM_calls(self):
        docs = ""
        for i in range(5):
            docs = self.controller.get_docs(self.user_input, 3)

        for i in range(5):
            cot_answer = self.controller.generate_cot_response(self.user_input,False, self.temp_text)

        for i in range(5):
            middle_prompt = self.controller.self_consistency(self.user_input,cot_answer, 3, self.temp_text)

        for i in range(5):
            feedback = self.controller.feedback(self.user_input, middle_prompt, self.temp_text, docs)
            
        for i in range(5):
            final_prompt = self.controller.refine(middle_prompt, self.user_input, feedback, self.temp_text)

        for i in range(5):
            self.controller.next_steps(self.user_input, final_prompt, self.temp_text)
            


if __name__ == "__main__":
    unittest.main()