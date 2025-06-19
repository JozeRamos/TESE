import unittest
from Controller.controller import LLM
import os
import time

class TestController(unittest.TestCase):
    def setUp(self):
        # Initialize the controller before each test
        self.controller = LLM(os.path.join('source', 'Scenarios', 'calory_tracker.json'))

        self.temp_text = self.controller.prepare_conversation_history()

        self.user_input = "What do i need to do?"

        # self.controller.build_index()
        

    def test_single(self):
        print(self.controller.is_important(self.user_input, True))
        print(self.controller.is_important(self.user_input, False))


    # def test_LLM_calls(self):
    #     docs = ""
    #     cot_answer = self.controller.generate_cot_response(self.user_input, False, self.temp_text)

    #     print("Is Important:", self.controller.is_important(self.user_input, False))

    #     self.controller.check_optional(self.user_input)

    #     self.controller.redirect_user(self.user_input, self.temp_text)

    #     middle_prompt = self.controller.self_consistency(self.user_input, cot_answer, 3, self.temp_text)
    #     print("Middle Prompt:", middle_prompt)

    #     feedback = self.controller.feedback(self.user_input, middle_prompt, self.temp_text, docs)
    #     print("Feedback:", feedback)

    #     final_prompt = self.controller.refine(middle_prompt, self.user_input, feedback, self.temp_text)
    #     print("Final Prompt (after refine):", final_prompt)

    #     next_steps = self.controller.next_steps(self.user_input, final_prompt, self.temp_text)
    #     print("Next Steps:", next_steps)

    #     final_prompt = final_prompt + "\n\nNext Steps: " + next_steps
    #     print("Final Prompt (with Next Steps):", final_prompt)



if __name__ == "__main__":
    unittest.main()