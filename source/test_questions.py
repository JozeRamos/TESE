
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
        
        
    # ------------------------------------------------------------------------------------------------------------

    def test_is_question_with_question(self):
        start_time = time.time()
        self.assertTrue(self.controller.is_question("Is this a question?"))
        end_time = time.time()
        print(f"test_is_question_with_question took {end_time - start_time:.4f} seconds")

    def test_is_question_with_statement(self):
        start_time = time.time()
        self.assertFalse(self.controller.is_question("Ask the user for the number of meals"))
        end_time = time.time()
        print(f"test_is_question_with_statement took {end_time - start_time:.4f} seconds")

    def test_is_question_with_empty_string(self):
        self.assertFalse(self.controller.is_question(""))

    def test_is_question_with_whitespace(self):
        self.assertFalse(self.controller.is_question("   "))

    def test_is_question_with_question_mark_in_middle(self):
        self.assertTrue(self.controller.is_question("Is this a question? Yes, it."))

    def test_is_question_with_special_characters(self):
        self.assertTrue(self.controller.is_question("What is this?!"))

# ------------------------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------------------------

    def test_check_optional_false(self):
        start_time = time.time()
        self.assertFalse(self.controller.check_optional("Ask the user for the number of meals")[0])
        end_time = time.time()
        print(f"test_check_optional_false took {end_time - start_time:.4f} seconds")
        
    def test_check_optional_true(self):
        start_time = time.time()
        self.assertTrue(self.controller.check_optional("What do I need to do?")[0])
        end_time = time.time()
        print(f"test_check_optional_true took {end_time - start_time:.4f} seconds")

# ------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()