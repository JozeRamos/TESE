import json
import os
import time
import pickle
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from semantic_router.encoders import HuggingFaceEncoder
from tqdm import tqdm
from groq import Client
from sentence_transformers import SentenceTransformer
import joblib
from concurrent.futures import ThreadPoolExecutor
import requests
from urllib.parse import urlencode, quote


class LLM:
    def __init__(self, json_file):
        # Load configuration
        with open(json_file, 'r') as file:
            data = json.load(file)
        self._load_config(data)
        PYTORCH_CUDA_ALLOC_CONF=True
        # Load API keys and clients
        self._load_api_keys()
        self.client = Client(api_key=os.getenv("GROQ_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self.base_url = "http://20.229.187.95:8000/api/9cbcfbcc-fd93-4b5b-a7f2-956f5c2d48ff/llm_completion"

        # Encoder
        try:
            self.encoder = HuggingFaceEncoder(
                name="sentence-transformers/all-MiniLM-L6-v2",
                device="cuda"  # <-- Forces GPU usage
            )
        except Exception as e:
            self.encoder = HuggingFaceEncoder(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Chat history
        self.chat_history = []

        self.llm_name = "llama3-70b-8192"
        self.provider = "groq"
        self.host = '20.229.187.95'
        self.port = '8000'
        self.token = '9cbcfbcc-fd93-4b5b-a7f2-956f5c2d48ff'

        # Just connect to the index if it exists
        self.index_name = "groq-llama-3-rag"
        self.index = None
        if self.index_name in [idx["name"] for idx in self.pc.list_indexes()]:
            self.index = self.pc.Index(self.index_name)

        initial = self.Inital_prompt()

        self.chat_history.append("Scenario description:\n" + initial + "\n")

    def _load_config(self, data):
        self.ai_role = data["ai_role"]
        self.user_role = data["user_role"]
        self.scenario_name = data["scenario_name"]
        self.ai_persona = data["ai_persona"]
        self.place = data["place"]
        self.task = data["task"]
        self.format = data["format"]
        self.exemplar = data["exemplar"]
        self.stage_description = data["stage_description"]
        self.hint = data["hint"]
        self.positive_feedback = data["positive_feedback"]
        self.constructive_feedback = data["constructive_feedback"]
        self.next_stage_condition = data["next_stage_condition"]
        self.stages = data["stages"]
        self.tones = data["tones"]
        self.stage_correct_response = []
        self.stage_correct_response_check = []
        self.stage_informations = []
        self.current_stage = 0
        self.optionals = data["all_optional"]
        for stage in self.stages:
            self.stage_informations.append(stage["stage_step"][0])
            self.stage_correct_response.append([step["correct_response"] for step in stage["stage_step"][1:]])
            self.stage_correct_response_check.append([False] * len(stage["stage_step"][1:]))
    
    def get_stage_description(self):
        return self.stage_description
    
    def _load_api_keys(self):
        api_file_path = os.path.join('source', 'API.txt')
        api2_file_path = os.path.join('source', 'API2.txt')

        with open(api_file_path, 'r') as file:
            os.environ["GROQ_API_KEY"] = file.read().strip()

        with open(api2_file_path, 'r') as file:
            os.environ["PINECONE_API_KEY"] = file.read().strip()


    def build_index(self):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Get the embedding dimension
        embedding_dimension = model.get_sentence_embedding_dimension()

        # File paths for caching
        processed_data_path = "processed_data.pkl"
        embeds_path = "embeds.pkl"

        # Step 1: Load and process dataset

        def load_file(file_path):
            with open(file_path, "rb") as f:
                return joblib.load(f)

        if os.path.exists(processed_data_path) and os.path.exists(embeds_path):
            print("Loading processed data and embeddings from cache...")

            # Load files in parallel
            with ThreadPoolExecutor() as executor:
                processed_data, embeds = executor.map(load_file, [processed_data_path, embeds_path])
        else:
            print("Processing dataset and computing embeddings...")
            data = load_dataset("open-phi/programming_books_llama", split="train[:10000]")

            # Handle null values and ensure consistent types in metadata
            def process_metadata(x, idx):
                def ensure_string(value):
                    if isinstance(value, list):
                        return ", ".join(map(str, value))  # Convert list to comma-separated string
                    return str(value) if value is not None else ""  # Convert None to empty string

                # Truncate metadata fields to reduce size
                def truncate(value, max_length=1000):
                    return value[:max_length] if len(value) > max_length else value

                return {
                    "id": str(idx),
                    "metadata": {
                        "topic": truncate(ensure_string(x["topic"])),
                        "queries": truncate(ensure_string(x["queries"])),
                        "context": truncate(ensure_string(x["context"])),
                    }
                }

            processed_data = data.map(process_metadata, with_indices=True)
            processed_data = processed_data.remove_columns([
                "topic", "context", "concepts", "queries", "outline", "model", "markdown"
            ])

            def truncate_chunk(text, max_length=1000):
                return text[:max_length] if len(text) > max_length else text

            chunks = [truncate_chunk(f'{x["topic"]}:\n{x["queries"]}\n{x["context"]}') for x in data]
            embeds = []
            for i in range(0, len(chunks), 32):  # or smaller
                batch = chunks[i:i+32]
                embeds.extend(model.encode(batch))

            # Save processed data and embeddings to cache
            with open(processed_data_path, "wb") as f:
                pickle.dump(processed_data, f)
            with open(embeds_path, "wb") as f:
                pickle.dump(embeds, f)
            print("Processed data and embeddings saved to cache.")

        # Step 2: Check if the index exists and matches the required configuration
        existing_indexes = self.pc.list_indexes()
        index_exists = any(idx["name"] == self.index_name for idx in existing_indexes)

        if index_exists:
            # Check if the dimension matches
            for idx in existing_indexes:
                if idx["name"] == self.index_name:
                    if idx["dimension"] == embedding_dimension:
                        print(f"Index '{self.index_name}' already exists with the correct dimension. Skipping recreation.")
                        break
                    else:
                        print(f"Index '{self.index_name}' exists but with a different dimension. Deleting it to reset...")
                        self.pc.delete_index(self.index_name)
                        # Wait for the index to be fully deleted
                        while self.index_name in [idx["name"] for idx in self.pc.list_indexes()]:
                            time.sleep(1)
                        break
        else:
            print(f"Index '{self.index_name}' does not exist. Creating it...")

        # Create the index if it doesn't exist or was deleted
        if not index_exists or idx["dimension"] != embedding_dimension:
            print(f"Creating index '{self.index_name}' with dimension {embedding_dimension}...")
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
            self.pc.create_index(
                self.index_name,
                dimension=embedding_dimension,
                metric='cosine',
                spec=spec
            )

        # Wait for the index to be ready
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
            self.index = self.pc.Index(self.index_name)

        # Step 3: Upsert data
        batch_size = 256
        for i in tqdm(range(0, len(processed_data), batch_size)):
            i_end = min(len(processed_data), i + batch_size)
            batch = processed_data[i:i_end]
            chunk_batch = embeds[i:i_end]
            to_upsert = list(zip(batch["id"], chunk_batch, batch["metadata"]))
            self.index.upsert(vectors=to_upsert)

    
    def get_docs(self, query: str, top_k: int) -> list[str]:
        # encode query
        xq = self.encoder([query])
        # search pinecone index
        res = self.index.query(vector=xq, top_k=top_k, include_metadata=True)
        # get doc text
        docs = [x["metadata"]['context'] for x in res["matches"]]
        # Format the documents into a readable string
        formatted_docs = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])
        return formatted_docs
    
    def Inital_prompt(self):
        prompt_template = f"""
        You are an AI agent acting as {self.ai_role}, assisting a user playing the role of {self.user_role} in the scenario "{self.scenario_name}". 
        Your persona is {self.ai_persona}, and your goal is to guide the user through a scenario-based learning experience at {self.place}.

        ### Task:
        {self.task} 
        The task must be completed step by step, progressing through multiple stages. Each stage presents a new challenge or decision point.

        ### Format:
        {self.format} 

        ### Exemplar:
        {self.exemplar} 

        ### Interaction Rules:
        - At each stage, provide an **initial prompt** describing the situation.  
        - If the user struggles, provide **subtle hints**—**never reveal the correct answer** directly.  
        - Respond dynamically to the user's input, offering **adaptive feedback** based on their choices.  
        - Only advance the user when they make the correct or reasonable decision.  

        ### Response Format:
        - **Initial Prompt**: {self.stage_description}  
        - **Hints**: {self.hint}  
        - **Feedback**:  
        - ✅ **Correct**: {self.positive_feedback}  
        - ❌ **Incorrect**: {self.constructive_feedback}  
        - **Next Stage**: {self.next_stage_condition}
        ### Stage description:
        """

        # Iterate through the stages and print descriptions
        for i, stage in enumerate(self.stages):
            first = True
            for j, step in enumerate(stage["stage_step"]):

                if first:
                    prompt_template += f"""
                ### Stage {i+1}:"""
                    first = False
                    continue

                prompt_template += f"""
                ## Step {i+1}.{j+1}:
                    - **Description**: {step["description"]}  
                    - **Hint**: {step["hint"]}  
                    - **Correct Response**: {step["correct_response"]}
                """
        
        prompt_template += f"""
        ### Tone & Style:
        - Use a **{self.tones[0]}** to match the urgency of the scenario.  
        - Write in a **{self.tones[1]}** for clarity and engagement.  
        - Maintain a **role-playing dynamic** to keep the user immersed in the experience.  
        """
        
        return prompt_template


    def logic(self, user_input, bar_change):
        """
        Performs the logic for generating a response based on the user input.

        Args:
            user_input (str): The user's input.
            bar_change (function): A function to update the progress bar.

        Returns:
            str: The final response generated based on the logic.
        """
        # Step 1: Prepare conversation history
        bar_change(0, 5, "Is it a question?")
        temp_text = self.prepare_conversation_history()
        docs = self.get_docs(user_input, 3)
        is_question = self.is_question(user_input)

        # Step 2: Determine input type
        bar_change(5, 15, "Is it a question\important?")
        is_important = self.is_important(user_input, is_question)
        is_info, final_prompt = self.check_optional(user_input)


        if not is_question:
            completed, self.current_stage = self.check_stage_completion()


            if completed:
                bar_change(15, 101, "Ending stage.")
                return "End"
        
        if not is_info:
            if is_important:
                # Step 3: Handle important input
                bar_change(15, 20, "Generating CoT response...")
                cot_answer = self.generate_cot_response(user_input, is_question, temp_text)

                bar_change(20, 30, "Performing self-consistency checks...")
                middle_prompt = self.self_consistency(user_input, cot_answer, 3, temp_text)
            else:
                # Step 4: Handle non-important input
                bar_change(15, 25, "Redirecting user...")
                middle_prompt = self.redirect_user(user_input, temp_text)

            # Step 5: Refine response
            bar_change(40, 50, "Generating feedback and refining response...")
            feedback1 = self.feedback(user_input, middle_prompt, temp_text, docs)
            refine1 = self.refine(middle_prompt, user_input, feedback1, temp_text)

            bar_change(60, 70, "Generating feedback and refining response...")
            feedback2 = self.feedback(user_input, refine1, temp_text, docs)
            final_prompt = self.refine(refine1, user_input, feedback2, temp_text)

        # Step 6: Finalize response
        bar_change(85, 90, "Finalizing response...")
        if is_question or is_important or is_info:
            next_steps = self.next_steps(user_input, final_prompt, temp_text)
            final_prompt = final_prompt + "\n\nNext Steps: " + next_steps

        # Step 7: Update conversation history
        bar_change(95, 100, "Response generated.")
        self.update_conversation_history(final_prompt, user_input)

        return final_prompt
        
    def check_optional(self, user_input):
        items_before_colon = [item.split(":")[0] for item in self.optionals + self.stage_informations[self.current_stage]]
        prompt = f"""
        You are given a sentence and a list of strings. Respond with "true" if the sentence either:
        1. Exactly matches one of the strings in the list,
        **or**
        2. Is a question that requests the list itself.

        Otherwise, respond with "false".

        Sentence: {user_input}
        List of strings: {items_before_colon}
        """


        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": prompt + user_input}]
        # )

        data = json.dumps({"prompt": prompt, "provider": self.provider, "model": self.llm_name})
        response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})


        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print("---------------------------------------------------------------")
        # print(f"\nCheck_optional first:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")

        if response.json()["answer"].strip().lower() == "true":
            prompt = f"""
            You are given a sentence and a list of strings. Extract and give all the information from the list of strings to answer the question from the sentence. Respond only with the information needed to answer the question.

            Sentence: {user_input}
            List of strings: {self.optionals + self.stage_informations[self.current_stage]}
            """

            
            data = json.dumps({"prompt": prompt, "provider": self.provider, "model": self.llm_name})
            response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})

            # response = self.client.chat.completions.create(
            #     model=self.llm_name,
            #     messages=[{"role": "user", "content": prompt + user_input}]
            # )
            # # Extract token usage from the response
            # Time = response.usage.total_time
            # prompt_tokens = response.usage.prompt_tokens
            # completion_tokens = response.usage.completion_tokens
            # total_tokens = response.usage.total_tokens

            # print("---------------------------------------------------------------")
            # print(f"\nCheck_optional second:\nTime Taken: {Time}")
            # print(f"Prompt tokens: {prompt_tokens}")
            # print(f"Completion tokens: {completion_tokens}")
            # print(f"Total tokens: {total_tokens}\n")

            return True, response.json()["answer"]
        
        return False, ""

    # Helper Methods
    def prepare_conversation_history(self):
        temp_text = "Here are the previous messages in the conversation:\n"
        if len(self.chat_history) > 5:
            temp_text += "".join(self.chat_history[0])  # First message
            temp_text += "... (skipping some messages) ...\n"
            temp_text += "".join(self.chat_history[-4:])  # Last five messages
        else:
            temp_text += "".join(self.chat_history)  # All messages
        return temp_text

    def update_conversation_history(self, llm_response, user_input):
        self.chat_history.append("User message: (" + user_input + ")\nLLM message: (" + llm_response + ")\n")

    def check_stage_completion(self):        
        flag = False
        temp = 0
        for index, stage in enumerate(self.stage_correct_response_check):
            for value in stage:
                if value == False:
                    flag = True
                    temp = index
                    break
            if flag:
                break
        return not flag, temp

    def is_important(self, user_input, is_question):
        prompt = ("You will be given a user input and a numbered list of possible correct responses."
        "Compare the user input to each response. Only return the index of the response that is an exact or near-exact match. If none match, return -1.\n\n"
        "Rules:\n"
        "- Do not guess or infer based on vague similarity.\n"
        "- Only match if the meaning is directly and clearly aligned.\n"
        "- Return just the index (e.g., -1, 0, 1, 2...)\n\n"
        "Possible responses:\n"
        )

        for index, correct_response in enumerate(self.stage_correct_response[self.current_stage]):
            prompt += f"{index}: {correct_response}\n"

        prompt += f"\nUser Input:\n{user_input}\n\nReturn only the index or -1 (e.g., -1, 0, 1, 2...)."
        prompt += ( "Examples:\n"
        "---\n"
        "User Input:\nI would take their blood pressure.\n"
        "Correct Responses:\n"
        "0: Check the patients pulse\n"
        "1: Administer CPR\n"
        "2: Measure the patients blood pressure\n"
        "3: Call for emergency support\n"
        "Answer: 2\n"
        "---\n"
        "User Input:\nSee if they are alive.\n"
        "Correct Responses:\n"
        "0: Check the patients pulse\n"
        "1: Administer CPR\n"
        "2: Measure the patiens blood pressure\n"
        "3: Call for emergency support\n"
        "Answer: -1\n"
        "---\n"
        "User Input:\nUse a for loop to print numbers 1 to 10.\n"
        "Correct Responses:\n"
        "0: Define a function in Python\n"
        "1: Use a for loop to print numbers 1 to 10\n"
        "2: Create a list with 10 elements\n"
        "3: Write a recursive factorial function\n"
        "Answer: 1\n"
        "---\n"
        "User Input:\nMake something that repeats printing numbers.\n"
        "Correct Responses:\n"
        "0: Define a function in Python\n"
        "1: Use a for loop to print numbers 1 to 10\n"
        "2: Create a list with 10 elements\n"
        "3: Write a recursive factorial function\n"
        "Answer: -1\n"
        "---\n"
        "User Input:\nI would simplify the fraction.\n"
        "Correct Responses:\n"
        "0: Multiply both sides of the equation by 2\n"
        "1: Simplify the fraction\n"
        "2: Factor the quadratic\n"
        "3: Convert the decimal to a fraction\n"
        "Answer: 1\n"
        "---\n"
        "User Input:\nMake the number smaller.\n"
        "Correct Responses:\n"
        "0: Multiply both sides of the equation by 2\n"
        "1: Simplify the fraction\n"
        "2: Factor the quadratic\n"
        "3: Convert the decimal to a fraction\n"
        "Answer: -1\n"
        "---\n"
        )


        data = json.dumps({"prompt": prompt, "provider": self.provider, "model": self.llm_name})
        response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})


        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": prompt + user_input}]
        # )

        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        try:
            response = int(response.json()["answer"])
        except ValueError:
            for i in range(-1, len(self.stage_correct_response[self.current_stage])):
                if str(i) in response.json()["answer"]:
                    response = i
                    break
        
        

        # print("---------------------------------------------------------------")
        # print(f"\Is_important:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")

        if response == -1:
            return False
        else:
            if not is_question:
                self.stage_correct_response_check[self.current_stage][response] = True
            return True

    
    def is_question(self, user_input):
        input_str = user_input.strip().lower()
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'can', 'could', 'should', 'would', 'is', 'are', 'does', 'did', 'will']

        if input_str.endswith('?'):
            return True
        if any(input_str.startswith(word) for word in question_words):
            return True
        
        user_input_prompt = f"""Is the following input a question?
        Input: "{user_input}"
        Respond with True if it's a question, otherwise respond with False.
        """

        
        data = json.dumps({"prompt": user_input_prompt, "provider": self.provider, "model": self.llm_name})
        content = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})


        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": user_input_prompt}]
        # )
        # content = response.choices[0].message.content.strip().lower()        
        
        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print("---------------------------------------------------------------")
        # print(f"\nIs_question:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")

        if content == "true":
            return True
        else:
            return False
    
    
    def generate_cot_response(self, user_input, is_question, chat_history):
        cot_agent_prompt = f"""\nCurrent Prompt:
        You are an AI using Chain-of-Thought (CoT) to guide a user in a scenario-based learning task.

        Context:
        User role: {self.user_role} | Scenario: "{self.scenario_name}"
        Input: "{user_input}" → Classified as {"Question" if is_question else "Action"}

        Instructions:
        Step 1 (CoT):
        If Question: Infer intent, hint subtly, avoid direct answers.
        If Action: Judge as valid/invalid/unclear; consider effects.

        Step 2 (Response):
        Stay in-role as {self.ai_role}; give immersive, adaptive feedback.

        Output:
        Use CoT before replying.
        Guide learning through role-play.
        Be concise, immersive, and educational.
        """

        
        data = json.dumps({"prompt": chat_history + cot_agent_prompt, "provider": self.provider, "model": self.llm_name})
        response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})



        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": chat_history + cot_agent_prompt}]
        # )
        
        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print("---------------------------------------------------------------")
        # print(f"\nGenerate_cot_response:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")

        return response.json()["answer"]
    
    
    def self_consistency(self, user_input, previous_ai_response, num_variations, chat_history):
        self_consistency_prompt = f"""\nCurrent Prompt:
        You are ensuring self-consistency in a scenario-based learning setting.

        Context:
        User role: {self.user_role} | Scenario: "{self.scenario_name}"
        Input: "{user_input}" | Prior response: "{previous_ai_response}"

        Instructions:
        Generate {num_variations} distinct responses, each with independent reasoning, maintaining:
        Scenario flow
        Role immersion
        Pedagogical value

        Compare responses:
        Find consistent patterns
        Pick the one with best logic, feedback quality, and engagement

        Select final reply:
        For questions → subtle hint
        For actions → accurate, immersive feedback

        Always preserve role and scenario logic

        Output:
        A single, refined response grounded in CoT consistency and learning impact with less than 100 words.
        """

        
        data = json.dumps({"prompt": chat_history + self_consistency_prompt, "provider": self.provider, "model": self.llm_name})
        response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})


        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": chat_history + self_consistency_prompt}]
        # )
        
        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print("---------------------------------------------------------------")
        # print(f"\nSelf_consistency:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")

        return response.json()["answer"]
    
    
    def feedback(self, user_input, previous_ai_response, chat_history, docs):
        feedback_prompt = f"""\nCurrent Prompt:
        You are evaluating your previous response in an interactive **scenario-based learning** environment.

        ### Context:
        - The user, acting as **{self.user_role}**, is navigating the scenario **"{self.scenario_name}"**.
        - The user's input was: **"{user_input}"**.
        - Your initial response was: **"{previous_ai_response}"**.

        ### Instructions:
        Analyze your response by considering the following:
        1. **Relevance**: Did the response correctly address the users input?
        2. **Guidance Quality**: If the users input was a question, did the response provide **subtle hints** without revealing the answer?
        3. **Correctness**: If the users input was an action, was the feedback **accurate and educational**?
        4. **Engagement**: Did the response maintain an immersive role-playing dynamic?
        5. **Clarity**: Was the response **clear, concise, and informative**?
        6. **Improvement Areas**: Identify any parts where the response could be more engaging, instructive, or immersive.

        ### Output:
        Provide a structured feedback report with the following:
        - **Strengths**: What aspects of the response were effective?
        - **Weaknesses**: What areas could be improved?
        - **Actionable Suggestions**: How can the response be refined?
        - **Size**: Keep the feedback concise, ideally under 100 words.

        Use the following documents (if relevant) to help you with the feedback, if they are not relevant, ignore them:\n\n{docs}
        """

        data = json.dumps({"prompt": chat_history + feedback_prompt, "provider": self.provider, "model": self.llm_name})
        response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})

        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": chat_history + feedback_prompt}]
        # )
        
        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print("---------------------------------------------------------------")
        # print(f"\nFeedback:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")
        
        return response.json()["answer"]

    def refine(self, previous_ai_response, user_input, self_feedback, chat_history):
        refinement_prompt = f"""\nCurrent Prompt:
        You are refining your previous response based on self-evaluation.

        ### Context:
        - The user, acting as **{self.user_role}**, is in the scenario **"{self.scenario_name}"**.
        - Their input was: **"{user_input}"**.
        - Your initial response was: **"{previous_ai_response}"**.
        - Your self-evaluation feedback was: **"{self_feedback}"**.

        ### Instructions:
        Use the feedback to generate a **revised response** that:
        1. **Addresses Weaknesses**: Correct any inaccuracies or vague explanations.
        2. **Enhances Guidance**: If the users input was a question, make the hints **more subtle yet effective**.
        3. **Improves Feedback Quality**: If the users input was an action, ensure the response **reinforces learning**.
        4. **Maintains Engagement**: Keep responses immersive, in-character as **{self.ai_role}**, and scenario-appropriate.
        5. **Increases Clarity**: Ensure the response is **concise, easy to understand, and pedagogically sound**.

        ### Output:
        Provide an improved version of your original response, ensuring it meets the refinement criteria while preserving scenario immersion and with less than 100 words.
        Strict rule: Do not include any introductory phrases—respond only with the raw hint text.
        """
        
        data = json.dumps({"prompt": chat_history + refinement_prompt, "provider": self.provider, "model": self.llm_name})
        response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})


        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": chat_history + refinement_prompt}]
        # )
        
        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print("---------------------------------------------------------------")
        # print(f"\nRefine:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")

        return response.json()["answer"]
    
    def redirect_user(self, user_input, chat_history):
        hint_prompt = f"""\nCurrent Prompt:
        You are an intelligent learning guide assisting a user in a scenario-based task.
        Your role is to provide gentle, context-aware nudges to help them reflect and correct course—without giving away answers.
        The user has made an input that is not aligned with the scenario or task at hand.

        Context:
        - AI Role: {self.ai_role}
        - User Role: {self.user_role}
        - Scenario: "{self.scenario_name}"
        - User Input: "{user_input}"

        Guidelines:
        - Evaluate the user's response based on the current stage of the scenario.
        - Identify where their reasoning may have gone off track.
        - Offer a thoughtful, in-character hint to guide their thinking—do **not** reveal the correct answer.
        - Encourage reflection, questioning, or a re-examination of the scenario.


        The following are the correct responses for the current stage, make sure to not mention them directly:
        index: correct response --> check
        """

        for index, correct_response in enumerate(self.stage_correct_response[self.current_stage]):
            hint_prompt += f"{index}: {correct_response} --> {self.stage_correct_response_check[self.current_stage][index]}\n"

        hint_prompt += f"""\n
        Output:
        Reply as a single, immersive hint (under 100 words), written in-character. Be supportive, focused, and subtly guide the learner toward better reasoning whithout revealing the answer.
        Make sure to say to the user that they are not aligned with the scenario or task at hand.
        """

        data = json.dumps({"prompt": chat_history + hint_prompt, "provider": self.provider, "model": self.llm_name})
        response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})

        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": chat_history + hint_prompt}]
        # )
        
        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print("---------------------------------------------------------------")
        # print(f"\nRedirect_user:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")

        return response.json()["answer"]

    def next_steps(self, user_action, previous_ai_response, chat_history):
        hint_prompt = f"""\nCurrent Prompt:
        You are guiding a user through a scenario-based learning task by giving subtle, role-appropriate hints.

        Context:
        Role: {self.user_role} | Scenario: "{self.scenario_name}" | Stage: {self.current_stage + 1}
        Last action: "{user_action}"
        Your last response: "{previous_ai_response}"

        Next Steps Description:
        """
        
        for index, complete in enumerate(self.stage_correct_response_check[self.current_stage]):
            if not complete:
                hint_prompt += f"""{index}: {self.stages[self.current_stage]["stage_step"][index + 1]}\n"""

        hint_prompt += f"""
        Instructions:
        Assess the users action.
        Give a subtle hint that encourages critical thinking—never reveal the answer.
        Stay in-character as {self.ai_role} and aligned with the scenario.

        Hint Strategy:
        Correct → Nudge toward the next logical step.
        Incorrect → Gently redirect.
        Unclear → Prompt for clarification with a guiding cue.

        Output:
        One immersive, hint-based response—subtle, clear, and pedagogically effective with less than 100 words.
        """
        
        data = json.dumps({"prompt": chat_history + hint_prompt, "provider": self.provider, "model": self.llm_name})
        response = requests.post(f"http://{self.host}:{self.port}/api/{self.token}/llm_completion", data=data, headers={"Content-Type": "application/json"})

        # response = self.client.chat.completions.create(
        #     model=self.llm_name,
        #     messages=[{"role": "user", "content": chat_history + hint_prompt}]
        # )
        
        # # Extract token usage from the response
        # Time = response.usage.total_time
        # prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_tokens = response.usage.total_tokens

        # print("---------------------------------------------------------------")
        # print(f"\nNext_steps:\nTime Taken: {Time}")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}\n")

        return response.json()["answer"]
    