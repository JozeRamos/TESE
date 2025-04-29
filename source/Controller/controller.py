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

        # Encoder
        self.encoder = HuggingFaceEncoder(
            name="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda"  # <-- Forces GPU usage
        )
        
        # Chat history
        self.chat_history = []

        self.llm_name = "llama3-70b-8192"

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

    def _load_api_keys(self):
        with open('source/API.txt', 'r') as file:
            os.environ["GROQ_API_KEY"] = file.read().strip()
        with open('source/API2.txt', 'r') as file:
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

        # Step 2: Create index if needed
        if self.index_name in [idx["name"] for idx in self.pc.list_indexes()]:
            print(f"Index '{self.index_name}' already exists. Deleting it to reset the dimension...")
            self.pc.delete_index(self.index_name)
            # Wait for the index to be fully deleted
            while self.index_name in [idx["name"] for idx in self.pc.list_indexes()]:
                time.sleep(1)

        # Create the index with the correct dimension
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


    def get_ai_role(self):
        return self.ai_role

    def get_user_role(self):
        return self.user_role

    def get_scenario_name(self):
        return self.scenario_name

    def get_ai_persona(self):
        return self.ai_persona

    def get_place(self):
        return self.place

    def get_task(self):
        return self.task

    def get_format(self):
        return self.format

    def get_exemplar(self):
        return self.exemplar

    def get_stage_description(self):
        return self.stage_description

    def get_hint(self):
        return self.hint

    def get_positive_feedback(self):
        return self.positive_feedback

    def get_next_stage_condition(self):
        return self.next_stage_condition

    def get_stages(self):
        return self.stages

    def get_tones(self):
        return self.tones
    
    def get_docs(self, query: str, top_k: int) -> list[str]:
        # encode query
        xq = self.encoder([query])
        # search pinecone index
        res = self.index.query(vector=xq, top_k=top_k, include_metadata=True)
        # get doc text
        docs = [x["metadata"]['context'] for x in res["matches"]]
        return docs
    
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
        # Step 1: Prepare conversation history
        temp_text = self.prepare_conversation_history(user_input)
        
        bar_change(0, 5, "Is it a question?")

        # Step 2: Determine if the input is a question
        v = self.is_question(user_input)

        bar_change(10, 20, "Generating CoT response...")

        # Step 3: Generate the chain-of-thought (CoT) response
        cot_answer = self.generate_cot_response(
            self.client.chat, user_input, v, temp_text
        )

        bar_change(20, 30, "Performing self-consistency checks...")

        # Step 4: Perform self-consistency checks
        self_consistency1 = self.self_consistency(
            user_input, cot_answer, 3, temp_text
        )

        bar_change(40, 50, "Generating feedback and refining response...")

        # Step 5: Generate feedback and refine the response (first iteration)
        feedback1 = self.feedback(
            user_input, self_consistency1, temp_text
        )
        refine1 = self.refine(
            self_consistency1, user_input, feedback1, temp_text
        )

        bar_change(60, 70, "Generating feedback and refining response...")

        # Step 6: Generate feedback and refine the response (second iteration)
        feedback2 = self.feedback(
            user_input, refine1, temp_text
        )
        refine2 = self.refine(
            refine1, user_input, feedback2, temp_text
        )

        bar_change(85, 90, "Finalizing response...")

        # Step 7: Handle "next steps" if the input is not a valid question
        if v:
            next_steps = self.next_steps(
                user_input, refine2, 1, temp_text
            )
            refine2 = refine2 + "\n\nNext Steps: " + next_steps

        bar_change(95, 100, "Response generated.")

        # Step 8: Update conversation history and return the final response
        self.update_conversation_history(refine2)
        return refine2

    # Helper Methods
    def prepare_conversation_history(self, user_input):
        temp_text = "Here are the previous messages in the conversation:\n"
        for message in self.chat_history:
            temp_text += message
        self.chat_history.append("User message: (" + user_input + ")\n")
        return temp_text

    def update_conversation_history(self, llm_response):
        self.chat_history.append("LLM message: (" + llm_response + ")\n")
    

    
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


        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": user_input_prompt}]
        )
        return response.choices[0].message.content
    
    
    def generate_cot_response(self, chat, user_input, is_question, chat_history):
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


        response = chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": chat_history + cot_agent_prompt}]
        )

        return response.choices[0].message.content
    
    
    def self_consistency(self, num_variations, chat_history):
        self_consistency_prompt = f"""\nCurrent Prompt:
        You are ensuring self-consistency in a scenario-based learning setting.

        Context:
        User role: {self.user_role} | Scenario: "{self.scenario_name}"
        Input: "{self.user_input}" | Prior response: "{self.previous_ai_response}"

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
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": chat_history + self_consistency_prompt}]
        )
        return response.choices[0].message.content
    
    
    def feedback(self, user_input, previous_ai_response, chat_history):
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
        """
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": chat_history + feedback_prompt}]
        )
        return response.choices[0].message.content

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
        """
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": chat_history + refinement_prompt}]
        )
        return response.choices[0].message.content
    
    
    def next_steps(self, user_action, previous_ai_response, current_stage, chat_history):
        hint_prompt = f"""\nCurrent Prompt:
        You are guiding a user through a scenario-based learning task by giving subtle, role-appropriate hints.

        Context:
        Role: {self.user_role} | Scenario: "{self.scenario_name}" | Stage: {current_stage}
        Last action: "{user_action}"
        Your last response: "{previous_ai_response}"

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
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": chat_history + hint_prompt}]
        )
        return response.choices[0].message.content
    