import json
import os
import groq
import threading

class LLM:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        
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
        self.chat_history = []
        # Read the API key from the file
        with open('source/API.txt', 'r') as file:
            self.api_key = file.read().strip()
        # Set the API key as an environment variable
        os.environ["GROQ_API_KEY"] = self.api_key
        # Initialize client
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        
        initial = self.Inital_prompt(self.ai_role, self.user_role, self.scenario_name, self.ai_persona, self.place,
                    self.task, self.format, self.exemplar, self.stage_description, self.hint,
                        self.positive_feedback, self.constructive_feedback,
                        self.next_stage_condition, self.stages,
                                self.tones[0], self.tones[1])
        

        self.chat_history.append("Initial prompt: " + initial[0] + "\n")
        self.chat_history.append("LLM response: " + initial[1] + "\n")


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

    def get_constructive_feedback(self):
        return self.constructive_feedback

    def get_next_stage_condition(self):
        return self.next_stage_condition

    def get_stages(self):
        return self.stages

    def get_tones(self):
        return self.tones
    
    def Inital_prompt(self, ai_role, user_role, scenario_name, ai_persona, place, task, format, exemplar, stage_description, hint, positive_feedback, constructive_feedback, next_stage_condition, stages, tone_1, tone_2):
        prompt_template = f"""
        You are an AI agent acting as {ai_role}, assisting a user playing the role of {user_role} in the scenario "{scenario_name}". 
        Your persona is {ai_persona}, and your goal is to guide the user through a scenario-based learning experience at {place}.

        ### Task:
        {task} 
        The task must be completed step by step, progressing through multiple stages. Each stage presents a new challenge or decision point.

        ### Format:
        {format} 

        ### Exemplar:
        {exemplar} 

        ### Interaction Rules:
        - At each stage, provide an **initial prompt** describing the situation.  
        - If the user struggles, provide **subtle hints**—**never reveal the correct answer** directly.  
        - Respond dynamically to the user's input, offering **adaptive feedback** based on their choices.  
        - Only advance the user when they make the correct or reasonable decision.  

        ### Response Format:
        - **Initial Prompt**: {stage_description}  
        - **Hints**: {hint}  
        - **Feedback**:  
        - ✅ **Correct**: {positive_feedback}  
        - ❌ **Incorrect**: {constructive_feedback}  
        - **Next Stage**: {next_stage_condition}  
        """

        for i in range(len(stages)):
            prompt_template += f"""
            ### Stage description:
            ### Stage {i+1}:
            - **Description**: {stages[i]["description"]}  
            - **Hint**: {stages[i]["hint"]}  
            - **Correct Response**: {stages[i]["correct_response"]} → AI: "{stages[i]["positive_feedback"]}"  
            - **Incorrect Response**: {stages[i]["incorrect_response"]} → AI: "{stages[i]["constructive_feedback"]}"  
            """
        
        prompt_template += f"""
        ### Tone & Style:
        - Use a **{tone_1}** to match the urgency of the scenario.  
        - Write in a **{tone_2}** for clarity and engagement.  
        - Maintain a **role-playing dynamic** to keep the user immersed in the experience.  
        """
        response = self.client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt_template}]
        )
        return [prompt_template, response.choices[0].message.content]


    def logic(self, user_input, bar_change):
        # Step 1: Prepare conversation history
        temp_text = self.prepare_conversation_history(user_input)
        
        bar_change(0, 5, "Is it a question?")

        # Step 2: Determine if the input is a question
        v = self.is_question(self.client.chat, self.user_role, self.scenario_name, user_input)

        bar_change(10, 20, "Generating CoT response...")

        # Step 3: Generate the chain-of-thought (CoT) response
        cot_answer = self.generate_cot_response(
            self.client.chat, self.user_role, self.ai_role, self.scenario_name, user_input, v, temp_text
        )

        bar_change(20, 30, "Performing self-consistency checks...")

        # Step 4: Perform self-consistency checks
        self_consistency1 = self.self_consistency(
            self.client.chat, self.user_role, self.scenario_name, user_input, cot_answer, 3, temp_text
        )

        bar_change(40, 50, "Generating feedback and refining response...")

        # Step 5: Generate feedback and refine the response (first iteration)
        feedback1 = self.feedback(
            self.client.chat, self.user_role, self.scenario_name, user_input, self_consistency1, temp_text
        )
        refine1 = self.refine(
            self.client.chat, self_consistency1, self.user_role, self.ai_role, self.scenario_name, user_input, feedback1, temp_text
        )

        bar_change(60, 70, "Generating feedback and refining response...")

        # Step 6: Generate feedback and refine the response (second iteration)
        feedback2 = self.feedback(
            self.client.chat, self.user_role, self.scenario_name, user_input, refine1, temp_text
        )
        refine2 = self.refine(
            self.client.chat, refine1, self.user_role, self.ai_role, self.scenario_name, user_input, feedback2, temp_text
        )

        bar_change(85, 90, "Finalizing response...")

        # Step 7: Handle "next steps" if the input is not a valid question
        if "false" in v.lower():
            next_steps = self.next_steps(
                self.client.chat, self.user_role, self.ai_role, self.scenario_name, user_input, refine2, 1, temp_text
            )
            refine2 += "\nNext Steps: " + next_steps

        bar_change(95, 100, "Response generated.")

        # Step 8: Update conversation history and return the final response
        self.update_conversation_history(user_input, refine2)
        return "refine2"

    # Helper Methods
    def prepare_conversation_history(self, user_input):
        temp_text = "Here are the previous messages in the conversation:\n"
        for message in self.chat_history:
            temp_text += message
        self.chat_history.append("User message: (" + user_input + ")\n")
        return temp_text

    def update_conversation_history(self, llm_response):
        self.chat_history.append("LLM message: (" + llm_response + ")\n")
    

    
    def is_question(self, chat, user_input):
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


        response = chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": user_input_prompt}]
        )
        return response.choices[0].message.content
    
    
    def generate_cot_response(self, chat, user_role, ai_role, scenario_name, user_input, is_question, chat_history):
        cot_agent_prompt = f"""\nCurrent Prompt:
        You are an AI agent using **Chain-of-Thought (CoT) reasoning** to analyze and respond to the user's input in an interactive **scenario-based learning environment**.

        ### Context:
        - The user, acting as **{user_role}**, is navigating the scenario **"{scenario_name}"**.
        - Their previous input was: **"{user_input}"**.
        - You have already determined that this input is a **{"Question" if is_question else "Action"}**.

        ### Instructions:
        - **Step 1: Think Step-by-Step (CoT)**
        - **If the input is a question**, analyze the intent and provide **subtle guidance** without giving the direct answer.
        - **If the input is an action**, determine if it is **valid**, **invalid**, or **requires clarification**.
        - Reason through possible consequences of the user's input before responding.

        - **Step 2: Act as an Agent Worker**
        - Simulate an expert in the field related to the scenario (e.g., a doctor for a medical scenario).
        - Stay in-character as **{ai_role}** to enhance immersion.
        - Provide **adaptive feedback** to help the user learn from their decisions.

        ### Response Logic:
        1. **For Questions:**
        - Think through what information the user is missing.
        - Provide a **hint or guidance** rather than directly stating the answer.
        - Encourage the user to think critically.

        2. **For Actions:**
        - If the action is **correct**, acknowledge it and describe its impact on the scenario.
        - If the action is **incorrect**, give **subtle feedback** without revealing the answer.
        - If the action is **unclear**, prompt the user to clarify their intent.

        ### Example Interactions:

        #### 1️⃣ User Asks a Question:
        - **User:** "Should I check for breathing first?"  
        - **AI (CoT Reasoning):** "The user is asking about the correct sequence of actions in an emergency. Instead of directly answering, I should guide them toward thinking about initial assessments."  
        - **AI (Agent Worker Response):** "Assessing the patients condition is crucial. What key signs would indicate their breathing status?"  

        #### 2️⃣ User Takes a Correct Action:
        - **User:** "Check for breathing."  
        - **AI (CoT Reasoning):** "Checking for breathing is a fundamental first aid step. This action is correct and should progress the scenario."  
        - **AI (Agent Worker Response):** "Good! The patient is breathing but unconscious. Whats your next step?"  

        #### 3️⃣ User Takes an Incorrect Action:
        - **User:** "Start CPR."  
        - **AI (CoT Reasoning):** "CPR is only necessary if the patient is not breathing. I should redirect the user without revealing the answer outright."  
        - **AI (Agent Worker Response):** "CPR is an important skill, but consider checking the patients condition first. What signs would indicate that CPR is needed?"  

        #### 4️⃣ User Input is Unclear:
        - **User:** "Help the patient."  
        - **AI (CoT Reasoning):** "This input is too vague to process. I need to prompt the user to clarify what they mean."  
        - **AI (Agent Worker Response):** "How would you like to assist the patient? Checking their condition or calling for help could be a good start."  

        ### Output:
        - Respond with **immersive, role-play appropriate** guidance.
        - Maintain the **CoT reasoning process** before responding.
        - Always **prioritize learning and scenario immersion**.
        """


        response = chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": chat_history + cot_agent_prompt}]
        )

        return response.choices[0].message.content
    
    
    def self_consistency(self, chat, user_role, scenario_name, user_input, previous_ai_response, num_variations, chat_history):
        self_consistency_prompt = f"""\nCurrent Prompt:
        You are ensuring **self-consistency** in your response within an interactive **scenario-based learning environment**.

        ### Context:
        - The user, acting as **{user_role}**, is navigating the scenario **"{scenario_name}"**.
        - Their input was: **"{user_input}"**.
        - Your initial response was: **"{previous_ai_response}"**.

        ### Instructions:
        1. **Generate Multiple Independent Reasoning Paths**  
        - Produce **{num_variations}** different responses to the user's input.  
        - Each response should be **independently reasoned**, taking into account:
            - Scenario progression
            - Role-playing immersion
            - Pedagogical effectiveness  

        2. **Evaluate Consistency Across Responses**  
        - Compare the generated responses and identify the **common patterns**.
        - Determine which response aligns best with **logical progression, accuracy, and engagement**.
        
        3. **Select the Most Reliable Response**  
        - The final response should reflect the **most consistent** reasoning across variations.
        - If discrepancies arise, choose the response that:
            - Provides the **best subtle hint** (for questions)
            - Gives the **most accurate but engaging feedback** (for actions)
            - Maintains **scenario immersion and role-play quality**.

        ### Example Process:

        #### ✅ User Takes an Action:
        - **User:** "Check the patient's pulse."  
        - **AI Generates Three Variations:**
        1. "Checking the pulse is crucial for assessing the patients condition. Do you feel a strong, regular pulse?"
        2. "A pulse check gives key information. If its weak or absent, what might you consider next?"
        3. "Monitoring circulation is important. What do you observe after checking their pulse?"  

        - **AI Evaluates:**  
        - All responses align with **correct scenario logic**.  
        - Response 2 is chosen because it **guides the user without giving away the answer**.  

        #### 🔄 User Asks a Question:
        - **User:** "Should I move the patient?"  
        - **AI Generates Three Variations:**
        1. "Before moving them, what risks should you consider?"
        2. "Think about the patients injuries. What factors determine whether moving is safe?"
        3. "Moving a patient can be risky. What assessment should you do first?"  

        - **AI Evaluates:**  
        - All responses hint at **checking for injuries first**.  
        - Response 2 is chosen as it **maintains engagement and encourages critical thinking**.  

        ### Output:
        - **A final, self-consistent response**, based on analyzing multiple reasoning paths.
        - The response should be **logical, immersive, and pedagogically effective**.
        """
        response = chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": chat_history + self_consistency_prompt}]
        )
        return response.choices[0].message.content
    
    
    def feedback(self, chat, user_role, scenario_name, user_input, previous_ai_response, chat_history):
        feedback_prompt = f"""\nCurrent Prompt:
        You are evaluating your previous response in an interactive **scenario-based learning** environment.

        ### Context:
        - The user, acting as **{user_role}**, is navigating the scenario **"{scenario_name}"**.
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
        """
        response = chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": chat_history + feedback_prompt}]
        )
        return response.choices[0].message.content

    def refine(self, chat, previous_ai_response, user_role, ai_role, scenario_name, user_input, self_feedback, chat_history):
        refinement_prompt = f"""\nCurrent Prompt:
        You are refining your previous response based on self-evaluation.

        ### Context:
        - The user, acting as **{user_role}**, is in the scenario **"{scenario_name}"**.
        - Their input was: **"{user_input}"**.
        - Your initial response was: **"{previous_ai_response}"**.
        - Your self-evaluation feedback was: **"{self_feedback}"**.

        ### Instructions:
        Use the feedback to generate a **revised response** that:
        1. **Addresses Weaknesses**: Correct any inaccuracies or vague explanations.
        2. **Enhances Guidance**: If the users input was a question, make the hints **more subtle yet effective**.
        3. **Improves Feedback Quality**: If the users input was an action, ensure the response **reinforces learning**.
        4. **Maintains Engagement**: Keep responses immersive, in-character as **{ai_role}**, and scenario-appropriate.
        5. **Increases Clarity**: Ensure the response is **concise, easy to understand, and pedagogically sound**.

        ### Output:
        Provide an improved version of your original response, ensuring it meets the refinement criteria while preserving scenario immersion.
        """
        response = chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": chat_history + refinement_prompt}]
        )
        return response.choices[0].message.content
    
    
    def next_steps(self, chat, user_role, ai_role, scenario_name, user_action, previous_ai_response, current_stage, chat_history):
        hint_prompt = f"""\nCurrent Prompt:
        You are guiding the user through an interactive **scenario-based learning environment** by providing **subtle hints** for the next steps.

        ### Context:
        - The user, acting as **{user_role}**, is progressing through the scenario **"{scenario_name}"**.
        - Their last action was: **"{user_action}"**.
        - Your previous response to their action was: **"{previous_ai_response}"**.
        - The scenario is currently at **Stage {current_stage}**.

        ### Instructions:
        - **Analyze the user's last action** and determine what should logically happen next.
        - Provide a **subtle hint** that encourages the user to think critically about their next step **without directly stating the answer**.
        - Keep the hint **immersive and role-appropriate**, aligned with **{ai_role}**.

        ### Hint Strategy:
        1. **If the action was correct** → Guide the user toward the **next logical step** in the scenario.
        2. **If the action was incorrect** → Give a hint to **redirect** them without explicitly telling them they were wrong.
        3. **If the action was unclear** → Prompt them for **clarification** while subtly hinting at what they should consider.

        ### Example Interactions:

        #### ✅ User Takes a Correct Action:
        - **User:** "Check for breathing."  
        - **AI (Hint):** "Good. Observing the patient's breathing is crucial. What signs might indicate they need immediate intervention?"  

        #### ❌ User Takes an Incorrect Action:
        - **User:** "Start CPR."  
        - **AI (Hint):** "CPR is a life-saving measure, but timing is important. What should you check first before deciding to start?"  

        #### 🔄 User Action is Unclear:
        - **User:** "Help the patient."  
        - **AI (Hint):** "There are many ways to assist. Are you focusing on assessing their condition, providing immediate aid, or calling for help?"  

        ### Output:
        - **A single, subtle hint** guiding the user toward the next step.
        - Maintain immersion and role-play dynamics.
        - Do not reveal the direct answer; instead, encourage **critical thinking**.
        """
        response = chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": chat_history + hint_prompt}]
        )
        return response.choices[0].message.content
    