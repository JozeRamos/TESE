import os
import groq

# Read the API key from the file
with open('API.txt', 'r') as file:
    api_key = file.read().strip()

# Set the API key as an environment variable
os.environ["GROQ_API_KEY"] = api_key

# Initialize client
client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

def Inital_prompt(ai_role, user_role, scenario_name, ai_persona, place, task, format, exemplar, stage_description, hint, positive_feedback, constructive_feedback, next_stage_condition, stage_description_f, hint_f, correct_response_f, positive_feedback_f, incorrect_response_f, constructive_feedback_f, tone_1, tone_2):
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

    for i in range(len(stage_description_f)):
        prompt_template += f"""
        ### Stage description:
        ### Stage {i+1}:
        - **Description**: {stage_description_f[i]}  
        - **Hint**: {hint_f[i]}  
        - **Correct Response**: {correct_response_f[i]} → AI: "{positive_feedback_f[i]}"  
        - **Incorrect Response**: {incorrect_response_f[i]} → AI: "{constructive_feedback_f[i]}"  
        """
    
    prompt_template += f"""
    ### Tone & Style:
    - Use a **{tone_1}** to match the urgency of the scenario.  
    - Write in a **{tone_2}** for clarity and engagement.  
    - Maintain a **role-playing dynamic** to keep the user immersed in the experience.  
    """
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt_template}]
    )
    return [prompt_template, response.choices[0].message.content]

def is_question(user_role, scenario_name, user_input):
    user_input_prompt = f"""
    You are processing user input in an interactive scenario-based learning environment.

    ### Context:
    The user, acting as {user_role}, is navigating through the scenario "{scenario_name}". They may provide input that could either be a **question** or an **action**.

    ### Instructions:
    - Your task is to analyze the user's input "{user_input}" and determine whether it is a **question** or not.
    - If the input is a **question**, respond with **True**.
    - If the input is not a **question** (i.e., it is an **action**), respond with **False**.

    ### Definition of a Question:
    - A question typically ends with a question mark (`?`) and asks for information, clarification, or guidance.

    ### Example Interactions:

    1. **User Input (Question)**  
    - **User:** "What should I do first?"  
    - **AI:** True  

    2. **User Input (Action)**  
    - **User:** "Check for breathing."  
    - **AI:** False  

    3. **User Input (Action)**  
    - **User:** "Start CPR."  
    - **AI:** False  

    4. **User Input (Question)**  
    - **User:** "Can I use a defibrillator now?"  
    - **AI:** True  

    ### Output:
    - Return either **True** or **False** based on whether the user's input is a question or not.
    """
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": user_input_prompt}]
    )

    return response.choices[0].message.content




def generate_cot_response(user_role, scenario_name, user_input, is_question):
    cot_agent_prompt = f"""
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
    - **AI (Agent Worker Response):** "Assessing the patient’s condition is crucial. What key signs would indicate their breathing status?"  

    #### 2️⃣ User Takes a Correct Action:
    - **User:** "Check for breathing."  
    - **AI (CoT Reasoning):** "Checking for breathing is a fundamental first aid step. This action is correct and should progress the scenario."  
    - **AI (Agent Worker Response):** "Good! The patient is breathing but unconscious. What’s your next step?"  

    #### 3️⃣ User Takes an Incorrect Action:
    - **User:** "Start CPR."  
    - **AI (CoT Reasoning):** "CPR is only necessary if the patient is not breathing. I should redirect the user without revealing the answer outright."  
    - **AI (Agent Worker Response):** "CPR is an important skill, but consider checking the patient’s condition first. What signs would indicate that CPR is needed?"  

    #### 4️⃣ User Input is Unclear:
    - **User:** "Help the patient."  
    - **AI (CoT Reasoning):** "This input is too vague to process. I need to prompt the user to clarify what they mean."  
    - **AI (Agent Worker Response):** "How would you like to assist the patient? Checking their condition or calling for help could be a good start."  

    ### Output:
    - Respond with **immersive, role-play appropriate** guidance.
    - Maintain the **CoT reasoning process** before responding.
    - Always **prioritize learning and scenario immersion**.
    """


    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": cot_agent_prompt}]
    )

    return response.choices[0].message.content

def feedback(user_role, scenario_name, user_input, previous_ai_response):
    feedback_prompt = f"""
    You are evaluating your previous response in an interactive **scenario-based learning** environment.

    ### Context:
    - The user, acting as **{user_role}**, is navigating the scenario **"{scenario_name}"**.
    - The user's input was: **"{user_input}"**.
    - Your initial response was: **"{previous_ai_response}"**.

    ### Instructions:
    Analyze your response by considering the following:
    1. **Relevance**: Did the response correctly address the user’s input?
    2. **Guidance Quality**: If the user’s input was a question, did the response provide **subtle hints** without revealing the answer?
    3. **Correctness**: If the user’s input was an action, was the feedback **accurate and educational**?
    4. **Engagement**: Did the response maintain an immersive role-playing dynamic?
    5. **Clarity**: Was the response **clear, concise, and informative**?
    6. **Improvement Areas**: Identify any parts where the response could be more engaging, instructive, or immersive.

    ### Output:
    Provide a structured feedback report with the following:
    - **Strengths**: What aspects of the response were effective?
    - **Weaknesses**: What areas could be improved?
    - **Actionable Suggestions**: How can the response be refined?
    """
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": feedback_prompt}]
    )
    return response.choices[0].message.content

def refine(previous_ai_response, user_role, scenario_name, user_input, self_feedback):
    refinement_prompt = f"""
    You are refining your previous response based on self-evaluation.

    ### Context:
    - The user, acting as **{user_role}**, is in the scenario **"{scenario_name}"**.
    - Their input was: **"{user_input}"**.
    - Your initial response was: **"{previous_ai_response}"**.
    - Your self-evaluation feedback was: **"{self_feedback}"**.

    ### Instructions:
    Use the feedback to generate a **revised response** that:
    1. **Addresses Weaknesses**: Correct any inaccuracies or vague explanations.
    2. **Enhances Guidance**: If the user’s input was a question, make the hints **more subtle yet effective**.
    3. **Improves Feedback Quality**: If the user’s input was an action, ensure the response **reinforces learning**.
    4. **Maintains Engagement**: Keep responses immersive, in-character as **{ai_role}**, and scenario-appropriate.
    5. **Increases Clarity**: Ensure the response is **concise, easy to understand, and pedagogically sound**.

    ### Output:
    Provide an improved version of your original response, ensuring it meets the refinement criteria while preserving scenario immersion.
    """
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": refinement_prompt}]
    )
    return response.choices[0].message.content

def next_steps(user_role, scenario_name, user_action, previous_ai_response, current_stage):
    hint_prompt = f"""
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
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": hint_prompt}]
    )
    return response.choices[0].message.content

def self_consistency(user_role, scenario_name, user_input, previous_ai_response, num_variations):
    self_consistency_prompt = f"""
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
    1. "Checking the pulse is crucial for assessing the patient’s condition. Do you feel a strong, regular pulse?"
    2. "A pulse check gives key information. If it’s weak or absent, what might you consider next?"
    3. "Monitoring circulation is important. What do you observe after checking their pulse?"  

    - **AI Evaluates:**  
    - All responses align with **correct scenario logic**.  
    - Response 2 is chosen because it **guides the user without giving away the answer**.  

    #### 🔄 User Asks a Question:
    - **User:** "Should I move the patient?"  
    - **AI Generates Three Variations:**
    1. "Before moving them, what risks should you consider?"
    2. "Think about the patient’s injuries. What factors determine whether moving is safe?"
    3. "Moving a patient can be risky. What assessment should you do first?"  

    - **AI Evaluates:**  
    - All responses hint at **checking for injuries first**.  
    - Response 2 is chosen as it **maintains engagement and encourages critical thinking**.  

    ### Output:
    - **A final, self-consistent response**, based on analyzing multiple reasoning paths.
    - The response should be **logical, immersive, and pedagogically effective**.
    """
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": self_consistency_prompt}]
    )
    return response.choices[0].message.content



# Example usage
if __name__ == "__main__":

    ai_role="Virtual Mentor",  
    user_role="Junior Developer",  
    scenario_name="Debugging a Recursive Function",  
    ai_persona="Experienced Software Engineer with a specialization in algorithms and debugging",  
    place="A fast-paced software development office where you're working on a critical backend feature",  
    task="Identify and fix a faulty recursive function designed to calculate the factorial of a number, ensuring it correctly handles base cases and recursive calls.",  
    format="Step-by-step text-based scenario where you troubleshoot and refine the recursive function through logical reasoning and debugging techniques.",  
    exemplar="User: 'Check the base case' → AI: 'Good! The base case prevents infinite recursion. Whats your next step?'",  

    stage_description="Your lead developer has assigned you a debugging task: a recursive function intended to compute factorials is running indefinitely when called with `n=5`. You need to investigate and fix the issue. Whats your first step?",  
    hint="Think about what prevents a recursive function from going on forever.",
    positive_feedback="You're on the right track! Recursion needs a well-defined stopping point.",  
    constructive_feedback="Something is missing that tells the function when to stop. What could it be?",  
    next_stage_condition="Proceed only if the user correctly identifies a flaw in the functions base case or recursive step.",  

    stage_1_description="The function takes an integer `n` but runs indefinitely when `n=5`, consuming system resources and eventually causing a stack overflow.",  
    hint_1="Ask yourself: under what condition should the function stop calling itself?",  
    correct_response_1="Check if the base case is missing or incorrectly defined.",  
    positive_feedback_1="Nice work! A function that keeps calling itself forever is a sign something isnt telling it to stop.",  
    incorrect_response_1="Modify the recursive call first.",  
    constructive_feedback_1="Think about why the function doesnt know when to stop. What would tell it to halt?",  

    stage_2_description="You have identified that the function lacks a correct base case. Now, you need to define an appropriate stopping condition.",  
    hint_2="Consider the smallest input that doesnt require further breakdown.",  
    correct_response_2="Ensure the function returns `1` when `n` is `0`.",  
    positive_feedback_2="Good thinking! A solid stopping condition keeps recursion in check.",  
    incorrect_response_2="Set the base case to return `0` when `n=1`.",  
    constructive_feedback_2="Not quite! The base case should reflect a fundamental property of the problem.",  

    stage_3_description="The base case is now correct, but the function still isn't returning the expected factorial values. It's time to verify the recursive step.",  
    hint_3="Each step should bring you closer to the stopping condition while preserving the logic of the operation.",  
    correct_response_3="Ensure the function correctly multiplies `n * factorial(n-1)`.",  
    positive_feedback_3="Nice! Each step should mirror how the problem naturally reduces itself.",  
    incorrect_response_3="Subtract `2` instead of `1` in the recursive call.",  
    constructive_feedback_3="Look carefully—does each step truly lead to the stopping point, or is something being skipped?",  

    stage_description_f = [stage_1_description, stage_2_description, stage_3_description]
    hint_f = [hint_1, hint_2, hint_3]
    correct_response_f = [correct_response_1, correct_response_2, correct_response_3]
    positive_feedback_f = [positive_feedback_1, positive_feedback_2, positive_feedback_3]
    incorrect_response_f = [incorrect_response_1, incorrect_response_2, incorrect_response_3]
    constructive_feedback_f = [constructive_feedback_1, constructive_feedback_2, constructive_feedback_3]

    tone_1="Professional and supportive, providing logical guidance without giving away direct answers.",  
    tone_2="Clear and engaging, encouraging problem-solving through thoughtful questioning."  

    initial = Inital_prompt(ai_role, user_role, scenario_name, ai_persona, place,
                   task, format, exemplar, stage_description, hint,
                     positive_feedback, constructive_feedback,
                       next_stage_condition, stage_description_f,
                         hint_f, correct_response_f, positive_feedback_f,
                           incorrect_response_f, constructive_feedback_f,
                             tone_1, tone_2)
    chat_history = []
    chat_history.append(initial[0])
    chat_history.append(initial[1])

    while True:
        con = input("Press 1 to continue or 0 to quit: ")
        if con == "0":
            break
        
        user_answer = input("What do you want to do? Press 1 for a hint.")
        print("\n🤖 Is the user input a question?")
        if user_answer != "1":
            v = is_question(user_role, scenario_name, user_answer)
            print(v)
        Cot_answer = generate_cot_response(user_role, scenario_name, user_answer, v)
        print("\n🔍 Step-by-step reasoning feedback:" + Cot_answer)
        self_consistency1 = self_consistency(user_role, scenario_name, user_answer, Cot_answer, 3)
        feedback1 = feedback(user_role, scenario_name, user_answer, Cot_answer)
        refine1 = refine(Cot_answer, user_role, scenario_name, user_answer, feedback1)
        feedback2 = feedback(user_role, scenario_name, user_answer, refine1)
        refine2 = refine(refine1, user_role, scenario_name, user_answer, feedback2)
        print("\n🔍 Self-evaluation feedback1:" + feedback1)
        print("\n🔧 Refinement based on self-evaluation1:" + refine1)
        print("\n🔍 Self-evaluation feedback2:" + feedback2)
        print("\n🔧 Refinement based on self-evaluation2:" + refine2)
        if "false" in v.lower():
            print("\n🔍 Next steps hint:")
            next_steps1 = next_steps(user_role, scenario_name, user_answer, refine2, 1)
            print(next_steps1)



