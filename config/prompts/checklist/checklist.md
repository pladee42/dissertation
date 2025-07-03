# Instruction
You are an helpful assistant who identifies and summarizes key factors in large
    language models (LLMs) evaluation to help humans evaluate LLMs
    efficiently.
Feed any query into different LLMs, I will get various responses. I need to
    know in quick whether these responses follows the instructions and
    answers the question in the user query better.
I'll provide you with a user query. Your task is to identify those key factors
    that will affect my judgment and summarize them into a list to improve
    the efficiency of my evaluation.

# Conversation between User and AI

## Current User Query
<|begin_of_query|>
{user_query}
<|end_of_query|>


# Task
Given the above information, I need you to create a binary question list, so
    that I can perform an efficient and accurate evaluation through
    answering several questions.
Your question should be concise and include any necessary key content and
    information (such as keywords, formats, correct counts and values) in
    the user query or expected to be shown in responses. Your questions
    should evaluate all
    possible responses. Avoid creating duplicate, cumbersome or vague
    questions. For example, you should ask "Is this response contain the
    correct answer ..." instead of "Is this response's answer correct?". Ask
    fewer questions by aggregating questions with repeated contexts into one question.
    you should include the best answer to each question ("yes" / "no") to ensure
    the response aligns with user's requirements.
    you should include the priority ("very low", "low", "medium", "high", "very high") 
    that indicate the importance of each question to the response
You should generate at least 8 - 10 questions

## Output Format
Please provide your outputs in the following example JSON format.

[{"question": "Is the email generated in the same style as example email?", "best_ans": "yes", "priority": "very high"},
{"question": "Is this response contain the correct answer?", "best_ans": "no", "priority": "medium"}]