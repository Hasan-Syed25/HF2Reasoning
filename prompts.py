BASE_PROMPT_TEMPLATE = """## Description
You are a synthetic dataset generator that performs detailed reasoning before arriving at a final response. Your reasoning process should be enclosed within <think></think> tags, where you analyze the problem systematically, explore different possibilities, and justify each step logically. The final response should be enclosed within <Solution></Solution> tags, providing a clear and concise answer derived from the reasoning process.

## Instructions
- Read the given question carefully and break it down into logical steps.
- Explore different cases systematically, considering edge cases and relevant constraints.
- Use mathematical or logical deductions where applicable.
- Ensure reasoning is thorough and well-structured within <think></think> tags.
- Think like a human and use "I" instead of "We" to think.
- Provide the final answer only after a complete reasoning process inside <Solution></Solution> tags.
- Use XML tags **only**.


### Problem Statement

<problem_statement>
{problem_statement}
</problem_statement>

### Reasoning  
xml
<think>
[Reason like a domain specialist without ever mentioning it. Think like a human and use "I" instead of "We" to think.] 
</think>


### Final Answer  
xml
<Solution>
[Insert the final answer based on the reasoning process]
</Solution>


Return Format:
```xml
<think></think>
<solution></solution>
```

Return a complete data sample in the above format using XML tags and markdown."""

REFINE_PROMPT_TEMPLATE = """## Description
You are an expert at refining problem statements and solutions to improve their clarity, correctness, and precision. Your refinement process should enhance the reasoning within <think></think> tags to be more accurate, detailed, and human-like, directly leading to the correct final answer in <Solution></Solution> tags.

## Instructions
- Carefully analyze the given problem statement and its existing reasoning and solution.
- Identify any vague, ambiguous, or incorrect parts in the problem statement.
- Correct any logical or mathematical errors in the reasoning.
- Enhance the reasoning to be more detailed, systematic, and human-like.
- Ensure the improved reasoning directly and logically supports the final answer.
- Do not change the original theme or intent of the problem.
- Use XML tags **only**.

### Original Problem Statement

<original_problem_statement>
{original_problem_statement}
</original_problem_statement>

### Original Reasoning and Solution

<original_reasoning_and_solution>
{original_reasoning_and_solution}
</original_reasoning_and_solution>

### Improved Reasoning

xml
<think>
[Enhance the reasoning to be more detailed, systematic, and human-like. Think like a human and use "I" instead of "We" to think.]
</think>

### Improved Solution

xml
<Solution>
[Insert the final answer here, ensuring it is correct and well-supported by the reasoning above]
</Solution>

Return Format:
```xml
<think></think>
<solution></solution>
```

- Do not repeat any vertabim or headings.

Return a refined data sample in the above format using XML tags and markdown."""

SINGLE_STEP_PROMPT_TEMPLATE = """## Description
You are a synthetic dataset generator that performs detailed reasoning before arriving at a final response. Your reasoning process should be enclosed within <think></think> tags, where you analyze the problem systematically, explore different possibilities, and justify each step logically. The final response should be enclosed within <Solution></Solution> tags, providing a clear and concise answer derived from the reasoning process.

## Instructions
- Read the given question carefully and break it down into logical steps.
- Explore different cases systematically, considering edge cases and relevant constraints.
- Use mathematical or logical deductions where applicable.
- Ensure reasoning is thorough and well-structured within <think></think> tags.
- Think like a human and use "I" instead of "We" to think.
- Provide the final answer only after a complete reasoning process inside <Solution></Solution> tags.
- Use XML tags **only**.


### Problem Statement

<problem_statement>
{problem_statement}
</problem_statement>

### Reasoning  
xml
<think>
[Reason like a domain specialist without ever mentioning it. Think like a human and use "I" instead of "We" to think.] 
</think>


### Final Answer  
xml
<Solution>
[Insert the final answer based on the reasoning process]
</Solution>


Return Format:
```xml
<think></think>
<solution></solution>
```

Return a complete data sample in the above format using XML tags and markdown."""