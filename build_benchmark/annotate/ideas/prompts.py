def extract_core_idea_query(statement: str, solution: str) -> str:
    return (
        """# INSTRUCTION
You are an expert mathematical anotator tasked with identifying the *core idea* - the central mathematical insight, from a math problem.

# GOALS
1. Identify what makes the solution work conceptually, not how to carry it out. Capture the untrial step or idea that is the greatest hint for the solution.
2. Never include any multi-step reasoning, equations or numeric computations. Don't include any anotations that are in the solution, but not in the original problem statement.
3. Never try to solve the problem on your own and don't include your reasoning or thoughs.
4. Output a sinlge valid JSON object matching the schema below.
5. Structure the ideas imperatively so they look like you are giving a hint to someone.
5. If the problem seems too easy, straightforward or you can't identify a core idea, store its value as 'null' and set the 'noCoreIdea' to 'true'

# SCHEMA
```json
{
    "noCoreIdea": <true|false>,
    "coreIdea": "<string - one short sentence (up to 30 words) naming the main insight to the problem>",
    "supporingIdeas: ["<strings - 0-3 short techniques phrases>"],
    "keywords": ["<strings - 1-2 word phrases summarizing the ideas, theorems, etc. in the solution"],
    "confidence": <0.0-1.0>
}

Here are the problem statement and solution:

Statement: {"""
        + statement
        + """}

Solution: {"""
        + solution
        + """}"""
    )


CORE_IDEA_SOLVE = """You are an Expert Mathematician tasked with solving a problem by following a hint. Please reason step by step and put your final answer in \\boxed{{}}.

Here is the problem to solve:
{statement}

To solve the problem use this hint: {idea}.

Your solution: """
