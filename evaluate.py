import os
import re
from openai import OpenAI
import json
from tqdm import tqdm

from openai import OpenAI
def extract_html_tags(text, keys):
    content_dict = {}
    keys = set(keys)
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        # print(matches)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict
def gpt_chat(sys: str, user: str, provider: str) -> str:
    config = {
        "doubao": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": "",
            "model": "doubao-1-5-pro-32k-250115"
        },
        "kimi": {
            "base_url": "https://api.moonshot.cn/v1",
            "api_key": "",
            "model": "moonshot-v1-32k"
        },
        "qwen72": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "model": "qwen2.5-72b-instruct"
        },
        "qwen32": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "model": "qwen2.5-32b-instruct"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/",
            "api_key": "",
            "model": "deepseek-chat"
        },
        "openai": {
            "base_url": "https://api.gptsapi.net/v1",
            "api_key": "",
            "model": "gpt-4o"
        }
    }

    if provider not in config:
        raise ValueError(f"Unknown provider: {provider}")

    cfg = config[provider]
    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"]
    )

    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    return completion.choices[0].message.content

GET_EVALUATE_TEMPLATE = """
Hello, I need your help to evaluate the <question> and <answer> based on the provided <context>, focusing on scoring and analyzing errors in the answer's steps with actionable improvement suggestions.

<context>
{context}
</context>

<question>
{question}
</question>

<answer>
{answer}
</answer>

Instructions:
- Evaluate the question and answer against the following strict requirements:
    1. The question must be entirely derived from the context's explicit information, data, or methods, with no external assumptions or information.
    2. The question must be complex, requiring multi-step analysis, synthesis, or inference (e.g., comparing data across sections, interpreting methods, or deriving conclusions from multiple data points).
    3. The answer must fully and accurately address the question using only context-provided information, demonstrating rigorous, logical, multi-step reasoning that matches the question's complexity.
- Assign two scores from 1 to 10:
  - **Question Score**: Reflects how well the question meets requirements 1 and 2.
    - 1 = Relies heavily on external assumptions or is unrelated to the context.
    - 5 = Partially context-based or insufficiently complex (e.g., requires only simple recall or single-step analysis).
    - 8 = Mostly context-based and complex, with minor reliance on assumptions or slightly insufficient depth.
    - 10 = Fully context-based, highly complex, requiring rigorous multi-step analysis or synthesis.
  - **Answer Score**: Reflects how accurately, completely, and rigorously the answer addresses the question per requirement 3.
    - 1 = Completely incorrect, irrelevant, or unrelated to the context/question.
    - 5 = Partially correct, missing key steps, or lacking rigorous multi-step reasoning.
    - 8 = Mostly correct with minor omissions, inaccuracies, or insufficient reasoning depth.
    - 10 = Fully correct, complete, context-supported, with clear, logical, multi-step reasoning.
- Provide a concise analysis of errors in the answer's steps:
  - Identify each steps that are incorrect, incomplete, or insufficiently rigorous (e.g., introducing external assumptions, misinterpreting data, omitting key context, or weak reasoning).
  - For each questionable steps, provide a clear, actionable improvement suggestion, specifying how to correct the step using context-specific information (e.g., "Use data from Table X to support the conclusion").
  - Do not analyze the question or provide detailed context quotes unless necessary to explain the improvement.
  - If the answer lacks explicit steps, note this as an error and suggest how to structure the reasoning.
- Enclose evaluation within <evaluation></evaluation> tags.
- Output format:
  - Question Score: [1-10]
  - Answer Score: [1-10]
  - Answer Error Analysis: [Description of errors in specific steps, followed by actionable improvement suggestions]
- Example Output:
  - Question Score: 5
  - Answer Score: 6
  - Answer Error Analysis: Step 2 introduces external irrigation methods not in the context; revise to use irrigation data from paragraph 3. Step 3 misses biodiversity data; include metrics from Table 4 to strengthen the conclusion.
"""

GET_EVALUATE_SYSTEM_MESSAGE = """
You are an expert in evaluating questions and answers based on provided contexts, focusing on scoring and analyzing errors in the answer's steps with actionable improvement suggestions.
You should make your output satisfy the output format strictly.
"""

if __name__ == "__main__":
    for i in range(1,4):
        file_index =i
        with open(f"./data/{file_index}/data.json","r",encoding="utf-8") as f:
            data=json.load(f)

        with open(f"./data/{file_index}/paper.txt","r",encoding="utf-8") as f :
            context=f.read()

        for j in range(0,3):
            if "answer_1" in data[j]:
                question_index = j
                p = GET_EVALUATE_TEMPLATE.format(
                    context=context,
                    question=data[j]["question"],
                    answer=data[j]["answer_1"]
                )
                eval = gpt_chat(
                    sys=GET_EVALUATE_SYSTEM_MESSAGE,
                    user=p,
                    provider="openai"
                )
                eval1 = extract_html_tags(eval, ["evaluation"])
                data[j]["evaluation_1"] = eval1["evaluation"][0]

        with open(f"./data/{file_index}/data.json","w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=4)


