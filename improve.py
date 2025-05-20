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
def extract_scores(text):
  patterns = {
      "Question Score": r"Question Score:\s*(\d+)",
      "Answer Score": r"Answer Score:\s*(\d+)"
  }
  scores = {}
  for key, pattern in patterns.items():
      match = re.search(pattern, text)
      if match:
          scores[key] = int(match.group(1))
      else:
          scores[key] = None
  return scores

IMPROVE_ANSWER_TEMPLATE = """
hi, I need your help to improve my detailed, multi-step answer based on the provided context and comments.

<questions>{question}</questions>

<context>{context}</context>

<answer>{answer}</answer>

<comments>{comment}</comments>

Instructions:
- Improve the answer based on the comments, ensuring it fulfills these requirements:
  - Provide a comprehensive answer to each question, addressing all aspects and subcomponents of the question.
  - Ensure the answer is entirely derived from the information, data, and methods explicitly provided in the context, with no assumptions or information beyond the context.
  - Answer in a logical, multi-step manner, explicitly labeling each step (e.g., "Step 1: Extract relevant data," "Step 2: Analyze relationships"). Each step should:
    - Clearly explain the reasoning process, including why specific data or methods are used.
    - Reference specific parts of the context (e.g., tables, figures, sections) to support the analysis.
    - Build toward the final conclusion through analysis, synthesis, or inference.
  - Ensure the steps demonstrate complex reasoning, such as comparing multiple data points, interpreting methodological implications, or deriving conclusions from results.
  - Do not include content unrelated to answering the question or unsupported by the context.
- No other explanations for the improvement are needed.
- Enclose each answer within <answer></answer> tags.
"""

IMPROVE_ANSWER_SYSTEM_MESSAGE = """
You are an expert at improving  logical, multi-step reasoning and comprehensive answers to complex questions based on the given context and comments. 
Your task is to refine the given answer using the feedback from the comments, keeping the same format and structure.
"""
if __name__ == "__main__":
    for i in tqdm(range(1,4)):
        file_index =i
        with open(f"./data/{file_index}/data.json","r",encoding="utf-8") as f:
            data=json.load(f)

        with open(f"./data/{file_index}/paper.txt","r",encoding="utf-8") as f :
            context=f.read()

        for j in tqdm(range(0,3)):
            score = extract_scores(data[j]["evaluation"])
            answer_score = score["Answer Score"]
            if answer_score < 9:
                prompt = IMPROVE_ANSWER_TEMPLATE.format(
                question=data[j]["question"],
                context=context,
                answer=data[j]["answer"],
                comment=data[j]["evaluation"]
                )
                new_answer = gpt_chat(
                    sys=IMPROVE_ANSWER_SYSTEM_MESSAGE,
                    user=prompt,
                    provider="deepseek"
                )
                new_answer = extract_html_tags(new_answer, ["answer"])
                data[j]["answer_1"] = new_answer["answer"][0]

        with open(f"./data/{file_index}/data.json","w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


