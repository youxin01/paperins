from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import os
import re
from openai import OpenAI
import json
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def clean_pdfmarkdown(text):
    # 保留换行符拆行
    lines = text.splitlines(keepends=True)

    # 删除 Markdown 和 HTML 图片
    image_pattern = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>", re.IGNORECASE)
    lines = [image_pattern.sub('', line) for line in lines]

    # 找到第一个标题中包含 appendix/references 的行，截断其之前内容
    cutoff_index = len(lines)
    for i, line in enumerate(lines):
        if '#' in line and any(keyword in line.lower() for keyword in ['appendix', 'references']):
            cutoff_index = i
            break

    cleaned_lines = lines[:cutoff_index]

    return ''.join(cleaned_lines)  # 不做换行或压缩，保持原始结构

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

def _validate_tasks(out):
    questions = extract_html_tags(out, ["question"])
    if "question" not in questions:
        return (
            out,
            False,
            f"Error: you did not generate questions within the <question></question> tags",
        )
    questions = questions["question"]
    # Check that there are at most max_questions questions
    if len(questions) > 3:
        return (
            out,
            False,
            f"Error: you can only ask at most {3} questions, but you asked {len(questions)}.",
        )

    return (questions, out), True, ""

GET_QUESTIONS_TEMPLATE ="""
Hello, I need your expertise to generate a set of complex, thought-provoking questions based solely on the provided context.

<context>{context}</context>

Instructions:
* Generate a list of questions that are entirely derived from the information, data, and methods explicitly provided in the context.
* Ensure that each question requires no assumptions or information beyond what is stated in the context.
* Each question must be complex, requiring multiple steps of analysis, synthesis, or inference to answer, such as comparing data across tables, interpreting methodological implications, or deriving conclusions from results.
* Enclose each question within <question></question> tags.
* Do not number the questions.
* Generate up to {max_questions} questions.
"""

GET_QUESTIONS_SYSTEM_MESSAGE = """
You are a distinguished professor with expertise in analyzing academic papers and formulating rigorous, context-bound questions. 
Your role is to generate questions that are intellectually challenging, requiring deep analysis of the provided context alone. 
You must not rely on external knowledge or make assumptions beyond the explicit content of the context. 
Your primary objective is to create questions that demand multiple steps of reasoning, while ensuring all questions are fully grounded in the context.
"""
GET_ANSWER_TEMPLATE = """
Hello, I need your expertise to provide detailed, multi-step answers to the questions based solely on the provided context, showcasing a clear and logical reasoning process.

<questions>{questions}</questions>

<context>{context}</context>

Instructions:
* Provide a comprehensive answer to each question, addressing all aspects and subcomponents of the question.
* Ensure the answer is entirely derived from the information, data, and methods explicitly provided in the context, with no assumptions or information beyond the context.
* Answer in a logical, multi-step manner, explicitly labeling each step (e.g., "Step 1: Extract relevant data," "Step 2: Analyze relationships"). Each step should:
  - Clearly explain the reasoning process, including why specific data or methods are used.
  - Reference specific parts of the context (e.g., tables, figures, sections) to support the analysis.
  - Build toward the final conclusion through analysis, synthesis, or inference.
* Ensure the steps demonstrate complex reasoning, such as comparing multiple data points, interpreting methodological implications, or deriving conclusions from results.
* Do not include content unrelated to answering the question or unsupported by the context.
* Answer questions in the same order they were asked.
* Enclose each answer within <answer></answer> tags.
* Do not number the answers.
"""

GET_ANSWER_SYSTEM_MESSAGE = """
You are an expert in providing logical, multi-step reasoning and detailed answers to complex questions based solely on given contexts. 
Your role is to thoroughly analyze the questions and context, generating answers that are accurate, comprehensive, and strictly derived from the context without any external assumptions. 
Your answers must showcase a clear reasoning process, with each step explicitly labeled and explained, demonstrating complex analysis such as synthesizing data from tables or figures, evaluating methodological implications, or deriving insights from results. 
Your goal is to deliver answers that are logical, well-structured, transparent in their reasoning, and easy to follow, ensuring the inference process is fully visible and grounded in the context.
"""

if __name__ == "__main__":
    converter = PdfConverter(artifact_dict=create_model_dict())
    for i in tqdm(range(1, 2)):
        directory = f"./data/{i}"
        # step1
        print(directory)
        pdf = [i for i in os.listdir(directory) if i.endswith(".pdf")]
        pdf_path = os.path.join(directory, pdf[0])
        rendered = converter(pdf_path)
        text, _, images = text_from_rendered(rendered)

        texts = text
        text1 = clean_pdfmarkdown(text)
        with open(os.path.join(directory, "paper.txt"), "w", encoding="utf-8") as f:
            f.write(text1)

        max_questions = 3
        context = GET_QUESTIONS_TEMPLATE.format(
            context=text1,
            max_questions=max_questions
        )
        questions = gpt_chat(
            sys=GET_QUESTIONS_SYSTEM_MESSAGE,
            user=context,
            provider="deepseek"
        )
        a= _validate_tasks(questions)

        # step2
        qs=""
        for idx in range(len(a[0][0])):
            qs += f"{idx+1}. {a[0][0][idx]}\n"

        qs_context = GET_ANSWER_TEMPLATE.format(
        questions=qs,
        context=text1
        )
        answer = gpt_chat(sys=GET_ANSWER_SYSTEM_MESSAGE, user=qs_context, provider="deepseek")
        final = extract_html_tags(answer, ["answer"])
        qs_ans=[]

        for idx in range(len(a[0][0])):
            tmp ={}
            tmp["question"] = a[0][0][idx]
            # tmp["answer"] = re.sub(r'\b\d+\.\s*', '',final["answer"][idx])
            tmp["answer"] = final["answer"][idx]
            qs_ans.append(tmp)
        with open(os.path.join(directory,"data.json"), "w", encoding="utf-8") as f:
            json.dump(qs_ans, f, ensure_ascii=False, indent=4)
