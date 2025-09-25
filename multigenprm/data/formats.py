CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\n'}}{% endif %}"""

def get_category_name(category: str) -> str:
    if category == "prm800k" or \
        category == "gsm8k" or \
            category == "math" or \
                category == "olympiadbench" or \
                    category == "omnimath":
        category_name = "math"
    elif category == "computer_science":
        category_name = "computer science"
    elif category == "other":
        category_name = ""
    else:
        category_name = category
    return category_name

def REFORMAT_PROMPT_FORMAT(category_name: str, question: str, steps: list[str]) -> str:
    prefix = " ".join(steps)    
    return (
        f"You will be presented with a solution to a {category_name} problem. "
        "Unfortunately, the solution lacks proper paragraphing, making it hard to read. "
        "Your task is to improve readability by reformatting the solution into well-structured paragraphs. "
        "Follow these specific guidelines:\n\n"

        "* Insert \\n\\n for paragraph breaks within the original solution. Do **NOT** alter any content of the original solution "
        "(the only exception is for itemized lists; see below).\n\n"

        "  - Each paragraph should represent a distinct, concise reasoning step that logically advances the solution.\n"
        "  - Reasoning steps can include case discussions, formula simplifications, or formula derivations. Each of these should be treated as an individual reasoning step and paragraphed accordingly.\n"
        "  - If an introductory analysis exists in the original solution, treat it as an initial reasoning step and place it as the first paragraph.\n"
        "  - Do **NOT** place any mathematical formulas in their own separate paragraphs; instead, include them within the same paragraph as the preceding text to form a cohesive reasoning step.\n"
        "  - The last paragraph should be \"The answer is (X).\" where X is an alphabet.\n\n"

        "* For any itemized lists (ordered or unordered), convert them into a written format, such as \"First/Second/Third.\" This is the **ONLY** content modification allowed.\n\n"

        "* Avoid making paragraphs too lengthy, as long paragraphs might contain multiple reasoning steps that should be paragraphed separately.\n\n"

        "* Disregard the accuracy of the solution content. Do **NOT** alter any of the original solution's content; focus solely on structuring it into logical, readable paragraphs.\n\n"

        "* Reply with the reformatted solution directly.\n\n"

        "--------------------------------------------------\n\n"
        f"[{category_name.capitalize()} Problem]\n\n{question.strip()}\n\n"
        f"[Solution]\n\n{prefix.strip()}"
    )
    
def DATA_PRM_PROMPT_FORMAT(category: str, question: str, steps: list[str]) -> str:
    category_name = get_category_name(category)
    steps = [ f"Step {str(i+1)}: {step}" for i, step in enumerate(steps) ]
    prefix = "\n".join(steps)
    return (
        f"You are given a {category_name} problem and a proposed multiple-step solution (with a step on each line):\n\n"
        f"[{category_name.capitalize()} Problem]\n{question}\n\n"
        f"[Solution]\n{prefix}\n\n"
        "Review and critique the proposed solution steps and determine whether each step is correct. If the solution is incomplete, only critique the steps that are provided. Your output must be in the following format:\n\n"
        "Step 1: The step is \\boxed{correct/incorrect}\n"
        "Step 2: The step is \\boxed{correct/incorrect}\n"
        "...\n"
        "Step n: The step is \\boxed{correct/incorrect}\n\n"
        "Once you find an incorrect step, you should stop since you do not need to analyze the remaining steps. If the solution is incomplete, only verify the provided steps."
    )
    
def PRM_PROMPT_FORMAT(category: str, question: str, steps: list[str], step_index: bool = True) -> str:
    category_name = get_category_name(category)
    if step_index:
        steps = [ f"Step {str(i+1)}: {step}" for i, step in enumerate(steps) ]
    prefix = "\n".join(steps)
    return (
        f"You are given a {category_name} problem and a proposed step-by-step solution:\n\n"
        f"[{category_name.capitalize()} Problem]\n{question}\n\n"
        f"[Solution]\n{prefix}\n\n"
        "Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps."
    )
    
def ORM_PROMPT_FORMAT(category: str, question: str, steps: list[str]) -> str:
    category_name = get_category_name(category)
    # steps = [ f"Step {str(i+1)}: {step}" for i, step in enumerate(steps) ]
    prefix = "\n\n".join(steps)
    return (
        f"You are a {category_name} teacher. Grade the solution, verifying correctness step by step.\n"
        "At the end of Solution verification, when you give your final grade, write it in the form \"Verification: Is the answer correct (Yes/No)? X\", where X is either Yes or No.\n\n"
        f"[{category_name.capitalize()} Problem]\n{question.strip()}\n\n"
        f"[Solution]\n{prefix.strip()}\n"        
    )