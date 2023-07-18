import openai, os

openai.api_key = os.environ.get("OPENAI_API_KEY")

# 使用 GPT-3.5 进行文本生成
def gpt35(prompt, model="text-davinci-002", temperature=0.4, max_tokens=1000, 
          top_p=1, stop=["\n\n", "\n\t\n", "\n    \n"]):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop
    )
    message = response["choices"][0]["text"]
    return message

code = """
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    elif minutes > 0:
        return f"{minutes}min{seconds}s"
    else:
        return f"{seconds}s"
"""

# 1. 解释代码的函数
def explain_code(function_to_test, unit_test_package="pytest"):
    prompt = f"""
# 使用 {unit_test_package} 编写优质单元测试的教程

在这个高级教程中，我们将使用 Python 3.10 和 `{unit_test_package}` 来编写一套单元测试，以验证以下函数的行为。
```python
{function_to_test}

在编写任何单元测试之前，让我们先查看一下函数的每个元素具体是做什么的，以及作者的意图可能是什么。
- 首先,"""
    response = gpt35(prompt)
    return response, prompt

# 代码解释，prompt
code_explanation, prompt_to_explain_code = explain_code(code)

# 2. 根据代码解释 + 用例要求 => 生成测试计划的函数
def generate_a_test_plan(full_code_explanation, unit_test_package="pytest"):
    prompt_to_explain_a_plan = f"""
  
优质的单元测试套件应该目标在：
- 测试函数在广泛可能输入下的行为
- 测试作者可能未预料到的边界情况
- 充分利用 `{unit_test_package}` 的特性，使测试容易编写和维护
- 容易阅读和理解，代码整洁，名称描述性
- 具有确定性，以便测试始终以相同的方式通过或失败

`{unit_test_package}` 提供了许多便利功能，使编写和维护单元测试变得简单。我们将使用它们来编写该函数的单元测试。

对于这个特定的函数，我们希望我们的单元测试能处理以下多样化的情况（在每个情况下，我们列出了一些示例）：
-"""
    prompt = full_code_explanation + prompt_to_explain_a_plan
    response = gpt35(prompt)
    return response, prompt

test_plan, prompt_to_get_test_plan = generate_a_test_plan(prompt_to_explain_code + code_explanation)

# print(test_plan)

# 3. 额外测试计划，如果需要的话
# not_enough_test_plan = """The function is called with a valid number of seconds
#     - `format_time(1)` should return `"1s"`
#     - `format_time(59)` should return `"59s"`
#     - `format_time(60)` should return `"1min"`
# """

# approx_min_cases_to_cover = 7
# elaboration_needed = test_plan.count("\n-") +1 < approx_min_cases_to_cover 
# if elaboration_needed:
#         prompt_to_elaborate_on_the_plan = f"""

# In addition to the scenarios above, we'll also want to make sure we don't forget to test rare or unexpected edge cases (and under each edge case, we include a few examples as sub-bullets):
# -"""
#         more_test_plan, prompt_to_get_test_plan = generate_a_test_plan(prompt_to_explain_code + code_explaination + not_enough_test_plan + prompt_to_elaborate_on_the_plan)
#         print(more_test_plan)

# 4. 生成测试用例的函数
def generate_test_cases(function_to_test, unit_test_package="pytest"):
    starter_comment = "下面，每个测试案例由传递给 @pytest.mark.parametrize 装饰器的元组表示"
    prompt_to_generate_the_unit_test = f"""

在进入各个测试之前，让我们首先将完整的单元测试套件作为一个整体来查看。我们已添加了有助于解释每行代码含义的注释。
```python
import {unit_test_package}  # 用于我们的单元测试

{function_to_test}

#{starter_comment}"""
    full_unit_test_prompt = prompt_to_explain_code + code_explanation + test_plan + prompt_to_generate_the_unit_test
    return gpt35(model="text-davinci-003", prompt=full_unit_test_prompt, stop="```"), prompt_to_generate_the_unit_test

unit_test_response, prompt_to_generate_the_unit_test = generate_test_cases(code)

# print(unit_test_response)

import ast

# 5.通过 AST 库进行语法检查
code_start_index = prompt_to_generate_the_unit_test.find("```python\n") + len("```python\n")
code_output = prompt_to_generate_the_unit_test[code_start_index:] + unit_test_response
try:
    ast.parse(code_output)
except SyntaxError as e:
    print(f"在生成的代码中有语法错误：{e}")

print(code_output)

#! 自动生成文件
