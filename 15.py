
# 获取实时天气

# 通过 RequestsChain 获取实时外部信息
# TransformationChain


from langchain.chains import LLMRequestsChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

template = """在 >>> 和 <<< 直接是来自Google的原始搜索结果.
请把对于问题 '{query}' 的答案从里面提取出来，如果里面没有相关信息的话就说 "找不到"
请使用如下格式：
Extracted:<answer or "找不到">
>>> {requests_result} <<<
Extracted:"""

PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)
requests_chain = LLMRequestsChain(llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=PROMPT))
question = "今天上海的天气怎么样？"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
}
result=requests_chain(inputs)
# print(result)
# print(result['output'])


# 通过 TransformationChain 转换数据格式
import re
def parse_weather_info(weather_info: str) -> dict:
    # 将天气信息拆分成不同部分
    parts = weather_info.split('. ')

    # 解析天气
    weather = parts[0].strip()

    # 初始化温度范围和风向风力为空字符串
    temperature_range = ""
    wind_direction = ""
    wind_force = ""

    # 解析温度范围，并提取最小和最大温度
    if len(parts) > 1:
        temperature_range = parts[1].strip().replace('℃', '')

    # 解析风向和风力
    if len(parts) > 2:
        wind_parts = parts[2].split(' ')
        if len(wind_parts) == 2:
            wind_direction = wind_parts[0].strip()
            wind_force = wind_parts[1].strip()

    # 返回解析后的天气信息字典，只包含存在的部分
    weather_dict = {
        'weather': weather,
        'temperature_range': temperature_range,
        'wind_direction': wind_direction,
        'wind_force': wind_force
    }

    return weather_dict


# weather_dict = parse_weather_info(' 中雨转小雨. 28/32℃')
# weather_dict = parse_weather_info(result['output'])
# print(weather_dict)

# 通过 TransformationChain 转换数据格式
from langchain.chains import TransformChain, SequentialChain

def transform_func(inputs: dict) -> dict:
    text = inputs["output"]
    return {"weather_info": parse_weather_info(text)}   

transformation_chain = TransformChain(
        input_variables=["output"], 
        output_variables=["weather_info"], 
        transform=transform_func
    )

final_chain = SequentialChain(
        chains=[requests_chain, transformation_chain], 
        input_variables=["query", "url"], 
        output_variables=["weather_info"]
    )

final_result = final_chain.run(inputs)

print(final_result)
