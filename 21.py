
# 导入所需的库
import openai, os
import gradio as gr
import azure.cognitiveservices.speech as speechsdk
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

# 设置 OpenAI 和 Azure 语音服务的 API 密钥和配置信息
openai.api_key = os.environ["OPENAI_API_KEY"]
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_SPEECH_KEY'), region=os.environ.get('AZURE_SPEECH_REGION'))

# 创建一个对话记忆(memory)对象，用于记录对话的历史
memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048)

# 创建一个 ConversationChain 对象，用于构建聊天对话链
conversation = ConversationChain(
    llm=OpenAI(max_tokens=2048, temperature=0.5), 
    memory=memory,
)

# 设置语音合成的配置信息，包括语音合成语言和声音名称
speech_config.speech_synthesis_language='zh-CN'
speech_config.speech_synthesis_voice_name='zh-CN-XiaohanNeural'

# 创建 SpeechSynthesizer 对象，用于语音合成
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# 定义函数 play_voice，用于播放文本的语音合成
def play_voice(text):
    speech_synthesizer.speak_text_async(text)

# 定义一个函数用于预测对话
def predict(input, history=[]):
    history.append(input)
    response = conversation.predict(input=input)
    history.append(response)
    # history[::2]: 从history列表中取出所有的偶数索引位置的元素。也就是取出history列表中所有的用户输入内容。
    # history[1::2]: 从history列表中取出所有的奇数索引位置的元素。也就是取出history列表中所有的聊天机器人的回复。
    # 通过zip()函数，将用户输入和聊天机器人的回复一一对应，组成一个元组，并将所有这些元组组成一个新的列表responses
    responses = [(u,b) for u,b in zip(history[::2], history[1::2])]
    # 形成一个元组将用户输入和机器人回复一一对应
    # responses: [('用户输入1', '聊天机器人回复1'), ('用户输入2', '聊天机器人回复2'), ...]
    return responses, history

# 定义函数 transcribe，用于将音频文件转录为文本
def transcribe(audio):
    os.rename(audio, audio + '.wav')
    audio_file = open(audio + '.wav', "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']    

# 定义函数 process_audio，用于处理音频输入并进行预测
def process_audio(audio, history=[]):
    text = transcribe(audio)
    return predict(text, history)

# 创建 Gradio 的聊天界面，允许用户通过文本输入或录音进行对话
with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="输入文本并按下回车键").style(container=False)
        
    with gr.Row():
        audio = gr.Audio(source="microphone", type="filepath")
        
    txt.submit(predict, [txt, state], [chatbot, state])
    audio.change(process_audio, [audio, state], [chatbot, state])

# 启动 Gradio 的应用界面
demo.launch()

