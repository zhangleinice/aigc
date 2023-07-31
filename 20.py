# 使用 Azure 云提供的 API
import os
import azure.cognitiveservices.speech as speechsdk

# 导入必要的库
# os 用于访问环境变量
# azure.cognitiveservices.speech 用于访问 Azure 认知服务的语音合成功能

# 通过环境变量获取 Azure 认知服务的订阅密钥和区域
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_SPEECH_KEY'), region=os.environ.get('AZURE_SPEECH_REGION'))

# 创建 AudioOutputConfig 对象
# 设置使用默认扬声器进行语音输出
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# 存储格式
# audio_config = speechsdk.audio.AudioOutputConfig(filename="./data/tts.wav")

# 设置合成语音的语言
# 这里将合成语音设置为中文普通话女声（zh-CN-XiaohanNeural）

speech_config.speech_synthesis_language='zh-CN'
speech_config.speech_synthesis_voice_name = 'zh-CN-XiaohanNeural'

# speech_config.speech_synthesis_voice_name='zh-CN-YunfengNeural'

# 创建 SpeechSynthesizer 对象
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)


# 定义要合成的文本内容
text = "今天天气真不错，ChatGPT真好用。"

# 合成语音并进行播放
# speech_synthesizer.speak_text(text)



# 使用 W3C 的标准，语音合成标记语言
ssml = """<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
    <voice name="zh-CN-YunyeNeural">
        儿子看见母亲走了过来，说到：
        <mstts:express-as role="Boy" style="cheerful">
            “妈妈，我想要买个新玩具”
        </mstts:express-as>
    </voice>
    <voice name="zh-CN-XiaomoNeural">
        母亲放下包，说：
        <mstts:express-as role="SeniorFemale" style="angry">
            “我看你长得像个玩具。”
        </mstts:express-as>
    </voice>
</speak>"""

speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()






