
# 使用百度开源的 PaddleSpeech

from paddlespeech.cli.tts.infer import TTSExecutor

tts_executor = TTSExecutor()

text = "今天天气十分不错，百度也能做语音合成。"
# 默认只支持中文
# text = "今天天气十分不错，Paddle Speech也能做语音合成。"

output_file = "./data/paddlespeech.wav"

tts_executor(text=text, output=output_file)


# !中英文混合的(有bug，需要对一下文档)
# from paddlespeech.cli.tts.infer import TTSExecutor

# tts_executor = TTSExecutor()

# text = "早上好, how are you? 百度Paddle Speech一样能做中英文混合的语音合成。"

# output_file = "./data/paddlespeech_mix.wav"

# # mix，代表这个模型是支持中英文混合生成的
# tts_executor(
#     text=text, 
#     output=output_file, 
#     # 也是基于 Transformer 的 fastspeech2 模型
#     am="fastspeech2_mix", 
#     voc="hifigan_csmsc", 
#     lang="mix", 
#     spk_id=174
# )


import wave
import pyaudio

def play_wav_audio(wav_file):
    # open the wave file
    wf = wave.open(wav_file, 'rb')

    # instantiate PyAudio
    p = pyaudio.PyAudio()

    # open a stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data from the wave file and play it
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

play_wav_audio(output_file)