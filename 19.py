
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")

# audio_file= open("./data/podcast_clip.mp3", "rb")

# translated_prompt="""This is a podcast discussing ChatGPT and PaLM model. The full name of PaLM is Pathways Language Model."""

# transcript = openai.Audio.transcribe(
#     "whisper-1", 
#     audio_file,
#     response_format="json",
#     language="en",
#     # 引导模型更好地做语音识别,转英文等等
#     prompt=translated_prompt,
# )

# print(transcript['text'])




from pydub import AudioSegment

podcast = AudioSegment.from_mp3("./data/podcast_long.mp3")

# PyDub handles time in milliseconds
ten_minutes = 15 * 60 * 1000

total_length = len(podcast)

start = 0
index = 0
while start < total_length:
    end = start + ten_minutes
    if end < total_length:
        chunk = podcast[start:end]
    else:
        chunk = podcast[start:]
    with open(f"./data/podcast_clip_{index}.mp3", "wb") as f:
        chunk.export(f, format="mp3")
    start = end
    index += 1



prompt = "这是一段Onboard播客，里面会聊到ChatGPT以及PALM这个大语言模型。这个模型也叫做Pathways Language Model。"
for i in range(index):
    clip = f"./data/podcast_clip_{i}.mp3"
    audio_file= open(clip, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, 
                                     prompt=prompt)
    # mkdir ./data/transcripts if not exists
    if not os.path.exists("./data/transcripts"):
        os.makedirs("./data/transcripts")
    # write to file
    with open(f"./data/transcripts/podcast_clip_{i}.txt", "w") as f:
        f.write(transcript['text'])
    # get last sentence of the transcript
    sentences = transcript['text'].split("。")
    prompt = sentences[-1]



