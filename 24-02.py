from diffusers import StableDiffusionUpscalePipeline
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

image_file = "./data/sketch-mountains-input.jpg"

init_image = Image.open(image_file).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, image=init_image,
              strength=0.75, guidance_scale=7.5).images

display(init_image)
display(images[0])

# prompt = "ghibli style, a fantasy landscape with castles"
# images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

# display(init_image)
# display(images[0])


# 排除一些内容
# prompt = "ghibli style, a fantasy landscape with castles"
# negative_prompt = "river"
# images = pipe(prompt=prompt, negative_prompt=negative_prompt,
#               image=init_image, strength=0.75, guidance_scale=7.5).images

# display(images[0])


# 提高分辨率
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
low_res_img_file = "./data/low_res_cat.png"
low_res_img = Image.open(low_res_img_file).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

prompt = "a white cat"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]

low_res_img_resized = low_res_img.resize((512, 512))

display(low_res_img_resized)
display(upscaled_image)
