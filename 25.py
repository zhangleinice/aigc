import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import numpy as np
import matplotlib.pyplot as plt
from diffusers.utils import load_image
from PIL import Image

# 从链接加载原始图像
image_file = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
original_image = load_image(image_file)

# 定义边缘检测函数


def get_canny_image(original_image, low_threshold=100, high_threshold=200):
    # 将原始图像转换为NumPy数组
    image = np.array(original_image)

    # 使用Canny边缘检测算法
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    # 将NumPy数组转换回PIL图像
    canny_image = Image.fromarray(image)
    return canny_image


# 获取边缘检测后的图像
canny_image = get_canny_image(original_image)

# 显示原始图像和边缘检测后的图像


def display_images(image1, image2):
    # 横向组合两幅图像
    combined_image = Image.new(
        'RGB', (image1.width + image2.width, max(image1.height, image2.height)))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    # 显示组合后的图像
    plt.imshow(combined_image)
    plt.axis('off')
    # 服务器下无法展示,保存图像到文件
    plt.savefig("./data/sd/ouput_image_vermeer.png")
    plt.show()


# 显示图像
display_images(original_image, canny_image)


# 加载预训练的 ControlNetModel 模型
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

# 使用预训练模型构建稳定扩散控制网络管道
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

# 启用模型的 CPU 卸载优化
pipe.enable_model_cpu_offload()

# 启用内存高效的注意力机制优化
pipe.enable_xformers_memory_efficient_attention()


prompt = ", best quality, extremely detailed"
prompt = [t + prompt for t in ["Audrey Hepburn",
                               "Elizabeth Taylor", "Scarlett Johansson", "Taylor Swift"]]
generator = [torch.Generator(device="cpu").manual_seed(42)
             for i in range(len(prompt))]

# 在管道中执行图像生成和控制任务
output = pipe(
    prompt,
    canny_image,
    negative_prompt=[
        "monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    num_inference_steps=20,
    generator=generator,
)


def draw_image_grids(images, rows, cols):
    # 创建 rows x cols 的网格以显示图像
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for row in range(rows):
        for col in range(cols):
            axes[row, col].imshow(images[col + row * cols])
    for ax in axes.flatten():
        ax.axis('off')
    # 显示图像网格
    plt.savefig("./data/sd/ouput_image_4.png")
    plt.show()


# 调用函数显示生成的图像网格
draw_image_grids(output.images, 2, 2)
