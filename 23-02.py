import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from IPython.display import display
from IPython.display import Image as IPyImage


# 加载数据集
dataset = load_dataset("rajuptvs/ecommerce_products_clip")
training_split = dataset["train"]

# 定义显示图像的函数


def display_images(images):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


# 校验数据集， 获取数据集中的前10张图像并显示
images = [example["image"] for example in training_split.select(range(10))]
# display_images(images)


# 初始化 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 获取图像的特征向量


def get_image_features(image):
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs.to(device)
        features = model.get_image_features(**inputs)
    return features.cpu().numpy()

# 为数据集中的每个示例添加图像特征向量


def add_image_features(example):
    example["features"] = get_image_features(example["image"])
    return example


# 对数据集应用函数以添加图像特征向量
training_split = training_split.map(add_image_features)

# 构建特征矩阵并创建 Faiss 索引
features = [example["features"] for example in training_split]
features_matrix = np.vstack(features)
dimension = features_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(features_matrix.astype('float32'))

# 获取文本的特征向量


def get_text_features(text):
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs.to(device)
        features = model.get_text_features(**inputs)
    return features.cpu().numpy()


# 显示搜索结果
def display_search_results(results):
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    axes = axes.ravel()

    for idx, result in enumerate(results):
        axes[idx].imshow(result["image"])
        axes[idx].set_title(f"Distance: {result['distance']:.2f}")
        axes[idx].axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


# 图片搜索
def search_image(image_path, top_k=5):
    # Load the image from the file
    image = Image.open(image_path).convert("RGB")

    # Get the image feature vector for the input image
    image_features = get_image_features(image)

    # Perform a search using the FAISS index
    distances, indices = index.search(image_features.astype("float32"), top_k)

    # Get the corresponding images and distances
    results = [
        {"image": training_split[i.item()]["image"],
         "distance": distances[0][j]}
        for j, i in enumerate(indices[0])
    ]

    return results

# 文字搜索


def search_text(query_text, top_k=5):
    # 获取输入查询文本的特征向量
    text_features = get_text_features(query_text)

    # 使用 Faiss 索引进行搜索
    distances, indices = index.search(text_features.astype("float32"), top_k)

    # 获取相应的图像和距离
    results = [
        {"image": training_split[i.item()]["image"],
         "distance": distances[0][j]}
        for j, i in enumerate(indices[0])
    ]

    return results


image_path = "./data/shirt.png"
query_text = "A red dress"

# 搜索示例
# results = search_image(image_path)
# display(IPyImage(filename=image_path, width=300, height=200))
# display_search_results(results)

results = search_text(query_text)
display_search_results(results)
