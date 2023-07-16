# 导入所需的模块和类
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR, ImageParser
from llama_index.response.notebook_utils import display_response, display_image
from llama_index.indices.query.query_transform.base import ImageOutputQueryTransform

# 创建一个图片解析器，用于从图片中提取文本内容
image_parser = ImageParser(keep_image=True, parse_text=True)

# 创建文件提取器，并添加图片解析器用于处理图像文件
file_extractor = DEFAULT_FILE_EXTRACTOR
file_extractor.update({
    ".jpg": image_parser,
    ".png": image_parser,
    ".jpeg": image_parser,
})

# 定义一个函数，用于将文件名作为元数据添加到文档中
filename_fn = lambda filename: {'file_name': filename}

# 创建一个SimpleDirectoryReader对象来读取收据数据目录中的文件，并应用文件提取器和文件名元数据
receipt_reader = SimpleDirectoryReader(
    input_dir='./data/receipts',
    file_extractor=file_extractor,
    file_metadata=filename_fn,
)

# 加载收据文档数据
receipt_documents = receipt_reader.load_data()



receipts_index = GPTVectorStoreIndex.from_documents(receipt_documents)
receipts_response = receipts_index.query(
    'When was the last time I went to McDonald\'s and how much did I spend. \
    Also show me the receipt from my visit.',
    query_transform=ImageOutputQueryTransform(width=400)
)

display_response(receipts_response)