from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()



# question = """
# 自收到商品之日起7天内，如产品未使用、包装完好，您可以申请退货。某些特殊商品可能不支持退货，请在购买前查看商品详情页面的退货政策。

# 根据以上信息，请回答下面的问题：

# Q: 你们的退货政策是怎么样的？
# """
# response, history = model.chat(tokenizer, question, history=[])
# print(response)


question = """
Q: 你们的退货政策是怎么样的？
A: 
"""
response, history = model.chat(tokenizer, question, history=[])
print(response)