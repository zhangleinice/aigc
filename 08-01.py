import openai
import pandas as pd
import os
from IPython.display import display

openai.api_key = os.getenv("OPENAI_API_KEY")


# list all open ai models
engines = openai.Engine.list()
pd = pd.DataFrame(openai.Engine.list()['data'])
# display(pd[['id', 'owner']])



from openai.embeddings_utils import get_embedding

text = "让我们来算算Embedding"

embedding_ada = get_embedding(text, engine="text-embedding-ada-002")
print("embedding-ada: ", len(embedding_ada))

similarity_ada = get_embedding(text, engine="text-similarity-ada-001")
print("similarity-ada: ", len(similarity_ada))

babbage_similarity = get_embedding(text, engine="babbage-similarity")
print("babbage-similarity: ", len(babbage_similarity))

babbage_search_query = get_embedding(text, engine="text-search-babbage-query-001")
print("search-babbage-query: ", len(babbage_search_query))

curie = get_embedding(text, engine="curie-similarity")
print("curie-similarity: ", len(curie))

davinci = get_embedding(text, engine="text-similarity-davinci-001")
print("davinci-similarity: ", len(davinci))