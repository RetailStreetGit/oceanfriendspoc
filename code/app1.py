import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from flask import Flask, jsonify

def BuildCorpusEmbedding(inputDF):
    openai.api_key = "XXXX"
    inputDF['davinci_similarity'] = inputDF["textToIndex"].apply(lambda x: get_embedding(x, engine='text-similarity-davinci-001-msft'))
    inputDF['davinci_search'] = inputDF["textToIndex"].apply(lambda x: get_embedding(x, engine='text-search-davinci-doc-001-msft'))
    inputDF.to_csv('./gpt3semanticsearch/code/clinicaltrialsembedding.csv')

def search_matches( product_description, n=3, pprint=True):
    openai.api_key = "XXX"
    embedding = get_embedding(product_description, engine='text-search-davinci-query-001')
    df['similarities'] = df['davinci_search'].apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)[['NCT Number','similarities']]
    if pprint:
        for r in res:
            print(r[:200])
            print()
    result = []
    for index, row in res.iterrows():
        result.append({'NCTNumber': row['NCT Number'],'similarity':row['similarities']})
    
    return result

app = Flask(__name__)
@app.route('/<string:searchString>/')
def hello(searchString):
    # result = search_matches(searchString,3,True)
    # return jsonify(result)
    return f"manish{searchString}"





# inputDF = pd.read_csv('./gpt3semanticsearch/code/clinicaltrials.csv')
# inputDF["textToIndex"] = inputDF['Inclusion Criteria'].apply(lambda x: str(x).replace('\n','.'))
# data =  openai.Engine.list()['data']
# print(data)
# BuildCorpusEmbedding(inputDF)
global df
print('about to read the corpus embedding')
# df = pd.read_csv('clinicaltrialsembedding.csv')
# df['davinci_search'] = df.davinci_search.apply(eval).apply(np.array)
print('about to start the app')
# app.run()
print('started flask app')


    

if __name__ == '__main__':
   app.run()