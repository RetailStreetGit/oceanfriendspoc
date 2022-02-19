import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from flask import Flask, jsonify
from scipy.spatial.distance import cdist
import time
app = Flask(__name__)

openai.api_key = "XXX"
print('about to read the corpus embedding')
df = pd.read_csv('clinicaltrialsembedding.csv')
corpusEmbedding = np.zeros([1285,12288 ])
df['davinci_search'] = df.davinci_search.apply(eval).apply(np.array)
index = 0
for x in df['davinci_search']:
    corpusEmbedding[index] = np.array(x)
    index = index + 1
print('built corpus embedding for vector eval')
# df = "test manish"

def search_matches( product_description, n=3, pprint=True):
    openai.api_key = "XXXXXXXXXXXXXXXXX"
    embedding = get_embedding(product_description, engine='text-search-davinci-query-001-')
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

def search_matchesNew( product_description, n=3, pprint=True):
    openai.api_key = "hhhhhhh"
    embedding = get_embedding(product_description, engine='text-search-davinci-query-001-')
    inputEmbedding = np.array(embedding)
    inputEmbedding = inputEmbedding.reshape([1,12288])

    Distance = cdist(corpusEmbedding,inputEmbedding,'cosine')
    DistanceArgSorted = np.argsort(Distance,axis=0)[0:n]


    result = []
    for x in DistanceArgSorted:
        result.append({'NCTNumber': df.iloc[x.item()]['NCT Number'],'distance':Distance[x.item()].item()})
    
    return result

@app.route('/<string:searchString>/')
def index(searchString):
    # return f"test123"
    result = search_matches(searchString,3,True)
    return jsonify(result)

@app.route('/<int:numberOfResults>/<string:searchString>/')
def index1(numberOfResults,searchString):
    # return f"test123"
    result = search_matchesNew(searchString,numberOfResults,True)
    return jsonify(result)

if __name__ == '__main__':
   app.run()
#    print(df['davinci_search'].shape)
#    myarray = np.array(df['davinci_search'])
#    print(myarray.shape)
#    openai.api_key = " J"
#    start = time.time()

#    embedding = get_embedding("laugh hysterically", engine='text-search-davinci-query-001-')
#    inputEmbedding = np.array(embedding)
#    inputEmbedding = np.array(embedding)
   
#    inputEmbedding = inputEmbedding.reshape([1,12288])
#    Distance = cdist(corpusEmbedding,inputEmbedding,'cosine')
        

#    DistanceArgSorted = np.argsort(Distance,axis=0)[0:3]

        
#    outputResult = {}
#    end = time.time()
#    print("The time of execution of above program is :", end-start)
#    print(df.iloc[DistanceArgSorted[0]]['NCT Number'])
#    result = []
#    for x in DistanceArgSorted:
#     print(f"index is {x.item()}")
#     print(f"NCt number is {df.iloc[x.item()]['NCT Number']}")
#     result.append({'NCTNumber': df.iloc[x.item()]['NCT Number'],'similarity':Distance[x.item()].item()})
#    print (result)

