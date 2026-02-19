import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""# Install lib and dependencies

## example 1
"""

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer("all-mpnet-base-v2")

seq1 = 'I was not unaware of the problem.'
seq2 = 'I had a very slight awareness of the problem.'
embedding1 = model.encode(seq1, convert_to_tensor=True)
embedding2 = model.encode(seq2, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
print("Sentence 1:", seq1)
print("Sentence 2:", seq2)
print("Similarity score:", cosine_scores.item())

'''
Sentence 1: I was not unaware of the problem.
Sentence 2: I had a very slight awareness of the problem.
Similarity score: 0.7190778255462646
'''

embeddings = model.encode([
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",])
similarities = model.similarity(embeddings, embeddings)
similarities
'''
tensor([[1.0000, 0.6817, 0.0492],
        [0.6817, 1.0000, 0.0421],
        [0.0492, 0.0421, 1.0000]])
        '''

# Step 1: Split text into sentences using spaCy
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(post)
sentences = [sent.text.strip() for sent in doc.sents]
sentences

# Step 2: Encode each sentence into embeddings
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = [model.encode(sentence, convert_to_tensor=True) for sentence in sentences]

# Step 3: Compute cosine similarities for consecutive sentence pairs
cosine_similarities = []
for i in range(len(embeddings) - 1):
    sim = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1]).item()
    cosine_similarities.append(sim)

# Step 4: Calculate the average similarity
average_similarity = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else None

# Output results
print("Sentences:", sentences)
print("Cosine Similarities:", cosine_similarities)
print("Average Similarity:", average_similarity)

"""# define a function"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

def process_text_similarity(data, text_column='clean_text'):
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data) # Convert data to DataFrame if it's not

    # Step 1: Split text into sentences
    def split_into_sentences(text):
        return [sent.text.strip() for sent in nlp(text).sents]

    data['sent_list'] = data[text_column].apply(split_into_sentences)

    # Step 2: Encode sentences into embeddings
    model = SentenceTransformer("all-mpnet-base-v2")
    data['sentence_embeddings'] = data['sent_list'].apply(
        lambda sentences: [model.encode(sentence, convert_to_tensor=True) for sentence in sentences]
    )

    # Step 3: Compute pairwise cosine similarities
    def compute_pairwise_similarity(embeddings):
        if len(embeddings) < 2:
            return []  # No pairs to compare if fewer than 2 sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1]).item()
            similarities.append(sim)
        return similarities

    data['pair_similarity'] = data['sentence_embeddings'].apply(compute_pairwise_similarity)

    # Step 4: Calculate the average similarity
    data['average_similarity'] = data['pair_similarity'].apply(
        lambda sims: np.mean(sims) if sims else None
    )

    return data

import pandas as pd

data = {
    'clean_text': [
        "The weather is nice. The sun is shining.",
        "I love programming. It's so much fun.",
        "This is an example text."
    ]
}

df = pd.DataFrame(data)

processed_df = process_text_similarity(df, text_column='clean_text')
processed_df

"""# ad (clinical group)"""

df = pd.read_csv(result + 'data.csv',engine='python')
df

import re
def clean_punctuation_spacing(text):
    text = re.sub(r'\s+([.,!?;])', r'\1', text)
    text = text.replace('?', '.').replace('!', '.')
    return text

df['clean_text'] = df['clean_text'].apply(clean_punctuation_spacing)
df

data = df[['idx', 'PAR', 'clean_text']]
data.head()

data.shape

df1 = process_text_similarity(data, text_column='clean_text')
df1.head()

df['average_similarity'] = df1['average_similarity']
df

# save the results to a new cvs. file
df.to_csv(brief + 'data.csv', index=False)

"""# health (clinical group)"""

health = pd.read_csv(result + 'data.csv',engine='python')
health.head()

data2 = health[['idx', 'PAR', 'clean_text']]
data2.head()

df2 = process_text_similarity(data2, text_column='clean_text')
df2.head()

health['average_similarity'] = df2['average_similarity']
health

# save the results to the cvs. file
health.to_csv(brief + 'data.csv', index=False)

"""# health (social media group)"""

h = pd.read_csv(result + 'data.csv',engine='python')
h

data = h[['idx', 'id', 'clean_text']]
data.head()

# Step 1: Split text into sentences
data['clean_text'] = data['clean_text'].fillna('').astype(str) #Ensure clean_text column contains only strings
data['sent_list'] = data['clean_text'].apply(lambda x: [sent.text.strip() for sent in nlp(x).sents])

# Step 2: Compute Sentence Embeddings
model = SentenceTransformer("all-mpnet-base-v2")

data['sentence_embeddings'] = data['sent_list'].apply(
    lambda sentences: [model.encode(sentence, convert_to_tensor=True) for sentence in sentences])

# Step 3: Compute Cosine Similarities
#Using the sentence embeddings, compute the cosine similarities between consecutive sentences for each row.
def compute_pairwise_similarity(embeddings):
    if len(embeddings) < 2:
        return []  # No pairs to compare if fewer than 2 sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1]).item()
        similarities.append(sim)
    return similarities

data['pair_similarity'] = data['sentence_embeddings'].apply(compute_pairwise_similarity)

# Step 4: Calculate the average similarity
data['average_similarity'] = data['pair_similarity'].apply(lambda sims: np.mean(sims) if sims else None)
data

h['average_similarity'] = data['average_similarity']
h

# save the results to a new cvs. file
h.to_csv(brief + 'data.csv', index=False)



"""# ad (social media grop)"""

ad = pd.read_csv(result + 'data.csv',engine='python')
ad.head()

data = ad[['idx', 'id', 'clean_text']]
data.head()

# Step 1: Split text into sentences
data['clean_text'] = data['clean_text'].fillna('').astype(str) #Ensure clean_text column contains only strings
data['sent_list'] = data['clean_text'].apply(lambda x: [sent.text.strip() for sent in nlp(x).sents])

# Step 2: Compute Sentence Embeddings
model = SentenceTransformer("all-mpnet-base-v2")

data['sentence_embeddings'] = data['sent_list'].apply(
    lambda sentences: [model.encode(sentence, convert_to_tensor=True) for sentence in sentences])

# Step 3: Compute Cosine Similarities
#Using the sentence embeddings, compute the cosine similarities between consecutive sentences for each row.
from sentence_transformers import util

def compute_pairwise_similarity(embeddings):
    if len(embeddings) < 2:
        return []  # No pairs to compare if fewer than 2 sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1]).item()
        similarities.append(sim)
    return similarities

data['pair_similarity'] = data['sentence_embeddings'].apply(compute_pairwise_similarity)

# Step 4: Calculate the average similarity
data['average_similarity'] = data['pair_similarity'].apply(lambda sims: np.mean(sims) if sims else None)

data

ad['average_similarity'] = data['average_similarity']
ad

# save the results to a new cvs. file
ad.to_csv(brief + 'data.csv', index=False)