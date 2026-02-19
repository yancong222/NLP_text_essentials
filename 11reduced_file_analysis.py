#set up
import numpy as np
import pandas as pd
import spacy

"""# load the data"""

ad_clinical = pd.read_csv( result + 'data.csv',engine='python')
ad_clinical

"""# keep the id and idx columns along with the unique clean_text values"""

df = ad_clinical[["idx","id","clean_text"]].drop_duplicates().reset_index(drop=True) #.drop_duplicates():Removes duplicates by comparing all three columns.
df

import re
def attach_punctuation(text):
    # Use regex to remove space before punctuation
    return re.sub(r'\s+([?.!,:;])', r'\1', text)

df['clean_text'] = df['clean_text'].apply(attach_punctuation)
df

"""# VADER sentiment analysis"""

!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores and classify sentiment
def analyze_VADER_sentiment(text):
    # Check if the input is a string; if not, convert it to a string.
    if not isinstance(text, str):
        text = str(text)

    scores = analyzer.polarity_scores(text)

    # Classify sentiment based on the compound score
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound'], sentiment])

# Apply the sentiment analysis function to the 'clean_text' column
df[['VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'VADER_sentiment']] = df['clean_text'].apply(analyze_VADER_sentiment)
df

"""# lexical complexity (TTR)"""

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def spacy_TTR(text):

    if not isinstance(text, str):
        return pd.Series([0, 0, 0, 0.0])  # Handle non-string inputs gracefully

    # Process the text using spaCy
    doc = nlp(text)

    # Calculate sentence count
    sent_count = len(list(doc.sents)) #doc.sents:SpaCy uses its sentence boundary detection (SBD) algorithm to split the text into sentences.

    # Extract words (excluding punctuation and spaces)
    words = [token.text.lower() for token in doc if token.is_alpha]
    word_count = len(words)

    # Calculate unique word count
    unique_word_count = len(set(words))

    # Calculate Type-Token Ratio (TTR)
    TTR = unique_word_count / word_count if word_count > 0 else 0

    # Return the results as a pandas Series
    return pd.Series([sent_count, word_count, unique_word_count, TTR])

# Apply the compute_TTR function to the 'clean_text' column and add new columns to the DataFrame
df[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = df['clean_text'].apply(spacy_TTR)
df

"""# Syntactic complexity"""

def syntactic_complexity_metrics(text):

    if not isinstance(text, str):
        return pd.Series([0, 0, 0, 0, 0, 0])  # Handle non-string inputs gracefully

    # Process the text using spaCy
    doc = nlp(text)

    sen_count = 0     # Total number of sentences (treated as clauses)
    total_dependents = 0
    total_coordinate_phrases = 0
    total_complex_nominals = 0

    # Iterate over sentences in the text
    for sent in doc.sents:
        sen_count += 1

        # Initialize counters for the current sentence
        sent_dependents = 0
        sent_coord_phrases = 0
        sent_complex_nominals = 0

        for token in sent:
            # 1. Count dependents per clause
            if token.dep_ in {'advcl', 'csubj', 'ccomp', 'acl', 'xcomp', 'relcl'}:
                sent_dependents += 1

            # 2. Count coordinate phrases per clause
            if token.dep_ in {'cc', 'conj'}:
                sent_coord_phrases += 1

            # 3. Count complex nominals per clause
            if token.dep_ in {"nsubj", "dobj", "pobj", "iobj", "nmod"}:
                if any(child.dep_ in {"amod", "poss", "compound"} for child in token.children):
                    sent_complex_nominals += 1

        total_dependents += sent_dependents
        total_coordinate_phrases += sent_coord_phrases
        total_complex_nominals += sent_complex_nominals

    # Avoid division by zero (if no sentences are found)
    if sen_count == 0:
        return pd.Series([0, 0, 0, 0, 0, 0])

    # Calculate per-clause averages
    dependents_per_clause = total_dependents / sen_count
    coord_phrases_per_clause = total_coordinate_phrases / sen_count
    complex_nominals_per_clause = total_complex_nominals / sen_count

    # Return results as a pandas Series
    return pd.Series([
        total_dependents, dependents_per_clause,
        total_coordinate_phrases, coord_phrases_per_clause,
        total_complex_nominals, complex_nominals_per_clause
    ])

df[['total_dependents', 'dependents_per_clause',
    'total_coord_phrases', 'coord_phrases_per_clause',
    'total_complex_nominals', 'complex_nominals_per_clause']] = df['clean_text'].apply(syntactic_complexity_metrics)
df

"""# Cosine similarity"""

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-mpnet-base-v2")

data = ad_clinical[["idx","id","clean_text"]].drop_duplicates().reset_index(drop=True) #.drop_duplicates():Removes duplicates by comparing all three columns.
data

import spacy
nlp = spacy.load('en_core_web_sm')

# Step 1: Split text into sentences
data['clean_text'] = data['clean_text'].fillna('').astype(str)
data['sent_list'] = data['clean_text'].apply(lambda x: [sent.text.strip() for sent in nlp(x).sents])
print(data[['clean_text', 'sent_list']].head())

example_text = data['clean_text'].iloc[0]
doc = nlp(example_text)
sentences = [sent.text.strip() for sent in doc.sents]
sentences

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

df['average_similarity'] = data['average_similarity']
df



"""# POS tag"""

def count_pos_tags_by_speaker(df, pos_column='sp_pos', group_column='id'):
    # Ensure the sp_pos column is in list format if it contains lists as strings
    df[pos_column] = df[pos_column].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Explode the sp_pos column to have one row per POS tag
    df_exploded = df.explode(pos_column)

    # Count occurrences of each POS tag grouped by the speaker (id)
    pos_counts = df_exploded.groupby([group_column, pos_column]).size().reset_index(name='count')

    # Pivot to create a table with POS tags as columns and counts as values
    pos_counts_pivot = pos_counts.pivot(index=group_column, columns=pos_column, values='count').fillna(0)

    # Add a new column for the total POS count by summing across all POS columns
    pos_counts_pivot['total_pos'] = pos_counts_pivot.sum(axis=1)

    return pos_counts_pivot

df1 = count_pos_tags_by_speaker(ad_clinical, pos_column='sp_pos', group_column='id')
df1

def pos_analysis(df):

    # Step 1: Create a new DataFrame with selected columns
    pos_df = df[[ 'ADJ', 'VERB', 'PRON', 'NOUN', 'total_pos']].copy()

    # Step 2: Calculate normalized (percentage) columns for each POS type
    pos_df['ADJ%'] = (pos_df['ADJ'] / pos_df['total_pos']) * 100
    pos_df['VERB%'] = (pos_df['VERB'] / pos_df['total_pos']) * 100
    pos_df['PRON%'] = (pos_df['PRON'] / pos_df['total_pos']) * 100
    pos_df['NOUN%'] = (pos_df['NOUN'] / pos_df['total_pos']) * 100

    # Step 3: Calculate the specified POS ratios
    pos_df['Noun_to_Verb'] = pos_df['NOUN'] / (pos_df['VERB'] )
    pos_df['Pron_to_Noun'] = pos_df['PRON'] / (pos_df['NOUN'] )
    pos_df['Noun_to_Adj'] = pos_df['NOUN'] / (pos_df['ADJ'])
    pos_df['Verb_to_Adj'] = pos_df['VERB'] / (pos_df['ADJ'])

    return pos_df

df2 = pos_analysis(df1)
df2.head()

merged_df = pd.merge(df, df2[['ADJ', 'VERB', 'PRON', 'NOUN', 'total_pos',
                                            'ADJ%', 'VERB%', 'PRON%','NOUN%',
                                            'Noun_to_Verb', 'Pron_to_Noun', 'Noun_to_Adj', 'Verb_to_Adj']],
                     on='id',
                     how='left')
merged_df

"""# sentiment:emotion dictionary"""

dic1 = ad_clinical.groupby(['id']).agg({'Valence': lambda x: np.mean(x),
                                 'Arousal': lambda x: np.mean(x),
                                 'Dominance': lambda x: np.mean(x),
                                  'SemD': lambda x: np.mean(x)}).reset_index() #.reset_index():将 id 从索引恢复为 DataFrame 的普通列
dic1

# save the result to the orginal csv. file
final = pd.merge(merged_df , dic1[['id','Valence', 'Arousal', 'Dominance', 'SemD']],
                     on='id',
                     how='left')
final

final.columns

# save the results
final.to_csv(result+ 'data.csv', index=False)