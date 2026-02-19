import pandas as pd
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

"""# Read the data"""

#read the data
df = pd.read_csv('data.csv')

df.columns

"""# Sentiment analysis with VADER"""

!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# define a function to get sentiment scores and classify sentiment
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

"""#Syntactic Complexity_Analysis"""

#Load the spaCy model
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

# apply those two functions to our dataset
df[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = df['clean_text'].apply(spacy_TTR)

df[['total_dependents', 'dependents_per_clause',
    'total_coord_phrases', 'coord_phrases_per_clause',
    'total_complex_nominals', 'complex_nominals_per_clause']] = df['clean_text'].apply(syntactic_complexity_metrics)

df

"""# Cosine similarity"""

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-mpnet-base-v2")

#Processes text data to compute the average similarity for each row in a DataFrame.
def process_text_average_similarity(data, text_column='content_clean'):
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

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
            return None  # Return None if fewer than 2 sentences
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

    # Drop intermediate columns to keep only the average similarity
    data = data.drop(columns=['sent_list', 'sentence_embeddings', 'pair_similarity'])

    return data

# Apply the function and add the average similarity as a new column
df = process_text_average_similarity(df, text_column='clean_text')
df

"""# Part of Speech"""

# Tokenize the text (split each sentence into words)
df['word_list'] = df['clean_text'].astype(str).apply(lambda x: x.split())
df.head(2)

df_subset =df[['id','clean_text','word_list']]
df_subset

# Explode the 'word_list' column, creating a separate row for each word, and renames the exploded column to 'word'
df_exploded = df_subset.explode('word_list').rename(columns={'word_list': 'word'})
df_exploded

#lowcasing, remove quotation marks (single quotes, double quotes, and special quotes) from the start and end of the strings
def clean_word_column(df):

    # Remove punctuation: Uses regex to remove standard punctuation (from string.punctuation) around each word.
    df['word'] = df['word'].str.replace(f"[{string.punctuation}]", "", regex=True)

    # Lowercasing: Converts each word to lowercase (we did this when preprocessing "text" column)
    df['word'] = df['word'].str.lower()

    # Quotation mark removal: Uses regex to remove single quotes, double quotes, and special quotes from the start and end of each word.
    df['word'] = df['word'].str.replace(r"^[‘’“”\"']|[‘’“”\"']$", '', regex=True)

    # Row filtering: Remove rows where 'word_lower' is NaN or contains non-string values
    df = df[df['word'].apply(lambda x: isinstance(x, str))]

    # Return the result
    return df

df_lex = clean_word_column(df_exploded)
df_lex[['id','word']]

# use Spacy: tokenize words, and POS tagging
def process_word_spacy(df, text_column='word'):
    # Ensure column is in string format and not NaN
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    # Create spaCy documents using nlp.pipe for efficient processing
    docs = list(nlp.pipe(df[text_column]))

    # Initialize lists to store processing results
    token_texts = []
    lemmas = []
    pos_tags = []
    tags = []
    token_counts = []

    # Process each document and extract relevant information
    for doc in docs:
        # Token text (original form), lemmas (base form), POS (simple and detailed)
        token_texts.append([token.text for token in doc])
        lemmas.append([token.lemma_ for token in doc])
        pos_tags.append([token.pos_ for token in doc])  # Simple POS tags
        tags.append([token.tag_ for token in doc])  # Detailed POS tags
        token_counts.append(len(doc))  # Number of tokens in each doc

    # Add the results as new columns in the DataFrame
    df = df.assign(
        sp_tokenized=token_texts,
        sp_n_token=token_counts,
        sp_lemma=lemmas,
        sp_pos=pos_tags,
        sp_tag=tags)
    return df

df_lex = process_word_spacy(df_lex)
df_lex

# save the results to a new csv file
df_lex.to_csv('data.csv', index=False)

#Count the number of POS tags (group by id)
def count_pos_tags_by_speaker(df, pos_column='sp_pos', group_column='id'):

    # Ensure the specified columns exist in the DataFrame
    if pos_column not in df.columns or group_column not in df.columns:
        raise ValueError(f"Columns '{pos_column}' and '{group_column}' must exist in the DataFrame.")

    # Explode the pos_column to have one row per POS tag
    df_exploded = df.explode(pos_column)

    # Group by the specified group column and count each POS tag
    pos_counts = (
        df_exploded.groupby(group_column)[pos_column]
        .value_counts()
        .unstack(fill_value=0))  # Fill NaNs with 0s for missing POS tags

    # Add a new column for the total POS count by summing across all POS columns
    pos_counts['total_pos'] = pos_counts.sum(axis=1)

    # Return the resulting DataFrame
    return pos_counts

df_pos = count_pos_tags_by_speaker(df_lex)
df_pos

"""POS: The simple UPOS part-of-speech tag

Universal POS tag: https://universaldependencies.org/u/pos/

- ADJ: adjective.
- ADP: adposition.
- ADV: adverb.
- AUX: auxiliary.
- CCONJ: coordinating conjunction.
- DET: determiner.
- INTJ: interjection.
- NOUN: noun.
- NUM: numeral.
- PART: particle.
- PRON: pronoun.
- PROPN: proper noun.
- PUNCT: punctuation.
- SCONJ: subordinating conjunction.
- SYM: symbol.
- VERB: verb.
- X: other.
"""

# count proportion of Part-of-speech
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

pos_result = pos_analysis(df_pos)
pos_result

pos_result.columns

# save the result to the orginal csv. file
# pd.merge(): Merges the two DataFrames on the common column 'id'
# how='left': Ensures that all rows from ad_social are kept, even if there’s no matching row in pos_result.

merged_df = pd.merge(df, pos_result[['ADJ', 'VERB', 'PRON', 'NOUN', 'total_pos',
                                            'ADJ%', 'VERB%', 'PRON%','NOUN%',
                                            'Noun_to_Verb', 'Pron_to_Noun', 'Noun_to_Adj', 'Verb_to_Adj']],
                     on='id',
                     how='left')
merged_df

"""# SentimentAnalysis using Emotion Dictionary"""

Valence_Emotion = pd.read_csv('Valence Emotion dictionary.csv')
Valence_Emotion

Valence_Emotion_dict = {}

for i in Valence_Emotion.index:
  Valence_Emotion_dict[Valence_Emotion['Word'][i]] = [Valence_Emotion['V.Mean.Sum'][i],
                                                      Valence_Emotion['A.Mean.Sum'][i],
                                                      Valence_Emotion['D.Mean.Sum'][i]]

#test
Valence_Emotion_dict['abandon']

SemD = pd.read_csv('SemD.csv')
SemD

SemD_dict = {}
for i in SemD.index:
  SemD_dict[SemD['item'][i]] = SemD['SemD'][i]

#test
SemD_dict['weird']

# define a function to retrieve values (valuence, arousal, dominance, semd) from the dictionaries**

def get_emotion_values(word):
    valence = Valence_Emotion_dict.get(word, [None, None, None])[0]
    arousal = Valence_Emotion_dict.get(word, [None, None, None])[1]
    dominance = Valence_Emotion_dict.get(word, [None, None, None])[2]
    semd = SemD_dict.get(word, None)
    return pd.Series([valence, arousal, dominance, semd], index=['Valence', 'Arousal', 'Dominance', 'SemD'])

# Apply the helper function to each row in the DataFrame
df_lex[['Valence', 'Arousal', 'Dominance', 'SemD']] = df_lex['word'].apply(get_emotion_values)

df_lex

# save the results to a new cvs. file
df_lex.to_csv('SAGEUSA_lexicon.csv', index=False)

# For each participant, calculate the mean (average) for the columns 'Valence', 'Arousal', 'Dominance', and 'SemD'.
emotion = df_lex.groupby(['id']).agg({'Valence': lambda x: np.mean(x),
                                 'Arousal': lambda x: np.mean(x),
                                 'Dominance': lambda x: np.mean(x),
                                  'SemD': lambda x: np.mean(x)}).reset_index()
emotion

# pd.merge(): Merges the two DataFrames on the common column 'id'
final = pd.merge(merged_df, emotion[['id','Valence', 'Arousal', 'Dominance', 'SemD']],
                     on='id',
                     how='left')
final

# save the results to a new cvs. file
final.to_csv('data.csv', index=False)
