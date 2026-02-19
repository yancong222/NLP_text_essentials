#set up
import re
import string
import spacy
import pandas as pd

"""# ad (social media group)

## read the data
"""

#read the data
ad_social = pd.read_csv(result + 'data.csv',engine='python')
ad_social

len(ad_social["id"])

len(set(ad_social["id"])) #each id is a unique number

"""## tokenize the text, and Explode the 'word_list' column"""

# Tokenize the text (split each sentence into words)
ad_social['word_list'] = ad_social['clean_text'].astype(str).apply(lambda x: x.split())
ad_social.head(2)

df=ad_social[['idx','id','clean_text','word_list']]
df

# Explode the 'word_list' column, creating a separate row for each word, and renames the exploded column to 'word'
df_exploded = df.explode('word_list').rename(columns={'word_list': 'word'})
df_exploded.head()

"""**lowcasing, remove quotation marks (single quotes, double quotes, and special quotes) from the start and end of the strings**"""

def clean_word_column(df):

    # Remove punctuation: Uses regex to remove standard punctuation (from string.punctuation) around each word.
    df['word'] = df['word'].str.replace(f"[{string.punctuation}]", "", regex=True)

    # Quotation mark removal: Uses regex to remove single quotes, double quotes, and special quotes from the start and end of each word.
    df['word'] = df['word'].str.replace(r"^[‘’“”\"']|[‘’“”\"']$", '', regex=True)

    # Row filtering: Remove rows where 'word_lower' is NaN or contains non-string values
    df = df[df['word'].apply(lambda x: isinstance(x, str))]

    # Return the result
    return df

df_lex = clean_word_column(df_exploded)
df_lex[['idx','id','word']]

"""## use Spacy: tokenize words, and POS tagging"""

# Load the spaCy model for English,which includes tokenization, lemmatization, and POS tagging capabilities.
nlp = spacy.load("en_core_web_sm")

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
        sp_tag=tags
    )

    return df

df_lex = process_word_spacy(df_lex)
df_lex.head()

# save the results to a new cvs. file
df_lex.to_csv(result + 'data.csv', index=False)

# Inform the user that the file was successfully saved
print(f"The file was successfully saved at: {result + 'data.csv'}")

"""## Count the number of POS tags (group by id)"""

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
        .unstack(fill_value=0)  # Fill NaNs with 0s for missing POS tags
    )

    # Add a new column for the total POS count by summing across all POS columns
    pos_counts['total_pos'] = pos_counts.sum(axis=1)

    # Return the resulting DataFrame
    return pos_counts

df_pos = count_pos_tags_by_speaker(df_lex)
df_pos.head()

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

## Proportion of Part-of-speech
"""

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
pos_result.head()

pos_result.columns

merged_df = pd.merge(ad_social, pos_result[['ADJ', 'VERB', 'PRON', 'NOUN', 'total_pos',
                                            'ADJ%', 'VERB%', 'PRON%','NOUN%',
                                            'Noun_to_Verb', 'Pron_to_Noun', 'Noun_to_Adj', 'Verb_to_Adj']],
                     on='id',
                     how='left')
merged_df

# save the results to a new cvs. file
merged_df.to_csv(brief + 'data.csv', index=False)

"""let us also save the results to the "brief results"

# Health (social media group)
"""

#read the data
health_social = pd.read_csv(result + 'data.csv',engine='python')

health_social.head(2)

len(health_social["id"])

len(set(health_social["id"])) #each id is a unique number

"""**tokenize** the text, and Explode the 'word_list' column"""

# Tokenize the text (split each sentence into words)
health_social['word_list'] = health_social['clean_text'].astype(str).apply(lambda x: x.split())
health_social.head(2)

df=health_social[['idx','id','clean_text','word_list']]
df

# Explode the 'word_list' column, creating a separate row for each word, and renames the exploded column to 'word'
df_exploded = df.explode('word_list').rename(columns={'word_list': 'word'})
df_exploded.head()

"""**clean the "word" column**

lowcasing, remove quotation marks (single quotes, double quotes, and special quotes) from the start and end of the strings
"""

df_lex = clean_word_column(df_exploded)
df_lex[['idx','id','word']]

"""**use Spacy: tokenize words, and POS tagging**"""

df_lex = process_word_spacy(df_lex)
df_lex.head()

# save the results
df_lex.to_csv(result + 'data.csv', index=False)

# Count the number of POS tags (group by id)
df_pos = count_pos_tags_by_speaker(df_lex)
df_pos.head()

# count proportion of Part-of-speech
pos_result = pos_analysis(df_pos)
pos_result.head()

# save the result to the orginal csv. file
merged_df = pd.merge(health_social, pos_result[['ADJ', 'VERB', 'PRON', 'NOUN', 'total_pos',
                                                'ADJ%', 'VERB%', 'PRON%','NOUN%',
                                                'Noun_to_Verb', 'Pron_to_Noun', 'Noun_to_Adj', 'Verb_to_Adj']],
                     on='id',
                     how='left')
merged_df

# save the results to a new cvs. file
merged_df.to_csv(brief + 'data.csv', index=False)

"""# AD (clinical label group)

"""

#read the data
ad_clinical = pd.read_csv(result + 'data.csv',engine='python')

ad_clinical.head(2)

ad_clinical.rename(columns={'PAR': 'id'}, inplace=True)

len(ad_clinical["id"]) == len(set(ad_clinical["id"])) #each id is a unique number

# Tokenize the text (split each sentence into words)
ad_clinical['word_list'] = ad_clinical['clean_text'].astype(str).apply(lambda x: x.split())
ad_clinical.head(2)

df=ad_clinical[['idx',"id",'clean_text','word_list']]
df

# Explode the 'word_list' column, creating a separate row for each word, and renames the exploded column to 'word'
df_exploded = df.explode('word_list').rename(columns={'word_list': 'word'})
df_exploded.head()

# Remove rows where the 'word' column is not alphabetic
df_exploded = df_exploded[df_exploded['word'].str.isalpha()]
df_exploded.head()

# use Spacy: POS tagging
df_lex = process_word_spacy(df_exploded)
df_lex.head()

# save the results
df_lex.to_csv(result + 'data.csv', index=False)

# Count the number of POS tags (group by id)
df_pos = count_pos_tags_by_speaker(df_lex)
df_pos.head()

# count proportion of Part-of-speech
pos_result = pos_analysis(df_pos)
pos_result.head()

# save the result to the orginal csv. file
merged_df = pd.merge(ad_clinical, pos_result[['ADJ', 'VERB', 'PRON', 'NOUN', 'total_pos',
                                              'ADJ%', 'VERB%', 'PRON%','NOUN%',
                                              'Noun_to_Verb', 'Pron_to_Noun', 'Noun_to_Adj', 'Verb_to_Adj']],
                     on='id',
                     how='left')
merged_df

# save the results to a new cvs. file
merged_df.to_csv(brief + 'data.csv', index=False)

"""# Health (clinical label group)"""

#read the data
health_clinical = pd.read_csv(result + 'data.csv',engine='python')

health_clinical.columns

health_clinical.head(2)

health_clinical.rename(columns={'PAR': 'id'}, inplace=True)

len(health_clinical["id"]) == len(set(health_clinical["id"])) #each id is a unique number

# Tokenize the text (split each sentence into words)
health_clinical['word_list'] = health_clinical['clean_text'].astype(str).apply(lambda x: x.split())
health_clinical.head(2)

df=health_clinical[['idx','id','clean_text','word_list']]
df

# Explode the 'word_list' column, creating a separate row for each word, and renames the exploded column to 'word'
df_exploded = df.explode('word_list').rename(columns={'word_list': 'word'})
df_exploded.head()

# Remove rows where the 'word' column is not alphabetic
df_exploded = df_exploded[df_exploded['word'].str.isalpha()]
df_exploded.head()

# use Spacy: POS tagging
df_lex = process_word_spacy(df_exploded)
df_lex.head()

# save the results
df_lex.to_csv(result + 'data.csv', index=False)

# Count the number of POS tags (group by id)
df_pos = count_pos_tags_by_speaker(df_lex)
df_pos.head()

# count proportion of Part-of-speech
pos_result = pos_analysis(df_pos)
pos_result.head()

pos_result.columns

# save the result to the orginal csv. file
merged_df = pd.merge(health_clinical, pos_result[['ADJ', 'VERB', 'PRON', 'NOUN', 'total_pos',
                                                  'ADJ%', 'VERB%', 'PRON%','NOUN%',
                                                  'Noun_to_Verb', 'Pron_to_Noun', 'Noun_to_Adj', 'Verb_to_Adj']],
                     on='id',
                     how='left')
merged_df

# save the results to a new cvs. file
merged_df.to_csv(brief + 'data.csv', index=False)