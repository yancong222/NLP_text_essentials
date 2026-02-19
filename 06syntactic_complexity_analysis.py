import spacy
import pandas as pd

"""# define the funtion: spacy_TTR

 Analyzes a single text string to compute sentence count, word count, unique word count, and Type-Token Ratio (TTR).

"""

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

"""**Let us see an example**"""

# Sample DataFrame with a 'clean_text' column
data = {'clean_text': [
        "I love using Python! It's so versatile.",
        "Data science and machine learning are fascinating.",
        "SpaCy is great for NLP tasks."]}

df = pd.DataFrame(data)

# Apply the compute_TTR function to the 'clean_text' column and add new columns to the DataFrame
df[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = df['clean_text'].apply(spacy_TTR)
df

"""# define the function: syntactic_complexity_metric

    Analyze a single text entry to compute syntactic complexity metrics:
    - Total dependents
    - Dependents per clause
    - Coordinate phrases per clause
    - Complex nominals per clause

**Degree of sophistication (spacy)**

- the number of dependents per clause (amount of subordination),
- the number of coordinate phrases per clause (amount of coordination),
- the number of complex nominals per clause (degree of phrasal sophistication).

All the syntactic indices will be computed using spaCy.
"""

nlp = spacy.load('en_core_web_sm')

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

"""# AD (social media group)"""

#read the data
ad_social = pd.read_csv(result + 'data.csv', engine='python')
ad_social.head(2)

# apply function to calculate TTR
ad_social[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = ad_social['clean_text'].apply(spacy_TTR)
ad_social.head(2)

# apply function to calculate Degree of sophistication
ad_social[['total_dependents', 'dependents_per_clause',
    'total_coord_phrases', 'coord_phrases_per_clause',
    'total_complex_nominals', 'complex_nominals_per_clause']] = ad_social['clean_text'].apply(syntactic_complexity_metrics)
ad_social.head(2)

# save the results to a new cvs. file
ad_social.to_csv(result + 'data.csv', index=False)

"""let us also save the result to the brief result column"""

df1 = pd.read_csv(brief + 'data.csv')
df1.head(2)

df1[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = df1['clean_text'].apply(spacy_TTR)
df1.head(2)

df1[['total_dependents', 'dependents_per_clause',
    'total_coord_phrases', 'coord_phrases_per_clause',
    'total_complex_nominals', 'complex_nominals_per_clause']] = df1['clean_text'].apply(syntactic_complexity_metrics)
df1.head(2)

# save the results to a new cvs. file
df1.to_csv(brief + 'data.csv', index=False)

"""# Health (social media group)"""

#read the data
health_social = pd.read_csv(result + 'data.csv',engine='python')
health_social.head(2)

health_social[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = health_social['clean_text'].apply(spacy_TTR)
health_social.head(2)

health_social[['total_dependents', 'dependents_per_clause',
    'total_coord_phrases', 'coord_phrases_per_clause',
    'total_complex_nominals', 'complex_nominals_per_clause']] = health_social['clean_text'].apply(syntactic_complexity_metrics)
health_social.head(2)

# save the results to the cvs. file
health_social.to_csv(result + 'data.csv', index=False)

"""let us also save the results to another cvs fule"""

df2 = pd.read_csv(brief + 'data.csv')
df2.head(2)

df2[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = df2['clean_text'].apply(spacy_TTR)
df2.head(2)

df2[['total_dependents', 'dependents_per_clause',
    'total_coord_phrases', 'coord_phrases_per_clause',
    'total_complex_nominals', 'complex_nominals_per_clause']] = df2['clean_text'].apply(syntactic_complexity_metrics)
df2.head(2)

# save the results to a new cvs. file
df2.to_csv(brief + 'data.csv', index=False)

"""# AD (clinical group)"""

#read the data
ad_clinical = pd.read_csv(result + 'data.csv',engine='python')
ad_clinical.head(2)

ad_clinical[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = ad_clinical['clean_text'].apply(spacy_TTR)
ad_clinical.head(2)

ad_clinical[['total_dependents', 'dependents_per_clause',
    'total_coord_phrases', 'coord_phrases_per_clause',
    'total_complex_nominals', 'complex_nominals_per_clause']] = ad_clinical['clean_text'].apply(syntactic_complexity_metrics)
ad_clinical.head(2)

# save the results to a new cvs. file
ad_clinical.to_csv(result + 'data.csv', index=False)

"""# Health (clinical group)"""

#read the data
health_clinical = pd.read_csv(result + 'data.csv',engine='python')
health_clinical.head(2)

health_clinical[['sent_count', 'word_count', 'unique_word_count', 'TTR']] = health_clinical['clean_text'].apply(spacy_TTR)
health_clinical.head(2)

health_clinical[['total_dependents', 'dependents_per_clause',
    'total_coord_phrases', 'coord_phrases_per_clause',
    'total_complex_nominals', 'complex_nominals_per_clause']] = health_clinical['clean_text'].apply(syntactic_complexity_metrics)
health_clinical.head(2)

# save the results to the cvs. file
health_clinical.to_csv(result + 'data.csv', index=False)