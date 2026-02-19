# NLP_text_essentials
NLP scripts for text analyses essentials 

Authors: Jingying Hu; Trisha Godara; Yan Cong. Purdue University. 

This repository is a component of the analyses conducted for the manuscript "How robust are linguistic markers of aging? The case of aging-related social media text", _Natural Language Processing Journal_. https://doi.org/10.1016/j.nlp.2026.100203. 


| Notebook | Description |
|--------|-------------|
| Feature_Extraction.py | A stand-alone, all-in-one python notebook for extracting and computation of all NLP markers |
| 01DocPreprocess_ADSocial_Clustering_filter.py | ADSocial: use K-means Clustering (unsupervised method) to groups post into different clusters and filter out irrelevant posts [i.e., quality control] |
| 02DocPreprocess_HCSocial.py | HCSocial: combine two orginal csv. files into one file, and clean the "text" column |
| 03DocPreprocess_delaware_restore punctuation.py | restore missing punctuation at the end of sentences in the “content_semi_clean” column |
| 04DocPreprocess_ClinicalGroups.py | Clinical Groups: filter the data and combine them together |
| 05SentimentAnalysis_VADER.py | Performing sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool. |
| 06Syntactic_Complexity_Analysis.py | Analyzing the lexical and syntactic complexity of "clean_text" columns |
| 07Cosine_Similarity.py | Using a SentenceTransformer model (all-mpnet-base-v2) to camculate the cosine similarity of consecutive sentences and calculate the average cosine similarity for each row |
| 08POS_tagging.py | Tagging the parts-of-speech label and calculate the percentage of each tag |
| 09SentimentAnalysis_Emotion_Dictionary.py | Performing sentiment analysis using the emotion dictionary |
| 10Match_text_length.py | Based on the "lexicon" file, match the text lengths (total tokens) between ad and HC groups |
| 11Reduced_file_analysis.py | calculate all features for the trimmed csv. files (two length-matched text) |

## Variable Descriptions

| Type of Analysis | Variable Name | Description | Data Type | Additional Notes |
|----------------|--------------|-------------|-----------|------------------|
| data preprocessing | text | Original text of the social media post. | String | |
| data preprocessing | clean_text | Processed text after cleaning (e.g., removing punctuation, stopwords, etc.). | String | "VADER-Sentiment-Analysis<br>https://github.com/cjhutto/vaderSentiment<br><br>positive sentiment: compound score >= 0.05<br>neutral sentiment: (compound score > -0.05) and (compound score < 0.05)<br>negative sentiment: compound score <= -0.05" |
| sentiment analysis (using VADER) | VADER_neg | Negative sentiment score calculated by VADER. | Float (0 to 1) | |
| sentiment analysis (using VADER) | VADER_neu | Neutral sentiment score calculated by VADER. | Float (0 to 1) | |
| sentiment analysis (using VADER) | VADER_pos | Positive sentiment score calculated by VADER. | Float (0 to 1) | |
| sentiment analysis (using VADER) | VADER_compound | Compound sentiment score calculated by VADER, reflecting overall sentiment. | Float (-1 to 1) | |
| sentiment analysis (using VADER) | VADER_sentiment | Label assigned based on VADER scores (e.g., positive, neutral, negative). | String (Categorical) | |
| sentiment analysis (using emotion dictionary) | Valence | Average valence score of words in the text calculated using the emotion dictionary | Float | indicating the emotional positivity or negativity of the text. |
| sentiment analysis (using emotion dictionary) | Arousal | Average arousal score of words in the text calculated using the emotion dictionary | Float | indicating the intensity or energy of the emotion. |
| sentiment analysis (using emotion dictionary) | Dominance | Average dominance score of words in the text calculated using the emotion dictionary | Float | indicating the degree of control or power associated with the emotion. |
| sentiment analysis (using emotion dictionary) | SemD | Semantic Density, a measure of the complexity or richness of semantic relationships. | Float | not inclued in the project proposal |
| lexical complexity | sent_count | Number of sentences in the text | Integer | use spacy to calculate |
| lexical complexity | word_count | Total number of words in the text. | Integer | |
| lexical complexity | unique_word_count | Count of unique words in the text. | Integer | |
| lexical complexity | TTR | Type-Token Ratio=unique_word_count/word_count | Float | Higher values indicate greater lexical diversity. |
| syntatical complexity | total_dependents | Total number of dependents | Integer | use spacy (dependency label) |
| syntatical complexity | dependents_per_clause | Average number of dependents per clause. | Float | |
| syntatical complexity | total_complex_nominals | Total number of complex nominals | Integer | |
| syntatical complexity | complex_nominals_per_clause | Average number of complex nominals per clause. | Float | |
| sentence similarity | average_similarity | Average cosine similarity score between sentences within a text/post | Float (0 to 1) | model = SentenceTransformer("all-mpnet-base-v2") |
| part of speech analysis | ADJ | Count of adjectives in the text. | Integer | use spacy to label POS tagging |
| part of speech analysis | VERB | Count of verbs in the text. | Integer | |
| part of speech analysis | PRON | Count of pronouns in the text. | Integer | |
| part of speech analysis | NOUN | Count of nouns in the text. | Integer | |
| part of speech analysis | total_pos | Total count of Part-of-Speech (POS) tags. | Integer | |
| part of speech analysis | ADJ% | Percentage of adjectives relative to the total word count. | Float (0 to 1) | |
| part of speech analysis | VERB% | Percentage of verbs relative to the total word count. | Float (0 to 1) | |
| part of speech analysis | PRON% | Percentage of pronouns relative to the total word count. | Float (0 to 1) | |
| part of speech analysis | NOUN% | Percentage of nouns relative to the total word count. | Float (0 to 1) | |
| part of speech analysis | Noun_to_Verb | Ratio of nouns to verbs in the text. | Float | |
| part of speech analysis | Pron_to_Noun | Ratio of pronouns to nouns in the text. | Float | |
| part of speech analysis | Noun_to_Adj | Ratio of nouns to adjectives in the text. | Float | |
| part of speech analysis | Verb_to_Adj | Ratio of verbs to adjectives in the text. | Float | |



