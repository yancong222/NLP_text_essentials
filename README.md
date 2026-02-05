# NLP_text_essentials
NLP scripts for text analyses essentials 

Authors: 

Jingying Hu. Purdue University. hu880@purdue.edu

Yan Cong. Purdue University. cong4@purdue.edu

## 03Scripts

| Notebook | Description |
|--------|-------------|
| 01DocPreprocess_ADSocial_Clustering_filter.ipynb | AD_SOCIAL: use K-means Clustering (unsupervised method) to groups post into different clusters and filter out irrelevant posts of |
| 02DocPreprocess_HealthSocial.ipynb | HealthSocial: combine two orginal csv. files into one file, and clean the "text" column |
| 03DocPreprocess_delaware_restore punctuation.ipynb | restore missing punctuation at the end of sentences in the “content_semi_clean” column |
| 04DocPreprocess_ClinicalGroups.ipynb | Clinical Groups: filter the data and combine them together |
| 05SentimentAnalysis_VADER.ipynb | Performing sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool. |
| 06Syntactic_Complexity_Analysis.ipynb | Analyzing the lexical and syntactic complexity of "clean_text" columns |
| 07Cosine_Similarity.ipynb | Using a SentenceTransformer model (all-mpnet-base-v2) to camculate the cosine similarity of consecutive sentences and calculate the average cosine similarity for each row |
| 08Part-of-speech_tagging.ipynb | Tagging the part-of-speech label and calculate the percentage of each tag |
| 09SentimentAnalysis_Emotion_Dictionary.ipynb | Performing sentiment analysis using the emotion dictionary |
| 10Match_text_length.ipynb | Based on the "lexicon" file, match the text lengths (total tokens) between ad and health groups |
| 11Reduced_file_analysis.ipynb | calculate all features for the trimmed csv. files (two length-matched text) |
