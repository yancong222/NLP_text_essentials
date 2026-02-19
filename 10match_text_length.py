import pandas as pd
import string
import re
import nltk

"""# social media groups:Health vs AD

Columns Overview:
- idx and id: Unique identifiers for each post.
- clean_text: The entire post's cleaned text.
- word: Words exploded from clean_text.
"""

#read the data
ad_social = pd.read_csv( result + 'data.csv',engine='python')
ad_social

#read the data
health_social = pd.read_csv(result + 'data.csv',engine='python')
health_social

"""Since the word column in both DataFrames is exploded from the clean_text column, we can reduce the rows in ad_social to match the total row count of health_social."""

# Reduce ad_social rows to match health_social's row count
ad_social_reduced = ad_social.iloc[:health_social.shape[0]].reset_index(drop=True)
ad_social_reduced

last_id = ad_social_reduced.iloc[-1]['id']

last_post_rows = ad_social_reduced[ad_social_reduced['id'] == last_id]

# Reconstruct the full text by concatenating words
reconstructed_text = ' '.join(last_post_rows['word'])

print(f" (id={last_id}):\n{reconstructed_text}")

# replace the content of the clean_text column for the last post (id equal to the last post) in the reduced ad_social
ad_social_reduced.loc[ad_social_reduced['id'] == last_id, 'clean_text'] = reconstructed_text

# Display the updated DataFrame for the last post
ad_social_reduced[ad_social_reduced['id'] == last_id]

#save the result
ad_social_reduced.to_csv(result + 'data.csv', index=False)

"""# Clinical diagnosis group: Health vs AD"""

#read the data
ad_clinical  = pd.read_csv(result + 'data.csv',engine='python')
ad_clinical

#read the data
health_clinical = pd.read_csv(result + 'data.csv',engine='python')
health_clinical

# Reduce ad_clinical rows to match health_clinical's row count
ad_clinical_reduced = ad_clinical.iloc[:health_clinical.shape[0]].reset_index(drop=True)
ad_clinical_reduced

#Find the `id` of the last post in the reduced ad_clinical DataFrame
last_id = ad_clinical_reduced.iloc[-1]['id']

#Filter rows corresponding to the last `id`
last_post_rows = ad_clinical_reduced[ad_clinical_reduced['id'] == last_id]
last_post_rows

original_text = last_post_rows['clean_text'].iloc[0]

# Reconstruct the full text by concatenating words
reconstructed_text = ' '.join(last_post_rows['word'])

print(f" (id={last_id}):\n{reconstructed_text}")

# replace the content of the clean_text column for the last post (id equal to the last post) in the reduced ad_clinical
ad_clinical_reduced.loc[ad_clinical_reduced['id'] == last_id, 'clean_text'] = reconstructed_text

# Display the updated DataFrame for the last post
ad_clinical_reduced[ad_clinical_reduced['id'] == last_id]

#save the result
ad_clinical_reduced.to_csv(data + 'data.csv', index=False)

"""# final check"""

print(f"Total columns in ad_social: {ad_social_reduced.shape[0]}")
print(f"Total columns in health_social: {health_social.shape[0]}")

print(f"Total columns in ad_clinical: {ad_clinical_reduced.shape[0]}")
print(f"Total columns in health_clinical: {health_clinical.shape[0]}")