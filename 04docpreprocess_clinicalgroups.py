import pandas as pd
import numpy as np

He = pd.read_csv(data + 'data.csv')
He.head(2)

# Display the unique values themselves
print("Unique types in 'diag':", He['diag'].unique())

# Filter the DataFrame to keep only rows where 'diag' is 'ModerateAD' or 'MildAD'
He_AD = He[He['diag'].isin(['ModerateAD', 'MildAD'])]

print(f"Number of rows: {He_AD.shape[0]}")
print(f"Number of columns: {He_AD.shape[1]}")
print(He_AD['diag'].unique())

# Filter the DataFrame to keep only rows where 'diag' is 'control'
He_health = He[He['diag'].isin(['Control'])]

print(f"Number of rows: {He_health.shape[0]}")
print(f"Number of columns: {He_health.shape[1]}")
print(He_health['diag'].unique())

"""### merge text group by speakers

The data was preprocessed and cleaned. Each participant’s utterances were originally split into separate rows in the ‘content’ column. We combined all utterances/contents spoken by the same participant into a single row/cell, grouped by participant ID.

#### He_AD

(1) for He_AD
"""

He_AD.columns

# Ensure 'clean_text' is treated as a string and handle NaNs
He_AD['clean_text'] = He_AD['clean_text'].fillna('').astype(str)

# Group by 'PAR' and aggregate
# - Concatenate 'clean_text' using ' '.join
# - Keep the first occurrence of other columns
grouped_He_AD = He_AD.groupby('PAR').agg({
    'clean_text': ' '.join,
    **{col: 'first' for col in He_AD.columns if col not in ['PAR', 'clean_text']}
}).reset_index()

grouped_He_AD

"""#### He_health

(2) for He_health
"""

He_health.columns

He_health['clean_text'] = He_health['clean_text'].fillna('').astype(str)

grouped_He_health = He_health.groupby('PAR').agg({
    'clean_text': ' '.join,
    **{col: 'first' for col in He_health.columns if col not in ['PAR', 'clean_text']}
}).reset_index()

grouped_He_health.head()

grouped_He_AD.shape

grouped_He_health.shape

print("Unique types in grouped_He_AD 'diag':", grouped_He_AD['diag'].unique())

print("Unique types in grouped_He_health 'diag':", grouped_He_health['diag'].unique())

"""## protocol_baycrest_delaware_mci

The data was preprocessed and cleaned. Each participant’s utterances were originally split into separate rows in the ‘content’ column. We combined all utterances/contents spoken by the same participant into a single row/cell, grouped by participant ID.

### Delaware_AD
"""

Delaware_AD = pd.read_csv(data + 'data.csv')
Delaware_AD = Delaware_AD.drop(columns=['Unnamed: 0']) #drop the Unnamed: 0 column
Delaware_AD

Delaware_AD.columns

# Display the unique values themselves
print("Unique types in 'diag':", Delaware_AD['CogDis'].unique())

Delaware_AD = Delaware_AD[Delaware_AD['CogDis'] == 'MCI']
Delaware_AD

# Display the unique values themselves
print("Unique types in 'diag':", Delaware_AD['CogDis'].unique())

Delaware_AD['text_clean'] = Delaware_AD['text_clean'].fillna('').astype(str)

grouped_Delaware_AD = Delaware_AD.groupby('participant').agg({
    'text_clean': ' '.join,
    **{col: 'first' for col in Delaware_AD.columns if col not in ['participant','text_clean']}
}).reset_index()

grouped_Delaware_AD

print("Unique types in grouped_Delaware_AD:", grouped_Delaware_AD['CogDis'].unique())

"""### Delaware_health"""

Delaware_health = pd.read_csv(data + 'data.csv')
Delaware_health = Delaware_health.drop(columns=['Unnamed: 0']) #drop the Unnamed: 0 column
Delaware_health.head(2)

# Display the unique values themselves
print("Unique types in 'diag':", Delaware_health['CogDis'].unique())

Delaware_health['text_clean'] = Delaware_health['text_clean'].fillna('').astype(str)

grouped_Delaware_health = Delaware_health.groupby('participant').agg({
    'text_clean': ' '.join,
    **{col: 'first' for col in Delaware_health.columns if col not in ['participant', 'text_clean']}
}).reset_index()

grouped_Delaware_health

"""## combine data from different dataset

### AD_clinical
"""

grouped_He_AD['corpus'] = 'He_Hinzen'

# Add a new column named 'index' starting from 0
grouped_He_AD.head(2)

grouped_He_AD.shape

grouped_Delaware_AD.head(2)

grouped_Delaware_AD.shape

grouped_He_AD.columns

grouped_Delaware_AD.columns

"""The common column in both dataframe, but they have different name, so we need to rename the column.
- participant/PAR
- age
- sex/gender
- clean_text/content_semi_clean
- diag/CogDis

"""

# rename the columns of grouped_Delaware_AD
# Define a dictionary mapping old column names to new column names
rename_dict = {
    'participant': 'PAR',
    'CogDis': 'diag',
    'sex': 'gender',
    'text_clean': 'clean_text'}

grouped_Delaware_AD = grouped_Delaware_AD.rename(columns=rename_dict)
grouped_Delaware_AD.head(2)

#Find common columns between the two DataFrames
common_columns = list(set(grouped_He_AD).intersection(set(grouped_Delaware_AD)))
common_columns

columns = [ 'PAR','corpus', 'diag', 'age','gender','clean_text']

# Concatenate the DataFrames using only the common columns
AD_clinical = pd.concat([grouped_He_AD[columns],
                         grouped_Delaware_AD[columns]],
                        ignore_index=True)

AD_clinical

grouped_He_AD.shape

grouped_Delaware_AD.shape

AD_clinical.shape

# save the results to a new cvs. file
AD_clinical.to_csv(result + 'data.csv', index=False)

"""### Health_clinical"""

grouped_He_health['corpus'] = 'He_Hinzen'
grouped_He_health.head(2)

grouped_Delaware_health.head(2)

# rename the columns of grouped_Delaware_health
# Define a dictionary mapping old column names to new column names
rename_dict = {
    'participant': 'PAR',
    'CogDis': 'diag',
    'sex': 'gender',
    'content_semi_clean': 'clean_text'}

grouped_Delaware_health = grouped_Delaware_health.rename(columns=rename_dict)
grouped_Delaware_health.head(2)

# Concatenate the DataFrames using only the common columns
Health_clinical = pd.concat([grouped_He_health[columns],
                         grouped_Delaware_health[columns]],
                        ignore_index=True)

Health_clinical

grouped_He_health.shape

grouped_Delaware_health.shape

Health_clinical.shape

# save the results to a new cvs. file
Health_clinical.to_csv(result + 'data.csv', index=False)

