#set up
import numpy as np
import pandas as pd

"""# Load the valence emotion dictionary"""

Valence_Emotion = pd.read_csv(dic + 'Valence Emotion dictionary.csv')
Valence_Emotion.head()

Valence_Emotion_dict = {}

for i in Valence_Emotion.index:
  Valence_Emotion_dict[Valence_Emotion['Word'][i]] = [Valence_Emotion['V.Mean.Sum'][i],
                                                      Valence_Emotion['A.Mean.Sum'][i],
                                                      Valence_Emotion['D.Mean.Sum'][i]]

#test
Valence_Emotion_dict['abandon']

"""# Load the semantic diversity dictionary"""

SemD = pd.read_csv(dic + 'SemD.csv')
SemD.head()

SemD_dict = {}
for i in SemD.index:
  SemD_dict[SemD['item'][i]] = SemD['SemD'][i]

#test
SemD_dict['weird'] # 1.47

"""# AD (social media group)

## read the data
"""

#read the data
ad_social = pd.read_csv(result + 'data.csv')
ad_social.shape

ad_social.head(3)

"""## define a function to retrieve values from disctionaries"""

# define a function to retrieve values (valuence, arousal, dominance, semd) from the dictionaries**

def get_emotion_values(word):
    valence = Valence_Emotion_dict.get(word, [None, None, None])[0]
    arousal = Valence_Emotion_dict.get(word, [None, None, None])[1]
    dominance = Valence_Emotion_dict.get(word, [None, None, None])[2]
    semd = SemD_dict.get(word, None)
    return pd.Series([valence, arousal, dominance, semd], index=['Valence', 'Arousal', 'Dominance', 'SemD'])

# test
print(get_emotion_values('shocking'))
'''
Valence      4.63
Arousal      5.30
Dominance    4.12
SemD         1.85
dtype: float64
'''

"""## apply the function to get the value"""

# Apply the helper function to each row in the DataFrame
ad_social[['Valence', 'Arousal', 'Dominance', 'SemD']] = ad_social['word'].apply(get_emotion_values)

ad_social

# save the results
ad_social.to_csv(result + 'data.csv', index=False)

"""## calcute the mean values for each "id/speaker"

For each unique 'id', calculate the mean (average) for the columns 'Valence', 'Arousal', 'Dominance', and 'SemD'.
"""

df1 = ad_social.groupby(['id']).agg({'Valence': lambda x: np.mean(x),
                                 'Arousal': lambda x: np.mean(x),
                                 'Dominance': lambda x: np.mean(x),
                                  'SemD': lambda x: np.mean(x)}).reset_index() #.reset_index():将 id 从索引恢复为 DataFrame 的普通列
df1.head()

df1.columns

ad_social_speakers = pd.read_csv(result + 'data.csv',engine='python')
ad_social_speakers.head(3)

# save the result to the orginal csv. file
merged_df1 = pd.merge(ad_social_speakers, df1[['id','Valence', 'Arousal', 'Dominance', 'SemD']],
                     on='id',
                     how='left')
merged_df1

# save the results
merged_df1.to_csv(brief + 'data.csv', index=False)

"""# Health (social media group)"""

#read the data
health_social = pd.read_csv(result + 'data.csv')
health_social.shape

health_social.head(3)

# Apply the helper function to each row in the DataFrame
health_social[['Valence', 'Arousal', 'Dominance', 'SemD']] = health_social['word'].apply(get_emotion_values)

health_social

# save the results
health_social.to_csv(result + 'data.csv', index=False)

df2 = health_social.groupby(['id']).agg({'Valence': lambda x: np.mean(x),
                                 'Arousal': lambda x: np.mean(x),
                                 'Dominance': lambda x: np.mean(x),
                                  'SemD': lambda x: np.mean(x)}).reset_index()
df2.head()

#read the file
health_social_speakers = pd.read_csv(result + 'data.csv',engine='python')
health_social_speakers.head(3)

# save the result to the original csv. file
merged_df2 = pd.merge(health_social_speakers, df2[['id','Valence', 'Arousal', 'Dominance', 'SemD']],
                     on='id',
                     how='left')
merged_df2

# save the results
merged_df2.to_csv(brief + 'data.csv', index=False)

"""# AD (clinical label group)"""

#read the data
ad_clinical= pd.read_csv(result + 'data.csv')
ad_clinical.shape

ad_clinical.head(3)

# Apply the helper function to each row in the DataFrame
ad_clinical[['Valence', 'Arousal', 'Dominance', 'SemD']] = ad_clinical['word'].apply(get_emotion_values)

# Display the DataFrame
ad_clinical

# save the results
ad_clinical.to_csv(result + 'data.csv', index=False)
print("The file was successfully saved")

df3 = ad_clinical.groupby(['id']).agg({'Valence': lambda x: np.mean(x),
                                 'Arousal': lambda x: np.mean(x),
                                 'Dominance': lambda x: np.mean(x),
                                  'SemD': lambda x: np.mean(x)}).reset_index()
df3.head()

ad_clinical_speakers = pd.read_csv(result + 'data.csv',engine='python')
ad_clinical_speakers.head(3)

# save the result to the original csv. file
merged_df3 = pd.merge(ad_clinical_speakers, df3[['id','Valence', 'Arousal', 'Dominance', 'SemD']],
                     on='id',
                     how='left')
merged_df3

# save the results
merged_df3.to_csv(brief + 'data.csv', index=False)

"""# Health (clinical label group)"""

#read the data
health_clinical = pd.read_csv(result + 'data.csv')
health_clinical.shape

health_clinical.head(3)

# Apply the helper function to each row in the DataFrame
health_clinical[['Valence', 'Arousal', 'Dominance', 'SemD']] = health_clinical['word'].apply(get_emotion_values)

# Display the DataFrame
health_clinical

# save the results
health_clinical.to_csv(result + 'data.csv', index=False)
print("The file was successfully saved")

df4 = health_clinical.groupby(['id']).agg({'Valence': lambda x: np.mean(x),
                                 'Arousal': lambda x: np.mean(x),
                                 'Dominance': lambda x: np.mean(x),
                                  'SemD': lambda x: np.mean(x)}).reset_index()
df4.head()

df4.columns

#load the file to save the result
health_clinical_speakers = pd.read_csv(result + 'data.csv',engine='python')
health_clinical_speakers.head(3)

# save the result to the original csv. file
merged_df4 = pd.merge(health_clinical_speakers, df4[['id','Valence', 'Arousal', 'Dominance', 'SemD']],
                     on='id',
                     how='left')
merged_df4

# save the results
merged_df4.to_csv(brief + 'data.csv', index=False)