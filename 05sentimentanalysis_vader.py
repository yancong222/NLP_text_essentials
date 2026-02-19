"""# Example: sentiment analysis with VADER"""

!pip install vaderSentiment

# a long continous string
text = "Shocking new data has shown xxx"
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

scores = analyzer.polarity_scores(text)
scores # {'neg': 0.089, 'neu': 0.847, 'pos': 0.064, 'compound': -0.907}

"""# Define a function to get sentiment scores and classify sentiment"""

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

"""# AD (social media group)"""

#read the data
ad_social = pd.read_csv(data + 'data.csv',engine='python')
ad_social

ad_social.shape

# Apply the sentiment analysis function to the 'clean_text' column
ad_social[['VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'VADER_sentiment']] = ad_social['clean_text'].apply(analyze_VADER_sentiment)
ad_social

ad_social = ad_social.drop(columns=['Unnamed: 0'])
ad_social.head(2)

# save the results to a new cvs. file
ad_social.to_csv(result + 'data.csv', index=False)

"""let us also save the results in a new df."""

df1 = ad_social[['idx', 'id',	'clean_text','VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'VADER_sentiment']]
df1.head(2)

# save the results to a new cvs. file
df1.to_csv(brief + 'data.csv', index=False)

"""# Health (social media group)"""

#read the data
health_social = pd.read_csv( data + 'data.csv',engine='python')
#health_social.insert(0, 'idx', health_social.index) #Add a new column named "index" using the existing DataFrame index
health_social

health_social.shape

# Apply the sentiment analysis function to the 'clean_text' column
health_social[['VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'VADER_sentiment']] = health_social['clean_text'].apply(analyze_VADER_sentiment)
health_social

# save the results to a new cvs. file
health_social.to_csv(result + 'data.csv', index=False)

df2 = health_social[['idx', 'id',	'clean_text','VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'VADER_sentiment']]
df2.head(2)

# save the results to a new cvs. file
df2.to_csv(brief + 'data.csv', index=False)

"""# AD (clinical group)"""

#read the data
ad_clinical = pd.read_csv(data + 'data.csv',engine='python')
ad_clinical.insert(0, 'idx', ad_clinical.index) #Add a new column named "index" using the existing DataFrame index
ad_clinical.head(2)

ad_clinical.shape

# Apply the sentiment analysis function to the 'clean_text' column
ad_clinical[['VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'VADER_sentiment']] = ad_clinical['clean_text'].apply(analyze_VADER_sentiment)
ad_clinical

# save the results to a new cvs. file
ad_clinical.to_csv(result + 'data.csv', index=False)

"""# Health (clinical group)"""

#read the data
health_clinical = pd.read_csv(data + 'data.csv',engine='python')
health_clinical.insert(0, 'idx', health_clinical.index) #Add a new column named "index" using the existing DataFrame index
health_clinical

health_clinical.shape

# Apply the sentiment analysis function to the 'clean_text' column
health_clinical[['VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'VADER_sentiment']] = health_clinical['clean_text'].apply(analyze_VADER_sentiment)
health_clinical

# save the results to a new cvs. file
health_clinical.to_csv(result + 'data.csv', index=False)