#set up
import pandas as pd
import numpy as np

"""#Health_social group: combine two csv. files together

## read the file
"""

df1 = pd.read_csv(data + 'data.csv')
df1.head(2)

df2 = pd.read_csv(data + 'data.csv')
df2.head(2)

"""## the basic info of each csv"""

print("'data.csv':")
print("Number of Rows:", df1.shape[0])             # Get number of rows
print("Number of Columns:", df1.shape[1])          # Get number of columns

print("Column Names:", df1.columns.tolist())       # Get column names

print("'data.csv':")
print("Number of Rows:", df2.shape[0])             # Get number of rows
print("Number of Columns:", df2.shape[1])          # Get number of columns

print("Column Names:", df2.columns.tolist())       # Get column names

# Find columns that are in df1 but not in df2
columns_df1 = set(df1.columns)
columns_df2 = set(df2.columns)
diff_df1 = columns_df1 - columns_df2
print("Columns in df1 but not in df2:", diff_df1)

# Find columns that are in df2 but not in df1
diff_df2 = columns_df2 - columns_df1
print("Columns in df2 but not in df1:", diff_df2)

# Find columns that are common to both DataFrames
common_columns = columns_df1.intersection(columns_df2) #use inetrsection ()

"""## combine two CSV files （Keeps Only Common Columns）"""

merged_df = pd.concat([df1[common_columns], df2[common_columns]], ignore_index=True)

merged_df.head(2)

print(df1.shape)
print(df2.shape)
print(merged_df.shape)

df1.shape[0] + df2.shape[0]== merged_df.shape[0]

"""## clean the "text" column: remove emoji, URLs, hashtags, and lowercase"""

# Function to clean text
import re
import pandas as pd

def clean_text(text):
    # Check if the input is a string
    if isinstance(text, str):

        #Convert text to lowercase
        text = text.lower()

        # Remove new lines and extra spaces
        text = text.replace('\n', ' ').strip()

        # Remove hyperlinks (URLs)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove hashtags
        text = re.sub(r'#\w+', '', text)

        # Remove emojis (basic pattern to match most common emojis)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        return text
    else:
        # Handle non-string values (e.g., return empty string or original value)
        return str(text) # or return '' or return text

# Apply the cleaning function to the 'text' column
merged_df['clean_text'] = merged_df['text'].apply(clean_text)

merged_df.head(2)

# save the results to a new cvs. file
merged_df.to_csv(result + 'data.csv', index=False)



print("Number of Rows:", merged_df.shape[0])             # Get number of rows
print("Number of Columns:", merged_df.shape[1])          # Get number of columns