import pandas as pd
import numpy as np
import re

"""# read the data"""

Delaware_AD = pd.read_csv(data + 'data.csv')
Delaware_AD.head(2)

Delaware_health = pd.read_csv(data + 'data.csv')
Delaware_health.head(2)

"""# define function to restore the punctatio

**Notice**:

> The "content_semi_clean" column has removed final punctuation like periods (.), we need to  restore them by checking the original content column and appending the appropriate punctuation to the cleaned text.

> The content column can sometimes include multiple types of metadata, such as [+ exc], timestamps (512280_513039), and annotations (e.g., [: Ella] [* s:r]). To restore punctuation to content_semi_clean reliably, we need to extract the meaningful text before any of these metadata and then append the punctuation correctly.
"""

# Function to restore punctuation

def restore_punctuation(row):
    original = row['content']  # Original content
    cleaned = row['content_semi_clean']  # Cleaned content

    # Step 1: Remove metadata like timestamps or brackets
    text_without_metadata = re.sub(r'\[.*?\]|\.*?\|\d+_\d+', '', original).strip()

    # Step 2: Check the last character of the meaningful text
    last_char = text_without_metadata[-1] if len(text_without_metadata) > 0 else ''

    # Step 3: If the last character is punctuation, append it to the cleaned text
    # Check if cleaned is a string before applying strip()
    if isinstance(cleaned, str) and last_char in ".!?":
        return cleaned.strip() + last_char

    # Step 4: If no punctuation is found or cleaned is not a string, return the cleaned value as is
    return cleaned

#Apply the function to restore punctuation
Delaware_AD['text_clean'] = Delaware_AD.apply(restore_punctuation, axis=1)
Delaware_AD

#  Apply the function to restore punctuation
Delaware_health['text_clean'] = Delaware_health.apply(restore_punctuation, axis=1)
Delaware_health

# save the results
Delaware_AD.to_csv(data + 'data.csv', index=False)

# save the results
Delaware_health.to_csv(data + 'data.csv', index=False)