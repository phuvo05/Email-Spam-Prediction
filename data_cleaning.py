
from sklearn.feature_extraction.text import CountVectorizer



corpus = [
    "The csv file contains 5172 rows, each row for each email. ",
    "There are 3002 columns.",
    "The first column indicates Email name. The name has been set with numbers and not recipients' name to protect privacy."
]


vectorizer = CountVectorizer () # ngram_range = (1,2), stop_words = "english"

data = vectorizer.fit_transform(corpus)
vocabulary = vectorizer.get_feature_names_out()

print('vocabulary:')
print(vocabulary)

result = data.toarray()
print('Bag of words:')
print(result)