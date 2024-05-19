'''
source: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#tutorial-setup
'''
from sklearn.datasets import fetch_20newsgroups

# only choosing 4 categories and parse them as dataset.
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print_prefix = "--> "
# the filenames are loaded as data. hence can use both filenames and data vars to refer.
print(print_prefix, twenty_train.target_names, len(twenty_train.filenames), len(twenty_train.data))

# basically the 4 categories loaded
print(print_prefix, "Number of targets (categories)", len(twenty_train.target_names))
print(print_prefix, "target name: ", twenty_train.target_names)

# target_names are loaded as vectors (i.e., index in this case) for faster processing by scikit.
print(print_prefix, "len(target): ", len(twenty_train.target))
#  these are just indices of categories array (which is of length 4) we chose. So all these would be between 0 and 3
print(print_prefix, "vector form of target: ", twenty_train.target[:6])
print(print_prefix, "Reverse name lookup of tagets using vectors: ",[twenty_train.target_names[vector] for vector in twenty_train.target[:5]])


### NEXT: FEATURE EXTRACTION 
from sklearn.feature_extraction.text import CountVectorizer

# Vectorize the dataset.
## By vectorizing, it tokenizes the data corpus such as stopwords, stems, etc., 
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

# of form (X[i,j] k); where i -> document_index, j -> word_index_in_ith_doc, k -> frequency_of_the word.
print(print_prefix, "Vector: ", X_train_counts[:1])
# but the above has a lot of bias for longer documents (which will have more frequent words) and repetitive words across documents.


## Further vectorization to enhance Feature extraction and search relevance.
# Tf transformation
from sklearn.feature_extraction.text import TfidfTransformer
#  just tf and not idf
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(print_prefix, " Tf transformed: ", X_train_tf[:1])

# tf-idf transformation
tf_idf_transformer = TfidfTransformer()
# one-liner to fit and transform
X_trian_tfidf = tf_idf_transformer.fit_transform(X_train_counts)
