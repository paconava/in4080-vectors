import warnings
warnings.filterwarnings("ignore")
import gensim.downloader as api
"""
Test a GloVe model downloaded with the gensim downloader API
"""

info = api.info()  # show info about available models/datasets
model = api.load("glove-wiki-gigaword-100")  # download the model and return as object ready for use

mstw = model.most_similar("war", topn=5)
mstn = model.most_similar("norway", topn=5)
mstnmwpp = model.most_similar(positive=['norway', 'peace'], negative=['war'], topn=5)

print("5 most similar words to war:")
# print([k for k, v in mstw])
print(mstw)
# Print some more
print('---------')

print("5 most similar words to norway:")
# print([k for k, v in mstn])
print(mstn)
# Print some more
print('---------')

print("5 most similar words to norway minus war plus peace:")
# print([k for k, v in mstnmwpp])
print(mstnmwpp)
# Print some more
print('---------')