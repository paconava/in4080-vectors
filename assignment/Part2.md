# Part 2: Train and investigate models for vector semantics

In this part, you'll get less direct instructions, and will have to google and read a bit in order to
complete the tasks. 

## Assignment 2a - Train a term2context model on the inaugural speeches

Delivery:
- Modified version of `assignment/train_term2context.py`
- Table of top 3 similar words in this file

Train a term2context model on the inaugural speeches, 
and get the 5 most similar words to the word "war". 
Use context size of `1` (1 word before and 1 word after each target word)

Hints:
- In this task, you shouldn't have to write much new code...
You should be able to reuse a lot of the functions you wrote in Part 1.
- Start with the test `test_part_one_with_libs.TestPartOneWithLibraries.test_most_similar_term`
- Modify the method `part_one_with_libs.get_most_similar_to` to return 5 most similar, and not just one.
Make sure you return the 5 words and scores for _the most similar_ terms.


### Term-Context: Most similar words to "war" 

| | Word | Score |
|:---:|:------:|:-------:|
| 1. | peace | 0.773883224004799 |
| 2. | conservation | 0.7627470588829093  |
| 3. | freedom | 0.7516372774683475 |
| 4. | july | 0.7285074038242858  |
| 5. | congress | 0.7265856751948191  |

## Assignment 2b - Train a word2vec model on the inaugural speeches

Delivery:
- Modified version of `assignment/train_word2vec.py`
- Table of top 5 similar words in this file

Train a word2vec model on the inaugural speeches, 
and get the 5 most similar words to the word "war".

In this task, you can reuse (copy paste) code from the first part.

A couple of notes:
- A word2vec model trains on sentences, so get the inaugural speeches as sentences
- In order to get reproducible results (i.e., the same as I did) for word2vec, you need to:
    - set the number of workers to 1
    - send in your own hashing function. Use `hashfxn` provided in the python script
    - let the algorithm run for 10 iterations
    - the word embedding size should be 100

### Word2Vec: Most similar words to "war" 

| | Word | Score |
|:---:|:------:|:-------:|
| 1. | conformity | 0.9789714813232422 |
| 2. | slavery | 0.9768784642219543  |
| 3. | itself | 0.9746180772781372 |
| 4. | majority | 0.9744900465011597  |
| 5. | nature | 0.9722113609313965  |


## Assignment 2c - Word embeddings on large corpora

Delivery:
- Modified version of `assignment/train_word2vec.py`
- Table of top 5 similar words to war, in the table below

The final part is quite simple, we'll have a look at what word embedding models look like when trained on real data.
In this particular exercise, we'll look at GloVe embeddings, as they are readily available.
Even though Word2Vec and GloVe are trained differently, the models have the same format: words and embeddings (vectors)

Instructions:
- Download the corpus `glove-wiki-gigaword-100` using the [gensim download API](https://github.com/RaRe-Technologies/gensim-data)
- Get the 5 most similar words to "war"
- Get the 5 most similar words to "norway"
- Get the 5 most similar words to "norway - war + peace"
- Given the last result, "norway - war + peace": Which country is more peaceful, Sweden or Denmark?

### Glove: Most similar words to "war" 

| | Word | Score |
|:---:|:------:|:-------:|
| 1. | wars | 0.7686851620674133 |
| 2. | conflict | 0.7660517692565918  |
| 3. | invasion | 0.7430229187011719 |
| 4. | military | 0.7365108132362366  |
| 5. | occupation | 0.7300143241882324  |

### Glove: Most similar words to "norway" 

| | Word | Score |
|:---:|:------:|:-------:|
| 1. | denmark | 0.8288264274597168 |
| 2. | sweden | 0.807325005531311  |
| 3. | finland | 0.8004074096679688 |
| 4. | iceland | 0.6992351412773132  |
| 5. | norwegian | 0.6949846148490906  |

### Glove: Most similar words to "norway - war + peace" 

| | Word | Score |
|:---:|:------:|:-------:|
| 1. | oslo | 0.6613224744796753 |
| 2. | sweden | 0.5961518883705139  |
| 3. | finland | 0.5912866592407227 |
| 4. | norwegian | 0.5682424902915955  |
| 5. | denmark | 0.5592893958091736  |