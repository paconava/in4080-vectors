# IN4080 Mandatory Assignment 3: Vector Semantics

- Your answer should be delivered in devilry no later than Tuesday, 13 November at 23:59
- We assume that you have read and are familiar with IFI's requirements and guidelines for
mandatory assignments:
    - https://www.uio.no/english/studies/examinations/compulsory-activities/mn-ifimandatory.html
    - https://www.uio.no/english/studies/examinations/compulsory-activities/mn-ifiguidelines.html
- This is an individual assignment. You should not deliver joint submissions.
- You may redeliver in Devilry before the deadline, but include all files in the last delivery.
Only the last delivery will be read!
- It should not be necessary to add any files, just edit the files that already exist in the `assignment` folder
- Zip the `assigment` folder, now edited so that they contain your solutions. This is your submission. 
- Name your submission `<username>_in4080_submission_3`

## Introduction
This assignment has two main parts:
1. Implement basic functions and concepts
2. Getting hands on experience with word2vec

## Structure
The structure of the assignment are based on **stubs** and **tests**:
- **Stubs** are incomplete functions, where you will write the code
- **Tests** are are there to test the code you have written. When you pass all the tests, you _may_ have a perfect assignment.
  This depends of the quality of the code, of course, and not just passing the tests ;)

You'll do one part using no libraries, and one part using libraries such
as NLTK, scikit-learn and numpy.

### Part 1a: Implement basic functions and concepts _without_ using libraries
    
In the first part, you'll implement some functions and concepts we have talked about in the lectures:
- TF-IDF
- Cosine distance
- Term-document matrix
- Term-context matrix
- Find most similar document and word based on TF-IDFs

This part consist of two parts, with corresponding stubs and tests.
In the first stub, you are only allowed to use basic python libraries, 
so nothing from the requirements

Hint: My imports in the solution are:

```
from collections import defaultdict
from math import log, sqrt
from typing import List, Dict, Tuple
``` 

Relevant files are:
- `assignment/part_one_no_libs.py.py`
- `tests/test_part_one_no_libs.py`

### Part 1b: Implement basic functions and concepts _using_ libraries

In part 1b, the tests you need to pass are exactly the same as in part 1, but we'll use
libraries instead of implementing it all by ourselves.

Hint: My imports in the solution are:
```
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ngrams
``` 

Relevant files are:
- `assignment/part_one_with_libs.py`
- `tests/test_part_one_with_libs.py`

## Part 2: Word2vec

**Information about this part is not ready yet, and will be out soon!** 

## Practical Information
How to start working:
- Get the code from github
    - `git clone git@github.uio.no:fredrijo/in4080-vectors.git`
    - See [Github help](https://help.github.com/articles/cloning-a-repository/) if you are stuck
- Create a python 3.6 environment for this test, e.g.
    - `conda create -n inf4080-vectors python=3.6`
- Activate the environment, e.g.
    - `source activate inf4080-vectors`
- Install the requirements:
    - `pip install `
- Start hacking:

    ![Hacking](https://media.giphy.com/media/LQYJ3DVpfB3va/giphy.gif)
- Run all tests:
    - `python -m unittest discover tests`
- During development, you can also run a single test suite:
    - `python -m unittest tests.test_part_one_no_libs.TestsPartOneNoLibraries`
- ...or a single test:
    - `python -m unittest tests.test_part_one_no_libs.TestsPartOneNoLibraries.test_most_similar_documents`

#### A couple of other notes
* Look at the tests and the method documentations (docstrings) before you start implementing,
to make sure you understand what to do.
* I've used typing in the functions, e.g. `def some_method(x: int) -> List[float]`
The typing here means that the argument `x` should be of type `int`, 
and that the return value should be a list of integers, written as `List[int]`.
This is considered good practice when developing python in a professional setting, 
but the code will run fine if you disregard the typing.
