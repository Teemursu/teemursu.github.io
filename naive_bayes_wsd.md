
#                                      English word sense disambiguation

## The task
_Create a machine learning system that can disambiguate the correct sense of a word in context. Disambiguate the four words hard, interest, line, and serve into the senses given in the Senseval 2 corpus (NLTK Corpus HOWTO: Senseval). You can augment your data with other corpora as well. You can perform either supervised or unsupervised machine learning._

#### Why this task? 

Generally, the Turing test has been thought of as a test in which (currently) a machine only mimics human behavior, and does not necessarily or exactly show intelligence. With this in mind, I'm interested in exploring ways in which we could actually teach a machine to "understand" language. In this sense, The Winograd Schemas and other NLP focused tasks in which a machine would have to identify semantic meanings would, in my opinion, move us closer to teach a machine a certain kind of intelligence. 

#### Naive Bayes

For this task I will use the Naive Bayes algorithm, which is a conditional probability model. It works by taking a dataset that contains classified instances. By extracting features from these instances and checking the appropriate classification for each instance, the algorithm calculates probabilities as to which extracted features belong to which classification. In this sense, Naive Bayes is a supervised learning algorithm, since we feed it training examples, or train data, and receive an output based on the algorithm's calculations. 

It is appropriate for this task because it does not require a large training set to be able to learn from one. It is also less sensitive to irrelevant features, as we have a multiple labels for each of the four words.


#### You need to describe what data you plan to use and how it will be partitioned into training, development/validation and test sets.

I am using the Senseval corpus. After randomization, I will split the data into training and testing sets. 
As for extracting features, I am planning on using a) context words (as in, words that appear around the focus word) and b) the 'senses' category, which represents the exact meaning of the focus word.



```python
# Import some necessary modules
import nltk, random
from nltk.corpus import senseval
```

Here we import some necessary modules. For this task, we only need a few.

`nltk` contains its Naive Bayes classifier and the Senseval corpus, which we will use as our data set.

We use `random` to shuffle the training set, so that we can evaluate the model properly, as we will get varying results each time we run the program.


```python
print("All fileids:", senseval.fileids())
print()
for fileid in senseval.fileids():
    print(senseval.instances(fileid)[0])
    print()

```

    All fileids: ['hard.pos', 'interest.pos', 'line.pos', 'serve.pos']
    
    SensevalInstance(word='hard-a', position=20, context=[('``', '``'), ('he', 'PRP'), ('may', 'MD'), ('lose', 'VB'), ('all', 'DT'), ('popular', 'JJ'), ('support', 'NN'), (',', ','), ('but', 'CC'), ('someone', 'NN'), ('has', 'VBZ'), ('to', 'TO'), ('kill', 'VB'), ('him', 'PRP'), ('to', 'TO'), ('defeat', 'VB'), ('him', 'PRP'), ('and', 'CC'), ('that', 'DT'), ("'s", 'VBZ'), ('hard', 'JJ'), ('to', 'TO'), ('do', 'VB'), ('.', '.'), ("''", "''")], senses=('HARD1',))
    
    SensevalInstance(word='interest-n', position=18, context=[('yields', 'NNS'), ('on', 'IN'), ('money-market', 'JJ'), ('mutual', 'JJ'), ('funds', 'NNS'), ('continued', 'VBD'), ('to', 'TO'), ('slide', 'VB'), (',', ','), ('amid', 'IN'), ('signs', 'VBZ'), ('that', 'IN'), ('portfolio', 'NN'), ('managers', 'NNS'), ('expect', 'VBP'), ('further', 'JJ'), ('declines', 'NNS'), ('in', 'IN'), ('interest', 'NN'), ('rates', 'NNS'), ('.', '.')], senses=('interest_6',))
    
    SensevalInstance(word='line-n', position=67, context=[('the', 'DT'), ('company', 'NN'), ('argued', 'VBD'), ('that', 'IN'), ('its', 'PRP$'), ('foreman', 'NN'), ('needn', 'NN'), ("'t", 'NN'), ('have', 'VBP'), ('told', 'VBN'), ('the', 'DT'), ('worker', 'NN'), ('not', 'RB'), ('to', 'TO'), ('move', 'VB'), ('the', 'DT'), ('plank', 'NN'), ('to', 'TO'), ('which', 'WDT'), ('his', 'PRP$'), ('lifeline', 'NN'), ('was', 'VBD'), ('tied', 'VBN'), ('because', 'IN'), ('"', '"'), ('that', 'WDT'), ('comes', 'VBZ'), ('with', 'IN'), ('common', 'JJ'), ('sense', 'NN'), ('.', '.'), ('"', '"'), ('the', 'DT'), ('commission', 'NN'), ('noted', 'VBD'), (',', ','), ('however', 'RB'), (',', ','), ('that', 'IN'), ('dellovade', 'NNP'), ('hadn', 'NN'), ("'t", 'NN'), ('instructed', 'VBD'), ('its', 'PRP$'), ('employees', 'NNS'), ('on', 'IN'), ('how', 'WRB'), ('to', 'TO'), ('secure', 'VB'), ('their', 'PRP$'), ('lifelines', 'NNS'), ('and', 'CC'), ('didn', 'VBD'), ("'t", 'NN'), ('heed', 'NN'), ('a', 'DT'), ('federal', 'JJ'), ('inspector', 'NN'), ("'s", 'POS'), ('earlier', 'JJR'), ('suggestion', 'NN'), ('that', 'IN'), ('the', 'DT'), ('company', 'NN'), ('install', 'VB'), ('special', 'JJ'), ('safety', 'NN'), ('lines', 'NNS'), ('inside', 'IN'), ('the', 'DT'), ('a-frame', 'NNP'), ('structure', 'NN'), ('it', 'PRP'), ('was', 'VBD'), ('building', 'VBG'), ('.', '.')], senses=('cord',))
    
    SensevalInstance(word='serve-v', position=42, context=[('some', 'DT'), ('tart', 'JJ'), ('fruits', 'NNS'), ('mixed', 'VBN'), ('with', 'IN'), ('greens', 'NNS'), ('make', 'VBP'), ('a', 'DT'), ('nice', 'JJ'), ('contrast', 'NN'), ('with', 'IN'), ('rich', 'JJ'), ('meat', 'NN'), ('dishes', 'NNS'), ('(', '('), ('see', 'VB'), ('orange', 'NNP'), ('and', 'CC'), ('onion', 'NNP'), ('salad', 'NNP'), (',', ','), ('page', 'NN'), ('111', 'CD'), (')', 'SYM'), (',', ','), ('but', 'CC'), ('if', 'IN'), ('you', 'PRP'), ('like', 'VB'), ('to', 'TO'), ('follow', 'VB'), ('the', 'DT'), ('meat', 'NN'), ('course', 'NN'), ('with', 'IN'), ('sweet', 'JJ'), ('fruit', 'NN'), (',', ','), ('it', 'PRP'), ('seems', 'VBZ'), ('wiser', 'JJR'), ('to', 'TO'), ('serve', 'VB'), ('it', 'PRP'), ('plain', 'JJ'), ('with', 'IN'), ('a', 'DT'), ('good', 'JJ'), ('sharp', 'JJ'), ('cheese', 'NN'), ('and', 'CC'), ('let', 'VB'), ('it', 'PRP'), ('take', 'VB'), ('the', 'DT'), ('place', 'NN'), ('of', 'IN'), ('a', 'DT'), ('sweet', 'JJ'), ('or', 'CC'), ('dessert', 'NN'), ('course', 'NN'), ('.', '.'), ('if', 'IN'), ('you', 'PRP'), ('insist', 'VBP'), ('on', 'IN'), ('serving', 'VBG'), ('fruit', 'NN'), ('as', 'IN'), ('a', 'DT'), ('salad', 'NN'), (',', ','), ('don', 'VB'), ("'t", 'NN'), ('cut', 'NN'), ('it', 'PRP'), ('into', 'IN'), ('cubes', 'NNS'), ('and', 'CC'), ('mix', 'NN'), ('it', 'PRP'), ('up', 'RB'), ('.', '.')], senses=('SERVE10',))
    
    

There are four file ids in the senseval corpus, one for each of the words. Each fileid, or word, contains a series of instances. Above is an example instance for each of the words. The instance contains the following information:

- Which word is in question, followed by its POS (Part of Speech) tag. E.g. word='hard-a' refers to the word 'hard' being an adjective.

- The position of our word, telling us how many words are preceding our focus word (remember to count from 0!)

- The context, which is a sentence or a longer sequence of words surrounding our focus word. Each of the tokens are tuples of two strings: the word itself and its POS tag.

- The sense of the focus word, which is the label in our model.


```python
''' 
A simple function to extract the label from each of the instances. The indexes the listed labels also correspond to the 
order of the instances in a word fileid.

The returned list will be used later in creating our featureset.
'''
def get_category(pos):
    
    category = []
    for inst in senseval.instances(pos):
        category.append(inst.senses)
    return category
```


```python
'''
A function to create our featureset, which is returned as a dictionary of all of the features of an instance.
''' 
def get_features(inst):
    
    features = {}
    p = inst.position
    
    '''
    As we are using the position of the focus word in an instance to get the previous and next words and tags,
    we might get errors where there are not enough elements after the focus words. For this reason, we add the tuple
    below to each of the instances. 
    '''
    inst.context.append(('<END>','<END>'))

    '''
    Because the Senseval corpus contains some unexpected elements and other quirks, such as the string 'FRASL' 
    among some of the contexts (which expectedly are lists of tuples), we need to define each of the features with testing.
    if an instance contains some of these elements that make our program fail, we instead return an empty set.
    
    Additionally, we ignore all tokens that are not longer than one character. This is because we want to ignore tokens
    such as punctuation, as well as all the random quirks of the data set. 
    '''
    
    try: 
        left_word = ' '.join(w for (w,t) in inst.context[p-1:p] if len(w) > 1)
        right_word = ' '.join(w for (w,t) in inst.context[p+1:p+2] if len(w) > 1)
        more_left_word = ' '.join(w for (w,t) in inst.context[p-2:p] if len(w) > 1)
        more_right_word = ' '.join(w for (w,t) in inst.context[p+1:p+3] if len(w) > 1)
        left_tag = ' '.join(t for (w,t) in inst.context[p-1:p] if len(t) > 1)
        right_tag = ' '.join(t for (w,t) in inst.context[p+1:p+2] if len(t) > 1)
        more_left_tag = ' '.join(t for (w,t) in inst.context[p-2:p] if len(t) > 1)
        more_right_tag = ' '.join(t for (w,t) in inst.context[p+1:p+3] if len(t) > 1)
    except: 
        return features
    '''
    The extracted features are listed below.
    '''
    features['1 Previous tag'] = left_tag
    features['1 Next tag'] = right_tag
    features['2 Previous tags'] = more_left_tag
    features['2 Next tags'] = more_right_tag
    features['1 Previous word'] = left_word
    features['1 Next word'] = right_word
    features['2 Previous words'] = more_left_word
    features['2 Next words'] = more_right_word
    return features
```


```python
'''
Because of the length of the label's name, it does not appear in the NLTK classifier's most informative features list. 
Hence, we change it into a shorter one.

'''
interest_unchanged = get_category('interest.pos')
interest_c = []
for tuple in interest_unchanged:
    if tuple == ('interest_1',):
        interest_c.append('inte_1')
    if tuple == ('interest_2',):
        interest_c.append('inte_2')
    if tuple == ('interest_3',):
        interest_c.append('inte_3')
    if tuple == ('interest_4',):
        interest_c.append('inte_4')
    if tuple == ('interest_5',):
        interest_c.append('inte_5')
    if tuple == ('interest_6',):
        interest_c.append('inte_6')
```


```python
'''
For each word, we send all of the word's instances to the above function. We use the zip()function to create tuples of
an instance and the correct label for the instance. As we iterate over an instance of a word and the label of our category
list, each tuple will have the correct label, since its index in the list corresponds to the order of the word instances.

'''
interest_featureset = [(get_features(inst), c) for c,inst in zip(interest_c, senseval.instances('interest.pos'))]
hard_featureset = [(get_features(inst), c) for c,inst in zip(get_category('hard.pos'), senseval.instances('hard.pos'))]
line_featureset = [(get_features(inst), c) for c,inst in zip(get_category('line.pos'), senseval.instances('line.pos'))]
serve_featureset = [(get_features(inst), c) for c,inst in zip(get_category('serve.pos'), senseval.instances('serve.pos'))] 
```


```python
print('Example of featureset for the word "hard":\n\n', hard_featureset[30])
print()
print('Amount of featuresets' 
      '\n for the word "hard:"', len(hard_featureset), 
      '\n for the word "interest:"', len(interest_featureset),
      '\n for the word "line",', len(line_featureset),
      '\n and the word "serve"', len(serve_featureset))
```

    Example of featureset for the word "hard":
    
     ({'1 Previous tag': 'DT', '1 Next tag': 'NN', '2 Previous tags': 'VBP DT', '2 Next tags': 'NN IN', '1 Previous word': '', '1 Next word': 'time', '2 Previous words': 'have', '2 Next words': 'time with'}, ('HARD1',))
    
    Amount of featuresets
     for the word "hard:" 4333 
     for the word "interest:" 2368 
     for the word "line", 4146 
     and the word "serve" 4378
    

So far we have approximately 15,000 features available for input.

As per the example above, the phrase "have hard time with" is classified by the label "HARD1" which represents (from the Senseval corpus) "not easy, requiring great physical or mental." The phrase is also sliced into features of previous words, next words, previous tags and next tags. With this, we have a tuple where the first element is a dictionary and the second element is the label (because of the format of the corpus, the label is a tuple as well with no second element.)

Below, we randomize the featuresets of each of the words, and use quarter for the training set and another quarter for the testing set. 


```python
def get_size(fileid):
    size = int(len(senseval.instances(fileid)) * 0.25)
    return size

size = get_size('hard.pos')
random.shuffle(hard_featureset)
hard_train_set, hard_test_set = hard_featureset[size:], hard_featureset[:size]

size = get_size('interest.pos')
random.shuffle(interest_featureset)
interest_train_set, interest_test_set = interest_featureset[size:], interest_featureset[:size]

size = get_size('serve.pos')
random.shuffle(serve_featureset)
serve_train_set, serve_test_set = serve_featureset[size:], serve_featureset[:size]

size = get_size('line.pos')
random.shuffle(line_featureset)
line_train_set, line_test_set = line_featureset[size:], line_featureset[:size]
```


```python
def word_bayes(train_set, test_set, word):
    bayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(word, "Naive Bayes accuracy percent:", (nltk.classify.accuracy(bayes_classifier, test_set))*100,"%")
    print()
    print(bayes_classifier.show_most_informative_features(20))
```

## Carrying out the evaluation

We can use the method `nltk.FreqDist()` to compute the distribution of the senses of the words hard, interest, line and serve in the Senseval corpus. By choosing the most common sense and comparing it to the others, we can calculate the following baselines for the following words:

- "hard": 79.7%
- "serve": 41.4%
- "interest": 52.9%
- "line": 53.5%

It is notable that a significant portion of the category "hard" senses is "HARD1." Similarly, each of the words have one sense that is significantly more common than the other meanings. In this sense, as we consider the baseline percentage and evaluation in general, we should not only consider the "worst case scenario" of us guessing the category randomly and correctly at the same time, but also see whether we can distinguish meanings from the most common senses label (and if not, then why?).


```python
hard_dist = nltk.FreqDist([i.senses[0] for i in senseval.instances('hard.pos')])
hard_baseline = hard_dist.freq('HARD1')
#hard_dist FreqDist({'HARD1': 3455, 'HARD2': 502, 'HARD3': 376})
#hard_baseline 0.797369028386799

serve_dist = nltk.FreqDist([i.senses[0] for i in senseval.instances('serve.pos')])
serve_baseline = serve_dist.freq('SERVE10')
# serve_dist FreqDist({'SERVE10': 1814, 'SERVE12': 1272, 'SERVE2': 853, 'SERVE6': 439})
# serve_baseline 0.4143444495203289

interest_dist = nltk.FreqDist([i.senses[0] for i in senseval.instances('interest.pos')])
interest_baseline = interest_dist.freq('interest_6')
interest_baseline
# interest_distFreqDist({'interest_6': 1252, 'interest_5': 500, 'interest_1': 361, 
#                        'interest_4': 178, 'interest_3': 66, 'interest_2': 11})
# interest_baseline 0.5287162162162162

line_dist = nltk.FreqDist([i.senses[0] for i in senseval.instances('line.pos')])
line_baseline = line_dist.freq('product')
# line_dist FreqDist({'product': 2217, 'phone': 429, 'text': 404, 'division': 374, 'cord': 373, 'formation': 349})
# line_baseline 0.5347322720694645
```

### Linguistic Findings

It seems that the words and tags that come after the focus word are more important than previous ones. This can be seen in the word "hard." For example, it becomes apparent that we mean "difficult" when "hard" is followed by "to" (e.g. "It is hard to do.")

Same phenomenon can be seen with the word category "interest", with the 11 most informative features being either next words or tags. The most informative feature, one next tag being NNS, probably represents cases like "interest rates."

The WH-determiner seems to be an important factor when determining the sense of the word "serve." 


```python

```




    ' \n| (\'HARD1\',):       | "not easy, requiring great physical or mental",                               |\n| (\'HARD2\',):       | "dispassionate",                                                              |\n| (\'HARD3\',):       | "resisting weight or pressure",                                               |\n| (\'interest_1\',):  | "readiness to give attention",                                                |\n| (\'interest_2\',):  | "quality of causing attention to be given to",                                |\n| (\'interest_3\',):  | "activity, etc. that one gives attention to",                                 |\n| (\'interest_4\',):  | "advantage, advancement or favor",                                            |\n| (\'interest_5\',):  | " a share in a company or business",                                          |\n| (\'interest_6\',):  | "money paid for the use of money",                                            |\n| (\'cord\',):        | "something (as a cord or rope) that is long and thin and flexible",           |\n| (\'formation\',):   | "a formation of people or things one beside another",                         |\n| (\'text\',):        | "text consisting of a row of words written across a page or computer screen", |\n| (\'phone\',):       | "a telephone connection",                                                     |\n| (\'product\',):     | "a particular kind of product or merchandise",                                |\n| (\'division\',):    | "a conceptual separation or distinction",                                     |\n| (\'SERVE12\',):     | "do duty or hold offices; serve in a specific function",                      |\n| (\'SERVE10\',):     | "provide (usually but not necessarily food)",                                 |\n| (\'SERVE2\',):      | "serve a purpose, role, or function",                                         |\n| (\'SERVE6\',):      | "be used by; as of a utility"                                                 |\n'




```python
word_bayes(hard_train_set, hard_test_set, "Hard -")
word_bayes(interest_train_set, interest_train_set, "Interest -")
word_bayes(serve_train_set, serve_test_set, "Serve -")
word_bayes(line_train_set, line_test_set, "Line -")

# | ('HARD1',):       | "not easy, requiring great physical or mental"                               |
# | ('HARD2',):       | "dispassionate"                                                              |
# | ('HARD3',):       | "resisting weight or pressure"                                               |
# | ('interest_1',):  | "readiness to give attention"                                                |
# | ('interest_2',):  | "quality of causing attention to be given to"                                |
# | ('interest_3',):  | "activity, etc. that one gives attention to"                                 |
# | ('interest_4',):  | "advantage, advancement or favor"                                            |
# | ('interest_5',):  | " a share in a company or business"                                          |
# | ('interest_6',):  | "money paid for the use of money"                                            |
# | ('cord',):        | "something (as a cord or rope) that is long and thin and flexible"           |
# | ('formation',):   | "a formation of people or things one beside another"                         |
# | ('text',):        | "text consisting of a row of words written across a page or computer screen" |
# | ('phone',):       | "a telephone connection"                                                     |
# | ('product',):     | "a particular kind of product or merchandise"                                |
# | ('division',):    | "a conceptual separation or distinction"                                     |
# | ('SERVE12',):     | "do duty or hold offices; serve in a specific function"                      |
# | ('SERVE10',):     | "provide (usually but not necessarily food)"                                 |
# | ('SERVE2',):      | "serve a purpose, role, or function"                                         |
# | ('SERVE6',):      | "be used by; as of a utility"                                                |
```

    Hard - Naive Bayes accuracy percent: 83.84118190212374 %
    
    Most Informative Features
                 1 Next word = 'to'            HARD1 : HARD2  =    189.3 : 1.0
                  1 Next tag = 'TO'            HARD1 : HARD2  =    141.9 : 1.0
            2 Previous words = "it 's"         HARD1 : HARD2  =    112.6 : 1.0
                 2 Next tags = 'TO VB'         HARD1 : HARD3  =     77.3 : 1.0
                 2 Next tags = 'NN CC'         HARD2 : HARD1  =     76.3 : 1.0
             2 Previous tags = 'PRP VBZ'       HARD1 : HARD3  =     68.8 : 1.0
                 1 Next word = 'work'          HARD2 : HARD1  =     65.5 : 1.0
             1 Previous word = "'s"            HARD1 : HARD3  =     59.8 : 1.0
                2 Next words = 'work'          HARD2 : HARD1  =     59.3 : 1.0
                 2 Next tags = 'NN NNS'        HARD3 : HARD1  =     58.3 : 1.0
                2 Next words = 'work and'      HARD2 : HARD1  =     49.9 : 1.0
                 1 Next word = 'line'          HARD2 : HARD1  =     47.0 : 1.0
             1 Previous word = 'no'            HARD2 : HARD1  =     43.3 : 1.0
                  1 Next tag = 'VBN'           HARD3 : HARD1  =     42.3 : 1.0
                 1 Next word = 'place'         HARD3 : HARD1  =     41.6 : 1.0
             2 Previous tags = 'NNS IN'        HARD2 : HARD1  =     39.8 : 1.0
                 2 Next tags = 'JJ'            HARD3 : HARD1  =     29.4 : 1.0
                 1 Next word = 'for'           HARD1 : HARD2  =     28.0 : 1.0
             2 Previous tags = 'VBN IN'        HARD3 : HARD1  =     25.7 : 1.0
              1 Previous tag = '``'            HARD2 : HARD1  =     22.9 : 1.0
    None
    Interest - Naive Bayes accuracy percent: 94.42567567567568 %
    
    Most Informative Features
                  1 Next tag = 'NNS'          inte_6 : inte_1 =    111.2 : 1.0
                 2 Next tags = 'IN VBG'       inte_1 : inte_6 =     74.6 : 1.0
                 1 Next word = 'in'           inte_5 : inte_6 =     51.6 : 1.0
             1 Previous word = 'other'        inte_3 : inte_6 =     47.5 : 1.0
                  1 Next tag = 'VBP'          inte_3 : inte_6 =     43.7 : 1.0
                 1 Next word = 'of'           inte_4 : inte_6 =     41.5 : 1.0
                 2 Next tags = 'NNS'          inte_6 : inte_1 =     39.1 : 1.0
              1 Previous tag = 'VBN'          inte_1 : inte_5 =     24.7 : 1.0
                 2 Next tags = 'TO VB'        inte_4 : inte_6 =     23.6 : 1.0
                 2 Next tags = 'DT'           inte_3 : inte_6 =     23.2 : 1.0
                 2 Next tags = 'NNS IN'       inte_6 : inte_5 =     22.9 : 1.0
                2 Next words = 'in the'       inte_5 : inte_4 =     21.5 : 1.0
                  1 Next tag = 'NN'           inte_6 : inte_1 =     21.4 : 1.0
             2 Previous tags = 'VB JJ'        inte_3 : inte_4 =     19.9 : 1.0
              1 Previous tag = 'TO'           inte_2 : inte_6 =     19.7 : 1.0
                  1 Next tag = 'TO'           inte_2 : inte_6 =     19.6 : 1.0
              1 Previous tag = ''             inte_6 : inte_5 =     17.8 : 1.0
             2 Previous tags = 'PRP$ NN'      inte_5 : inte_6 =     17.7 : 1.0
             2 Previous tags = 'VBP VBN'      inte_1 : inte_6 =     17.6 : 1.0
             1 Previous word = 'of'           inte_1 : inte_5 =     17.0 : 1.0
    None
    Serve - Naive Bayes accuracy percent: 78.70201096892139 %
    
    Most Informative Features
              1 Previous tag = 'WDT'          SERVE6 : SERVE1 =     85.4 : 1.0
                2 Next words = 'as'           SERVE2 : SERVE1 =     68.4 : 1.0
             2 Previous tags = 'WDT'          SERVE6 : SERVE1 =     60.0 : 1.0
                 1 Next word = 'as'           SERVE2 : SERVE1 =     57.0 : 1.0
             2 Previous tags = 'WP'           SERVE1 : SERVE1 =     40.9 : 1.0
              1 Previous tag = 'WP'           SERVE1 : SERVE2 =     40.5 : 1.0
                2 Next words = 'on the'       SERVE1 : SERVE1 =     38.5 : 1.0
                2 Next words = ''             SERVE1 : SERVE1 =     35.3 : 1.0
            2 Previous words = 'who'          SERVE1 : SERVE1 =     34.8 : 1.0
             1 Previous word = 'that'         SERVE2 : SERVE1 =     33.9 : 1.0
             1 Previous word = 'it'           SERVE2 : SERVE1 =     33.0 : 1.0
                 1 Next word = 'under'        SERVE1 : SERVE1 =     31.9 : 1.0
                2 Next words = 'to'           SERVE1 : SERVE6 =     31.7 : 1.0
            2 Previous words = 'which'        SERVE6 : SERVE1 =     31.2 : 1.0
             1 Previous word = 'before'       SERVE1 : SERVE1 =     30.4 : 1.0
             2 Previous tags = 'NN CC'        SERVE1 : SERVE2 =     29.5 : 1.0
             2 Previous tags = 'WDT MD'       SERVE2 : SERVE1 =     29.0 : 1.0
                 1 Next word = 'in'           SERVE1 : SERVE2 =     27.8 : 1.0
             2 Previous tags = 'NN WDT'       SERVE2 : SERVE1 =     26.9 : 1.0
                 2 Next tags = 'CD'           SERVE1 : SERVE1 =     26.8 : 1.0
    None
    Line - Naive Bayes accuracy percent: 72.00772200772201 %
    
    Most Informative Features
                 1 Next word = 'between'      divisi : produc =    152.9 : 1.0
             1 Previous word = 'in'           format : produc =     63.9 : 1.0
                 1 Next word = 'of'           produc : phone  =     62.8 : 1.0
             1 Previous word = 'telephone'     phone : text   =     45.8 : 1.0
              1 Previous tag = 'IN'           format : divisi =     43.3 : 1.0
             1 Previous word = 'long'         format : text   =     40.5 : 1.0
             2 Previous tags = 'VB IN'        format : produc =     38.3 : 1.0
                 2 Next tags = 'IN JJ'        produc : phone  =     32.3 : 1.0
             1 Previous word = 'fine'         divisi : text   =     31.5 : 1.0
             1 Previous word = 'new'          produc : format =     30.8 : 1.0
                 2 Next tags = 'IN NNS'       format : cord   =     30.2 : 1.0
            2 Previous words = 'new'          produc : format =     28.8 : 1.0
            2 Previous words = 'on the'        phone : format =     27.9 : 1.0
              1 Previous tag = 'PRP$'           cord : phone  =     26.8 : 1.0
                 1 Next word = 'like'           text : produc =     22.8 : 1.0
            2 Previous words = 'fine'         divisi : text   =     22.5 : 1.0
                 2 Next tags = 'VBD RB'         cord : produc =     22.1 : 1.0
             2 Previous tags = 'CD'            phone : produc =     20.5 : 1.0
                 1 Next word = 'for'          format : divisi =     19.7 : 1.0
             2 Previous tags = 'DT VBG'         cord : produc =     19.1 : 1.0
    None
    



### Best & Worst Categories

When the model was in its early stages and it was only tested against the most common words in the whole context, the "hard" category was recognized the best. Nevertheless, the "hard" category was unrealistically high with consistently over 97% accuracy (only based on all words in the context), and the "interest" category now seems to have some interesting findings. 

This might relate to the fact that one of the categories is significantly more common. Because of this, the Naive Bayes algorithm might learn and apply general features to the most common category. This can be seen below, where we have made a program that predicts user input. There are some troubles of making the model predict other senses than the most common one of the category (e.g. "a hard rock" and other similar sentences are still predicted as "HARD1.")

In this sense, the word "hard" might be the hardest to get right, as its baseline is already fairly high, 79,9%, and because of the tendency of the model to predict the most common category. Nevertheless, it seems to perform decently, as evidenced by the most informative features. In fact, the most informative features do give us meaningful information.

"Line" and "interest" could also be considered as difficult word categories to get correctly. The least used senses occur less than hundred times in the data, and the least common sense for "interest" even occurs only 11 times. Still, "interest" gives us consistently the highest accuracy, and might be another evidence of the Naive Bayes algorithm learning "bias" towards the most common label.

### Overfitting? 

As can be seen from above, the model can be somewhat biased and generate approximation errors. In addition, as the training and test sets are shuffled and split, the model should not be prone to overfitting.



## Program for the user end: predicting sentences

While this was not part of the project, I implemented it out of my own interest. While the part of the code that recognizes which word has been inputted is very basic, it can be developed further once the environment allows us to do so. For example, we could use regex to recognize all conjugated word forms.


```python
from nltk import tokenize

def get_features(inst,p):
    features = {}
    all_words = []
    left_words = []
    right_words = []
    
    inst.append('<END>')
#    if inst.context[p+1] == 'FRASL':
#        inst.context[p+1] = (inst.context[p+1],inst.context[p+1])
#    if inst.context[p+2] == 'FRASL':
#        inst.context[p+2] = (inst.context[p+1],inst.context[p+2])
    try: 
        left_word = ' '.join(w for (w,t) in inst.context[p-1:p] if len(w) > 1)
        right_word = ' '.join(w for (w,t) in inst.context[p+1:p+2] if len(w) > 1)
        more_left_word = ' '.join(w for (w,t) in inst.context[p-2:p] if len(w) > 1)
        more_right_word = ' '.join(w for (w,t) in inst.context[p+1:p+3] if len(w) > 1)
        left_tag = ' '.join(t for (w,t) in inst.context[p-1:p] if len(t) > 1)
        right_tag = ' '.join(t for (w,t) in inst.context[p+1:p+2] if len(t) > 1)
        more_left_tag = ' '.join(t for (w,t) in inst.context[p-2:p] if len(t) > 1)
        more_right_tag = ' '.join(t for (w,t) in inst.context[p+1:p+3] if len(t) > 1)
    except: 
        return features
    
    features['1 Previous tag'] = left_tag
    features['1 Next tag'] = right_tag
    features['2 Previous tags'] = more_left_tag
    features['2 Next tags'] = more_right_tag
    features['1 Previous word'] = left_word
    features['1 Next word'] = right_word
    features['2 Previous words'] = more_left_word
    features['2 Next words'] = more_right_word
    return features

def guess_sense(text, word, train_set):
    
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    pos=text.find(word)
    text = tokenize.wordpunct_tokenize(text)
    tokenized_text = nltk.pos_tag(text)
    pos=text.index(word)
    guess = classifier.classify(get_features(tokenized_text, pos))
    SV_SENSE_MAP = {
    ('HARD1',): "not easy, requiring great physical or mental",
    ('HARD2',): "dispassionate",          
    ('HARD3',): "resisting weight or pressure",   
    ('interest_1',): "readiness to give attention",
    ('interest_2',): "quality of causing attention to be given to",
    ('interest_3',): "activity, etc. that one gives attention to",
    ('interest_4',): "advantage, advancement or favor", 
    ('interest_5',): " a share in a company or business",
    ('interest_6',): "money paid for the use of money", 
    ('cord',): "something (as a cord or rope) that is long and thin and flexible",
    ('formation',): "a formation of people or things one beside another",
    ('text',): "text consisting of a row of words written across a page or computer screen", 
    ('phone',): "a telephone connection",   
    ('product',): "a particular kind of product or merchandise", 
    ('division',): "a conceptual separation or distinction",   
    ('SERVE12',): "do duty or hold offices; serve in a specific function",       
    ('SERVE10',): "provide (usually but not necessarily food)", 
    ('SERVE2',): "serve a purpose, role, or function",        
    ('SERVE6',): "be used by; as of a utility"      
}
    x = SV_SENSE_MAP[guess]
    print('Hmm...')
    print('I think by "{}" you mean'.format(word), str(x))
    
text = input("Type a sentence with the word 'hard', 'line', 'serve' or 'interest'.\n")

if text.find('hard') > -1:
    guess_sense(text, 'hard', hard_train_set)
elif text.find('line') > -1:
    guess_sense(text, 'line', line_train_set)
elif text.find('serve') != -1:
    guess_sense(text, 'serve', serve_train_set)
elif text.find('interest') != -1:
    guess_sense(text, 'interest', interest_train_set)
else:
    print('Didn\'t find the word "hard", "line", "serve" or "interest".')

```

    Type a sentence with the word 'hard', 'line', 'serve' or 'interest'.
    rock hard
    Hmm...
    I think by "hard" you mean not easy, requiring great physical or mental
    
