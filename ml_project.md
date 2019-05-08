
### English word sense disambiguation

#### The task
Create a machine learning system that can disambiguate the correct sense of a word in context. Disambiguate the four words hard, interest, line, and serve into the senses given in the Senseval 2 corpus (NLTK Corpus HOWTO: Senseval). You can augment your data with other corpora as well. You can perform either supervised or unsupervised machine learning.


The overall structure of the report must be given. At least there must be section headings and brief descriptions of what belongs to each section.

#### Why this task? 

What task are you trying to solve and why is this interesting?

    I am trying to 

In your report, do not just describe what you do, but also why you do it. The report should be understandable to some fellow student, who has some basic knowledge of machine learning, but isn't as much an expert as you are. Do not assume that all concepts are known to the reader, but describe them briefly, when you introduce them.


### Describe what machine learning methods you plan to use.

I will use the Naive Bayes algorithm. It works by... In this sense, it is a supervised learning algorithm. It is appropriate for this task because ... 



```python
# Import some necessary modules
import nltk, random
from nltk.corpus import senseval
```

### You need to describe what data you plan to use and how it will be partitioned into training, development/validation and test sets.

I am using the Senseval corpus. After randomization, I will split the data into training and testing sets. This is done by ... 

Validation/developement ? 

As for extracting features, I am planning on using a) context words (as in, words that appear around the focus word) and b) the 'senses' category, which represents the exact meaning of the focus word.



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
    



```python
def get_category(pos):
    
    category = []
    for inst in senseval.instances(pos):
        category.append(inst.senses)
    return category
```


```python
def get_features(inst):
    
    features = {}
    p = inst.position
    inst.context.append(('<END>','<END>'))

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
```


```python
interest_featureset = [(get_features(inst), c) for c,inst in zip(get_category('interest.pos'), senseval.instances('interest.pos'))]
hard_featureset = [(get_features(inst), c) for c,inst in zip(get_category('hard.pos'), senseval.instances('hard.pos'))]
line_featureset = [(get_features(inst), c) for c,inst in zip(get_category('line.pos'), senseval.instances('line.pos'))]
serve_featureset = [(get_features(inst), c) for c,inst in zip(get_category('serve.pos'), senseval.instances('serve.pos'))] 
print()
print('Example of featureset for the word "hard":\n\n', hard_featureset[30])
```

    
    Example of featureset for the word "hard":
    
     ({'1 Previous tag': 'DT', '1 Next tag': 'NN', '2 Previous tags': 'VBP DT', '2 Next tags': 'NN IN', '1 Previous word': '', '1 Next word': 'time', '2 Previous words': 'have', '2 Next words': 'time with'}, ('HARD1',))



```python
size = int(len(senseval.instances('hard.pos')) * 0.25)
random.shuffle(hard_featureset)
hard_train_set, hard_test_set = hard_featureset[size:], hard_featureset[:size]

size = int(len(senseval.instances('interest.pos')) * 0.25)
random.shuffle(interest_featureset)
interest_train_set, interest_test_set = interest_featureset[size:], interest_featureset[:size]

size = int(len(senseval.instances('serve.pos')) * 0.25)
random.shuffle(serve_featureset)
serve_train_set, serve_test_set = serve_featureset[size:], serve_featureset[:size]

size = int(len(senseval.instances('line.pos')) * 0.25)
random.shuffle(line_featureset)
line_train_set, line_test_set = line_featureset[size:], line_featureset[:size]
```


```python
def word_bayes(train_set, test_set, word):
    bayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(word, "Naive Bayes accuracy percent:", (nltk.classify.accuracy(bayes_classifier, test_set))*100,"%")
    print()
    print(bayes_classifier.show_most_informative_features(20))
    print()
    return bayes_classifier
```


```python
word_bayes(hard_train_set, hard_test_set, "Hard -")
word_bayes(interest_train_set, interest_train_set, "Interest -")
word_bayes(serve_train_set, serve_test_set, "Serve -")
word_bayes(line_train_set, line_test_set, "Line -")
```

    Hard - Naive Bayes accuracy percent: 86.0572483841182 %
    
    Most Informative Features
                 2 Next tags = 'TO VB'         HARD1 : HARD2  =    159.9 : 1.0
                 1 Next word = 'to'            HARD1 : HARD2  =    113.4 : 1.0
                  1 Next tag = 'TO'            HARD1 : HARD2  =     84.4 : 1.0
            2 Previous words = "it 's"         HARD1 : HARD2  =     70.2 : 1.0
                 1 Next word = 'work'          HARD2 : HARD3  =     62.5 : 1.0
                2 Next words = 'work'          HARD2 : HARD1  =     59.0 : 1.0
                 2 Next tags = 'NN CC'         HARD2 : HARD1  =     57.9 : 1.0
                2 Next words = 'work and'      HARD2 : HARD1  =     56.1 : 1.0
             2 Previous tags = 'NNS IN'        HARD2 : HARD1  =     54.3 : 1.0
                 1 Next word = 'line'          HARD2 : HARD1  =     43.1 : 1.0
                 2 Next tags = 'JJ'            HARD3 : HARD1  =     41.4 : 1.0
             1 Previous word = "'s"            HARD1 : HARD3  =     39.3 : 1.0
                  1 Next tag = 'VBN'           HARD3 : HARD1  =     39.1 : 1.0
                 1 Next word = 'place'         HARD3 : HARD1  =     34.4 : 1.0
             1 Previous word = 'no'            HARD2 : HARD1  =     30.7 : 1.0
                 1 Next word = 'for'           HARD1 : HARD2  =     28.1 : 1.0
                 2 Next tags = "NN ''"         HARD2 : HARD1  =     26.2 : 1.0
                 2 Next tags = 'VBN'           HARD3 : HARD1  =     26.2 : 1.0
                 2 Next tags = 'NNS CC'        HARD3 : HARD1  =     26.2 : 1.0
              1 Previous tag = 'SYM'           HARD3 : HARD1  =     25.4 : 1.0
    None
    
    Interest - Naive Bayes accuracy percent: 93.80630630630631 %
    
    Most Informative Features
                  1 Next tag = 'NNS'          intere : intere =    114.6 : 1.0
                 2 Next tags = 'IN VBG'       intere : intere =     67.0 : 1.0
                 1 Next word = 'in'           intere : intere =     53.1 : 1.0
                 1 Next word = 'of'           intere : intere =     47.9 : 1.0
             1 Previous word = 'other'        intere : intere =     46.7 : 1.0
                 2 Next tags = 'NNS'          intere : intere =     39.7 : 1.0
                  1 Next tag = 'NN'           intere : intere =     37.3 : 1.0
                  1 Next tag = 'TO'           intere : intere =     30.6 : 1.0
                 2 Next tags = 'TO VB'        intere : intere =     27.7 : 1.0
             1 Previous word = 'with'         intere : intere =     22.0 : 1.0
                  1 Next tag = 'MD'           intere : intere =     20.0 : 1.0
             1 Previous word = 'in'           intere : intere =     18.9 : 1.0
             2 Previous tags = 'VB JJ'        intere : intere =     18.7 : 1.0
                 2 Next tags = 'IN JJ'        intere : intere =     17.9 : 1.0
                 2 Next tags = 'IN DT'        intere : intere =     17.7 : 1.0
              1 Previous tag = ''             intere : intere =     17.3 : 1.0
                 2 Next tags = 'NNS VBP'      intere : intere =     17.2 : 1.0
             2 Previous tags = 'PRP$ NN'      intere : intere =     16.8 : 1.0
                 2 Next tags = 'IN NN'        intere : intere =     16.4 : 1.0
                 2 Next tags = 'NNS IN'       intere : intere =     16.1 : 1.0
    None
    
    Serve - Naive Bayes accuracy percent: 77.23948811700183 %
    
    Most Informative Features
                 1 Next word = 'as'           SERVE2 : SERVE6 =    181.0 : 1.0
              1 Previous tag = 'WDT'          SERVE6 : SERVE1 =     96.7 : 1.0
                2 Next words = 'as'           SERVE2 : SERVE1 =     64.1 : 1.0
             2 Previous tags = 'WDT'          SERVE6 : SERVE1 =     60.1 : 1.0
             1 Previous word = 'would'        SERVE2 : SERVE1 =     43.0 : 1.0
                 1 Next word = 'in'           SERVE1 : SERVE6 =     41.7 : 1.0
                  1 Next tag = 'NNP'          SERVE6 : SERVE1 =     41.2 : 1.0
              1 Previous tag = 'WP'           SERVE1 : SERVE2 =     40.3 : 1.0
             1 Previous word = 'it'           SERVE2 : SERVE1 =     37.3 : 1.0
             1 Previous word = 'that'         SERVE2 : SERVE1 =     35.6 : 1.0
                 2 Next tags = 'CD'           SERVE1 : SERVE1 =     31.8 : 1.0
            2 Previous words = 'which'        SERVE6 : SERVE1 =     31.3 : 1.0
             1 Previous word = 'before'       SERVE1 : SERVE1 =     30.5 : 1.0
             2 Previous tags = 'NNS WDT'      SERVE6 : SERVE1 =     27.3 : 1.0
             2 Previous tags = 'WDT MD'       SERVE2 : SERVE1 =     27.0 : 1.0
                2 Next words = 'to'           SERVE1 : SERVE6 =     26.9 : 1.0
             2 Previous tags = 'NN WDT'       SERVE2 : SERVE1 =     26.8 : 1.0
                2 Next words = 'on the'       SERVE1 : SERVE1 =     26.4 : 1.0
                 1 Next word = 'no'           SERVE2 : SERVE1 =     24.9 : 1.0
                2 Next words = ''             SERVE1 : SERVE1 =     24.7 : 1.0
    None
    
    Line - Naive Bayes accuracy percent: 69.01544401544402 %
    
    Most Informative Features
                 1 Next word = 'between'      divisi : produc =    146.4 : 1.0
             1 Previous word = 'in'           format : produc =     66.8 : 1.0
             1 Previous word = 'telephone'     phone : text   =     47.4 : 1.0
                 1 Next word = 'of'           produc : phone  =     45.1 : 1.0
             2 Previous tags = 'VB IN'        format : produc =     44.0 : 1.0
              1 Previous tag = 'IN'           format : divisi =     42.7 : 1.0
             1 Previous word = 'new'          produc : text   =     33.1 : 1.0
             1 Previous word = 'fine'         divisi : text   =     31.3 : 1.0
              1 Previous tag = 'CC'             cord : produc =     29.7 : 1.0
              1 Previous tag = 'PRP$'           cord : phone  =     28.4 : 1.0
            2 Previous words = 'on the'        phone : divisi =     25.2 : 1.0
            2 Previous words = 'fine'         divisi : text   =     21.7 : 1.0
                 2 Next tags = 'VBD RB'         cord : produc =     20.5 : 1.0
             2 Previous tags = 'VBP DT'       divisi : produc =     20.1 : 1.0
                 2 Next tags = 'IN JJ'        produc : phone  =     19.0 : 1.0
                 2 Next tags = 'IN NNS'       format : phone  =     18.9 : 1.0
                 1 Next word = 'at'           format : produc =     18.4 : 1.0
             1 Previous word = 'their'          text : produc =     18.2 : 1.0
             2 Previous tags = 'WRB DT'       divisi : produc =     17.6 : 1.0
             2 Previous tags = 'IN PRP$'        cord : phone  =     16.9 : 1.0
    None
    





    <nltk.classify.naivebayes.NaiveBayesClassifier at 0x7f578eb6f0b8>



### You need to describe how you plan to carry out the evaluation.

The evaluation is crucial and you need to provide some quantitative evaluation. Therefore, you will need annotated test data that is used as a gold standard.

The evaluation should provide some in-depth analysis of the phenomenon, such as: 
What features are most important? 
What category is recognized best? 
What category is the hardest to get right? 
    The word "hard" might be the hardest to get right, as its baseline is already fairly high, 79,9%.
Are there multiple different ways of evaluating the performance?

Are there signs or risks of overfitting? What can be done to prevent that from happening?

If applicable in your case, report what performance can be achieved using the majority baseline approach, that is, a naïve approach that just assigns every item to the most frequent class. Does your system produce significantly better than the naïve baseline?


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

We can use the method nltk.FreqDist() to compute the distribution of the different senses of the words hard, interest, line and serve in the Senseval corpus. By choosing the most common sense and comparing it to the others, we can calculate the following baselines for the following words:

- "hard": 79.7%
- "serve": 41.4%
- "interest": 52.9%
- "line": 53.5%


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
    please serve us dinner
    Hmm...
    I think by "serve" you mean provide (usually but not necessarily food)

