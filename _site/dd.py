
# coding: utf-8

# ### English word sense disambiguation
# 
# #### The task
# Create a machine learning system that can disambiguate the correct sense of a word in context. Disambiguate the four words hard, interest, line, and serve into the senses given in the Senseval 2 corpus (NLTK Corpus HOWTO: Senseval). You can augment your data with other corpora as well. You can perform either supervised or unsupervised machine learning.
# 
# 
# The overall structure of the report must be given. At least there must be section headings and brief descriptions of what belongs to each section.
# 
# #### Why this task? 
# 
# What task are you trying to solve and why is this interesting?
# 
#     I am trying to 
# 
# In your report, do not just describe what you do, but also why you do it. The report should be understandable to some fellow student, who has some basic knowledge of machine learning, but isn't as much an expert as you are. Do not assume that all concepts are known to the reader, but describe them briefly, when you introduce them.
# 

# ### Describe what machine learning methods you plan to use.
# 
# I will use the Naive Bayes algorithm. It works by... In this sense, it is a supervised learning algorithm. It is appropriate for this task because ... 
# 

# In[278]:


# Import some necessary modules
import nltk, random
from nltk.corpus import senseval


# ### You need to describe what data you plan to use and how it will be partitioned into training, development/validation and test sets.
# 
# I am using the Senseval corpus. After randomization, I will split the data into training and testing sets. This is done by ... 
# 
# Validation/developement ? 
# 
# As for extracting features, I am planning on using a) context words (as in, words that appear around the focus word) and b) the 'senses' category, which represents the exact meaning of the focus word.
# 

# In[279]:


print("All fileids:", senseval.fileids())
print()
for fileid in senseval.fileids():
    print(senseval.instances(fileid)[0])
    print()


# In[280]:


def get_category(pos):
    
    category = []
    for inst in senseval.instances(pos):
        category.append(inst.senses)
    return category


# In[303]:


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


# In[304]:


interest_featureset = [(get_features(inst), c) for c,inst in zip(get_category('interest.pos'), senseval.instances('interest.pos'))]
hard_featureset = [(get_features(inst), c) for c,inst in zip(get_category('hard.pos'), senseval.instances('hard.pos'))]
line_featureset = [(get_features(inst), c) for c,inst in zip(get_category('line.pos'), senseval.instances('line.pos'))]
serve_featureset = [(get_features(inst), c) for c,inst in zip(get_category('serve.pos'), senseval.instances('serve.pos'))] 
print()
print('Example of featureset for the word "hard":\n\n', hard_featureset[30])


# In[308]:


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


# In[284]:


def word_bayes(train_set, test_set, word):
    bayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(word, "Naive Bayes accuracy percent:", (nltk.classify.accuracy(bayes_classifier, test_set))*100,"%")
    print()
    print(bayes_classifier.show_most_informative_features(20))
    print()
    return bayes_classifier


# In[309]:


word_bayes(hard_train_set, hard_test_set, "Hard -")
word_bayes(interest_train_set, interest_train_set, "Interest -")
word_bayes(serve_train_set, serve_test_set, "Serve -")
word_bayes(line_train_set, line_test_set, "Line -")


# ### You need to describe how you plan to carry out the evaluation.
# 
# The evaluation is crucial and you need to provide some quantitative evaluation. Therefore, you will need annotated test data that is used as a gold standard.
# 
# The evaluation should provide some in-depth analysis of the phenomenon, such as: 
# What features are most important? 
# What category is recognized best? 
# What category is the hardest to get right? 
#     The word "hard" might be the hardest to get right, as its baseline is already fairly high, 79,9%.
# Are there multiple different ways of evaluating the performance?
# 
# Are there signs or risks of overfitting? What can be done to prevent that from happening?
# 
# If applicable in your case, report what performance can be achieved using the majority baseline approach, that is, a naïve approach that just assigns every item to the most frequent class. Does your system produce significantly better than the naïve baseline?

# In[142]:


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


# We can use the method nltk.FreqDist() to compute the distribution of the different senses of the words hard, interest, line and serve in the Senseval corpus. By choosing the most common sense and comparing it to the others, we can calculate the following baselines for the following words:
# 
# - "hard": 79.7%
# - "serve": 41.4%
# - "interest": 52.9%
# - "line": 53.5%

# In[324]:


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

