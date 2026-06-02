#!/usr/bin/env python
# coding: utf-8

# In[145]:


import pandas as pd
import numpy as np


# In[146]:


true_df= pd.read_csv(r"D:\True.csv", low_memory= False)
fake_df=pd.read_csv(r"D:\Fake.csv", low_memory= False)


# In[147]:


true_df


# In[148]:


true_df.drop("date", axis=1,inplace= True)
fake_df.drop("date",axis=1,inplace= True)


# In[149]:


true_df


# In[150]:


true_df['labels']=1
fake_df['labels']=0


# In[151]:


# combine the dataset
df= pd.concat([true_df,fake_df],axis=0)
# we have done a random shuffle so that model doesnot learn pattern
df= df.sample(frac=1).reset_index(drop= True)


# In[152]:


df.head()


# In[153]:


df = df[["title", "text", "labels"]]
# keep the imp cols


# In[154]:


df.isnull().sum()


# In[155]:


# now  we will merge title and text
df['content']= df['title']+ " " + df['text']


# In[156]:


df.head()


# In[157]:


df.drop(['title','text'], axis=1, inplace= True)
# we have done that because we have already merged title and text in content


# In[158]:


df.head()


# In[159]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")


# In[160]:


lemmatizer = WordNetLemmatizer()
stop_words= set(stopwords.words("english"))
def clean_text(text):
    text= text.lower()
    text= re.sub(r"http\S+","",text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words= text.split()
    words= [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    
    return " ".join(words)


# In[161]:


# apply cleaning
df['clean_content']= df['content'].apply(clean_text)


# In[162]:


from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.85
)
x_tdf= vector.fit_transform(df["clean_content"])


# ling_features = df[
#     ["readability",
#      "exclamations",
#      "capital_ratio",
#      "avg_word_length"]].values

# In[ ]:





# In[163]:


from scipy.sparse import hstack
x = x_tdf
y = df["labels"]


# In[164]:


from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test =tts(x,y,test_size=0.2, random_state=42)


# In[165]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42,class_weight="balanced")
model.fit(x_train,y_train)


# In[166]:


from sklearn.metrics import accuracy_score, classification_report
y_pred= model.predict(x_test)
accuracy= accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print(classification_report(y_test,y_pred))


# In[167]:


import joblib
joblib.dump(model,"fake_news_model.pkl")


# In[168]:


joblib.dump(vector, "tfidf_vector.pkl")


# def predict_proba(texts):
# 
#     cleaned = [clean_text(text) for text in texts]
# 
#     vect = vector.transform(cleaned)
# 
#     linguistic_data = []
# 
#     for text in texts:
# 
#         readability = read_score(text)
# 
#         exclamations = exclam_count(text)
# 
#         capitals = capital_ratio(text)
# 
#         avg_length = avg_word_length(text)
# 
#         linguistic_data.append([
#             readability,
#             exclamations,
#             capitals,
#             avg_length
#         ])
# 
#     linguistic_features = np.array(linguistic_data)
# 
#     final_features = hstack([
#         vect,
#         linguistic_features
#     ])
# 
#     return model.predict_proba(final_features)

# In[169]:


def predict_news(text):

    cleaned = clean_text(text)

    vect = vector.transform([cleaned])

    prediction = model.predict(vect)[0]

    probability = model.predict_proba(vect).max()

    if prediction == 0:
        label = "FAKE"
    else:
        label = "REAL"

    return label, probability
'''def predict_proba(texts):

    cleaned = [clean_text(text) for text in texts]

    vect = vector.transform(cleaned)

    return model.predict_proba(vect)'''


# In[180]:



sample_news = """CBSE opens verification and re-evaluation portal
CBSE said the portal was live during the early hours on Tuesday (June 2, 2026), missing its original deadline of June 1, 2026; several users on X complained that they were unable to login
Updated - June 02, 2026 11:37 am IST - NEW DELHI

Missing its original deadline of Monday (June 1, 2026), the CBSE’s re-evaluation portal, was opened during the early morning hours on Tuesday (June 2, 2026).

In a post on X, CBSE said that the portal was live. It also shared a video of the step-by-step procedure on how to apply for verification and re-evaluation. The portal can be accessed here.


However, several users replied to CBSE’s post on X, saying that they were unable to login and pointed to several other problems with logging in to the portal.

On Monday (June 1, 2026) afternoon, the CBSE had stated that the “verification and re-evaluation portal will go live soon” but remained inactive through the day.

 'Beware of pickpockets': Rahul slams Govt. over CBSE re-evaluation cost

The delays follow severe criticism over the board’s expanded On-Screen Marking (OSM) system used for evaluating the Class 12 examinations this year.

The application portal was originally slated to open on May 29. The CBSE later postponed the launch to June 1 to build a “transparent and glitch-free process”.

"""

label, confidence = predict_news(sample_news)
print("Prediction:", label)
print("Confidence:", confidence)


# In[181]:


from sklearn.metrics import confusion_matrix


print(confusion_matrix(y_test, y_pred))


# # we are using XAI
# from lime.lime_text import LimeTextExplainer
# explainer = LimeTextExplainer(class_names = ["FAKE","REAL"])
# 
# # probability_function
# def predict_proba(texts):
# 
#     cleaned = [clean_text(text) for text in texts]
# 
#     vect = vector.transform(cleaned)
# 
#     dummy_features = np.zeros((len(texts), 4))
# 
#     final_features = hstack([vect, dummy_features])
# 
#     return model.predict_proba(final_features)

# In[182]:


# we are using XAI
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(
    class_names=["FAKE", "REAL"]
)

# probability function
def predict_proba(texts):

    cleaned = [clean_text(text) for text in texts]

    vect = vector.transform(cleaned)

    return model.predict_proba(vect)


# In[183]:


exp = explainer.explain_instance(sample_news, predict_proba, num_features= 10)


# In[184]:


# shows which words caused fake prediction
#exp.show_in_notebook()
# since i am using python 3.12, there would be issue so:
from IPython.display import display, HTML
html = exp.as_html()
display(HTML(html))

