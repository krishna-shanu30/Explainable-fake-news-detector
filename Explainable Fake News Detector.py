#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np


# In[83]:


true_df= pd.read_csv(r"D:\fake-news-detector\True.csv", low_memory= False)
fake_df=pd.read_csv(r"D:\fake-news-detector\Fake.csv", low_memory= False)


# In[32]:


true_df


# In[33]:


true_df.drop("date", axis=1,inplace= True)
fake_df.drop("date",axis=1,inplace= True)


# In[34]:


true_df


# In[35]:


true_df['labels']=1
fake_df['labels']=0


# In[36]:


# combine the dataset
df= pd.concat([true_df,fake_df],axis=0)
# we have done a random shuffle so that model doesnot learn pattern
df= df.sample(frac=1).reset_index(drop= True)


# In[37]:


df.head()


# In[38]:


df = df[["title", "text", "labels"]]
# keep the imp cols


# In[39]:


df.isnull().sum()


# In[40]:


# now  we will merge title and text
df['content']= df['title']+ " " + df['text']


# In[41]:


df.head()


# In[42]:


df.drop(['title','text'], axis=1, inplace= True)
# we have done that because we have already merged title and text in content


# In[43]:


df.head()


# In[44]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")


# In[45]:


lemmatizer = WordNetLemmatizer()
stop_words= set(stopwords.words("english"))
def clean_text(text):
    text= text.lower()
    text= re.sub(r"http\S+","",text)
    text= re.sub(r"[^a-zA-Z\s]","",text)
    words= text.split()
    words= [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    
    return " ".join(words)


# In[46]:


# apply cleaning
df['clean_content']= df['content'].apply(clean_text)


# In[47]:


import textstat
def read_score(text):
    return textstat.flesch_reading_ease(text)
def exclam_count(text):
    return text.count("!")
def capital_ratio(text):
    if len(text) == 0:
        return 0
    capitals = sum(1 for c in text if c.isupper())
    return capitals / len(text)

def avg_word_length(text):

    words = text.split()

    if len(words) == 0:
        return 0

    return np.mean([len(word) for word in words])
df["readability"] = df["content"].apply(read_score)

df["exclamations"] = df["content"].apply(exclam_count)

df["capital_ratio"] = df["content"].apply(capital_ratio)

df["avg_word_length"] = df["content"].apply(avg_word_length)


# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer

vector=TfidfVectorizer(max_features=5000,ngram_range=(1,2))
x_tdf= vector.fit_transform(df["clean_content"])


# In[49]:


ling_features = df[
    ["readability",
     "exclamations",
     "capital_ratio",
     "avg_word_length"]].values


# In[50]:


from scipy.sparse import hstack
x= hstack([x_tdf, ling_features])
y= df["labels"]


# In[51]:


from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test =tts(x,y,test_size=0.2, random_state=42)


# In[52]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(x_train,y_train)


# In[53]:


from sklearn.metrics import accuracy_score, classification_report
y_pred= model.predict(x_test)
accuracy= accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print(classification_report(y_test,y_pred))


# In[54]:


import joblib
joblib.dump(model,"fake_news_model.pkl")


# In[55]:


joblib.dump(vector, "tfidf_vector.pkl")


# In[69]:


def predict_news(text):

    cleaned = clean_text(text)

    vect = vector.transform([cleaned])

    readability = read_score(text)

    exclamations = exclam_count(text)

    capitals = capital_ratio(text)

    avg_length = avg_word_length(text)

    linguistic_features = np.array([
        [readability, exclamations, capitals, avg_length]
    ])

    final_features = hstack([vect, linguistic_features])

    prediction = model.predict(final_features)[0]

    probability = model.predict_proba(final_features).max()

    if prediction == 0:
        label = "FAKE"
    else:
        label = "TRUE"

    return label, probability


# In[70]:


sample_news = """
Breaking news! Scientists confirm aliens landed in New York.
"""

label, confidence = predict_news(sample_news)

print("Prediction:", label)

print("Confidence:", confidence)


# In[73]:


get_ipython().system(' pip install lime ')


# In[74]:


import sys
get_ipython().system('{sys.executable} -m pip install lime')


# In[78]:


# we are using XAI
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names = ["FAKE","real"])

# probability_function
def predict_proba(texts):

    cleaned = [clean_text(text) for text in texts]

    vect = vector.transform(cleaned)

    dummy_features = np.zeros((len(texts), 4))

    final_features = hstack([vect, dummy_features])

    return model.predict_proba(final_features)


# In[79]:


exp = explainer.explain_instance(sample_news, predict_proba, num_features= 10)


# In[81]:


# shows which words caused fake prediction
#exp.show_in_notebook()
# since i am using python 3.12, there would be issue so:
from IPython.display import display, HTML
html = exp.as_html()
display(HTML(html))

