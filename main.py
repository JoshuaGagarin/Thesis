# JSON Encode-Decode library
import json
# Reading chat data
with open("chatdata.json", 'r') as f:
    chat_data = json.load(f)
    f.close()
# Reading Resposne 
with open("response.json") as f:
    response_dict = json.load(f)
    f.close()


# Scientific computational library
import numpy as np
# Training Data
training_dict = {}
# creating formatted data for fitiing model
for intent, question_list in chat_data.items():
    
   for question in question_list:
     training_dict[question] = intent
 

# Separating Features i.e questions and Labels i.e intents
feature =np.array(list(training_dict.keys()))
labels = np.array(list(training_dict.values()))
feature, labels

# WordVecotr with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# Converting text to WordVector
tf_vec = TfidfVectorizer().fit(feature)
X = tf_vec.transform(feature).toarray()
# Reshaping labels to fit data
y = labels.reshape(-1,1)

# Classifier
from sklearn.ensemble import RandomForestClassifier
# Fitting model
rnn = RandomForestClassifier(n_estimators=200)
rnn.fit(X, y)

# Creating response

def botanswer(q):
    process_text = tf_vec.transform([q]).toarray()
    prob = rnn.predict_proba(process_text)[0]
    max_ = np.argmax(prob)

    if prob[max_] < 0.6: #Only 60% and above accurate
        return "Sorry I am not getting you...!"
    else:
        return response_dict[rnn.classes_[max_]]
# # Chat with bot
while True:
    user = input("User>> ")
    if user == "quit":
        break
    print("Bot>> {}".format(botanswer(user)))