import json
with open("chatdata.json", 'r') as f:
    chat_data = json.load(f)
    f.close()

with open("response.json") as f:
    response_dict = json.load(f)
    f.close()


import numpy as np

training_dict = {}

for intent, question_list in chat_data.items():
    
   for question in question_list:
     training_dict[question] = intent
 

feature =np.array(list(training_dict.keys()))
labels = np.array(list(training_dict.values()))
feature, labels

from sklearn.feature_extraction.text import TfidfVectorizer

tf_vec = TfidfVectorizer().fit(feature)
X = tf_vec.transform(feature).toarray()
y = labels.reshape(-1,1)

from sklearn.ensemble import RandomForestClassifier
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