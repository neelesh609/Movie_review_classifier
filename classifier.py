
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import csv
import nltk
import pickle
import operator
import tkinter as tk
from tkinter import *



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("D:\\project data mining\\train.csv")
test_data = pd.read_csv("D:\\project data mining\\test.csv")

movie_data = [train_data,test_data]
final_data = pd.concat(movie_data)

split_1 = int(0.8 * len(final_data))
split_2 = int(0.9 * len(final_data))
train_data = final_data[:split_1]
dev_data = final_data[split_1:split_2]
test_data = final_data[split_2:]

sentiment_int = []
for index, row  in train_data.iterrows():
    if row['sentiment'] == 'pos':
        sentiment_int.append(1)
    else:
        sentiment_int.append(0)

train_data["sentiment_int"] = sentiment_int


sentiment_int = []
for index, row  in test_data.iterrows():
    if row['sentiment'] == 'pos':
        sentiment_int.append(1)
    else:
        sentiment_int.append(0)

test_data["sentiment_int"] = sentiment_int


sentiment_int = []
for index, row  in dev_data.iterrows():
    if row['sentiment'] == 'pos':
        sentiment_int.append(1)
    else:
        sentiment_int.append(0)

dev_data["sentiment_int"] = sentiment_int

small_df = train_data.groupby('sentiment_int').apply(lambda x: x.sample(frac=0.8))
original_len_small_df = len(small_df)
print(original_len_small_df)

tfidfconverter = TfidfVectorizer(min_df=0.002)

# For train data - use fit_transform
X_train = tfidfconverter.fit_transform(train_data['text']).toarray()

# For dev and test - use transform
X_dev_arr = tfidfconverter.transform(dev_data['text']).toarray()
X_test_arr = tfidfconverter.transform(test_data['text']).toarray()
X_dev = tfidfconverter.transform(dev_data['text'])
X_test = tfidfconverter.transform(test_data['text'])

# Put 'rating' column of each dataframe into y
y_train = np.asarray(train_data['sentiment_int'])
y_dev = np.asarray(dev_data['sentiment_int'])
y_test = np.asarray(test_data['sentiment_int'])

mse_dict = dict()
accuracy_dict = dict()
classifier_dict = dict()

start_time = time.time()

# Train and Predict the data using Multinomial Naive Bayes
multinomialNB = MultinomialNB(alpha=1)
multinomialNB.fit(X_train, y_train)
classifier_dict["Multinomial Naive Bayes"] = multinomialNB;
y_pred_mnb_dev = multinomialNB.predict(X_dev)

# Calculate the Mean Squared Error and Accuracy
mse_mnb_dev = mean_squared_error(y_test, y_pred_mnb_dev)
accuracy_mnb_dev = accuracy_score(y_test, y_pred_mnb_dev)*100

# Print the Mean Squared Error and Accuracy
print("Using Multinomial Naive Bayes:")
print("Mean Squared Error:", mse_mnb_dev)
print("Accuracy:", accuracy_mnb_dev)

# Store the Mean Squared Error and Accuracy in dictionaries
mse_dict["Multinomial Naive Bayes"] = mse_mnb_dev;
accuracy_dict["Multinomial Naive Bayes"] = accuracy_mnb_dev;

end_time = time.time()
print("runtime: %s sec" % (end_time - start_time))

start_time = time.time()

# Train and Predict the data using Linear SVM (C=1)
linearSVC1 = LinearSVC(C=1, dual=False)
linearSVC1.fit(X_train, y_train)
classifier_dict["Linear SVC (C=1)"] = linearSVC1;
y_pred_lsvc = linearSVC1.predict(X_dev)

# Calculate the Mean Squared Error and Accuracy
mse_lsvc1_dev = mean_squared_error(y_test, y_pred_lsvc)
accuracy_lsvc1_dev = accuracy_score(y_test, y_pred_lsvc)*100

# Print the Mean Squared Error and Accuracy
print("Using Linear SVC (C=1):")
print('Mean Squared Error:', mse_lsvc1_dev)
print('Accuracy:', accuracy_lsvc1_dev)

# Store the Mean Squared Error and Accuracy in dictionaries
mse_dict["Linear SVC (C=1)"] = mse_lsvc1_dev;
accuracy_dict["Linear SVC (C=1)"] = accuracy_lsvc1_dev;

end_time = time.time()
print("runtime: %s sec" % (end_time - start_time))

start_time = time.time()

# Train and Predict the data using Linear SVM (C=100)
linearSVC100 = LinearSVC(C=100, dual=False)
linearSVC100.fit(X_train, y_train)
classifier_dict["Linear SVC (C=100)"] = linearSVC100;
y_pred_lsvc = linearSVC100.predict(X_dev)

# Calculate the Mean Squared Error and Accuracy
mse_lsvc100_dev = mean_squared_error(y_test, y_pred_lsvc)
accuracy_lsvc100_dev = accuracy_score(y_test, y_pred_lsvc)*100

# Print the Mean Squared Error and Accuracy
print("Using Linear SVC (C=100):")
print('Mean Squared Error:', mse_lsvc100_dev)
print('Accuracy:', accuracy_lsvc100_dev)

# Store the Mean Squared Error and Accuracy in dictionaries
mse_dict["Linear SVC (C=100)"] = mse_lsvc100_dev;
accuracy_dict["Linear SVC (C=100)"] = accuracy_lsvc100_dev;

end_time = time.time()
print("runtime: %s sec" % (end_time - start_time))

start_time = time.time()

# Train and Predict the data using Linear SVM (C=1000)
linearSVC1000 = LinearSVC(C=1000, dual=False)
linearSVC1000.fit(X_train, y_train)
classifier_dict["Linear SVC (C=1000)"] = linearSVC1000;
y_pred_lsvc = linearSVC1000.predict(X_dev)

# Calculate the Mean Squared Error and Accuracy
mse_lsvc1000_dev = mean_squared_error(y_test, y_pred_lsvc)
accuracy_lsvc1000_dev = accuracy_score(y_test, y_pred_lsvc)*100

# Print the Mean Squared Error and Accuracy
print("Using Linear SVC (C=1000):")
print('Mean Squared Error:', mse_lsvc1000_dev)
print('Accuracy:', accuracy_lsvc1000_dev)

# Store the Mean Squared Error and Accuracy in dictionaries
mse_dict["Linear SVC (C=1000)"] = mse_lsvc1000_dev;
accuracy_dict["Linear SVC (C=1000)"] = accuracy_lsvc1000_dev;

end_time = time.time()
print("runtime: %s sec" % (end_time - start_time))

start_time = time.time()

# Train and Predict the data using Random Forest Classifier (n_estimators=10)
randomForest10 = RandomForestClassifier(max_depth=100, n_estimators=10, max_features=1)
randomForest10.fit(X_train, y_train)
classifier_dict["Random Forest Classifier (n_estimators=10)"] = randomForest10;
y_pred_rfc = randomForest10.predict(X_dev)

# Calculate the Accuracy
mse_rfc10_dev = mean_squared_error(y_test, y_pred_rfc)
accuracy_rfc10_dev = accuracy_score(y_test, y_pred_rfc)*100

# Print the  and Accuracy
print("Using Random Forest Classifier:")

print('Accuracy:', accuracy_rfc10_dev)

# Store the Mean Squared Error and Accuracy in dictionaries
mse_dict["Random Forest Classifier (n_estimators=10)"] = mse_rfc10_dev;
accuracy_dict["Random Forest Classifier (n_estimators=10)"] = accuracy_rfc10_dev;

end_time = time.time()
print("runtime: %s sec" % (end_time - start_time))

mse_dict_list = sorted(mse_dict.items(), key=operator.itemgetter(1), reverse=False)
accuracy_dict_list = sorted(accuracy_dict.items(), key=operator.itemgetter(1), reverse=True)
accuracy_dict_list

graph_accuracy_list = [item[1] for item in accuracy_dict_list]
graph_classifier_list = [item[0] for item in mse_dict_list]
graph_mse_list = [item[1] for item in mse_dict_list]

minY = 0;
maxY = max(graph_accuracy_list)

df = pd.DataFrame({'Accuracy': graph_accuracy_list}, index=graph_classifier_list)
ax = df.plot(figsize=(7,5), kind='bar', stacked=True)

ax. set(xlabel="Classifiers used", ylabel="Accuracy")

ax.set(ylim=[minY, maxY+2])

highest_accuracy_classifier = accuracy_dict_list[0]
print("Best Classifier considering highest accuracy:", highest_accuracy_classifier)

best_classifier_name = accuracy_dict_list[0][0]
bestClassifier = classifier_dict.get(best_classifier_name)
print(bestClassifier)

y_pred_test = bestClassifier.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)*100
print("Using Best Classifier:\n")
print('Accuracy:', accuracy_test)

def rate():
    input_review = t.get()
    print(input_review)
    mcom = {'text': [input_review]}
    mdf = pd.DataFrame(mcom, columns = ['text'])
    X_single = tfidfconverter.transform(mdf['text'])
    y_single = bestClassifier.predict(X_single)
    print("review: ", y_single[0])
    if y_single == 1:
        print('positive review')
        output.create_text(50,10, text="positive review")
    else:
        print('Negative')
        output.create_text(50,10, text="negative review")



root = tk.Tk()
root.geometry("500x250")
root.title("GUI")
root.configure(bg='black')

label2 = tk.Label(root, text= "Enter a review about the last movie you watched:", font= ('Helvetica 12 '))
label2.place(x=80 , y=70)

t = Entry(root)
t.place(x=85, y=120)

output = tk.Canvas(width="200", height=150)
output.place(x=50, y=200)

b = tk.Button(root, text="Search", command=rate)
b.place(x=220, y=160)

Label(root, text="Result : ", font=('Helvetica 10 ')).place(x=80, y=185)

display_canvas2 = tk.Canvas(root, bg="white", width=100, height=20)
display_canvas2.place(x=80, y=220)



root.mainloop()
