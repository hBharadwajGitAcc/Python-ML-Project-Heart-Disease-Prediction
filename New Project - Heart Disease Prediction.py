import pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


import os
print(os.getcwd())
os.chdir("C:\\Users\\user\\Downloads\\assignments\\python_projects")
os.getcwd()




dataset = pd.read_csv('dataset.csv')


dataset.head()


dataset.describe()


corr = dataset.corr()
corr




sns.countplot(dataset.target, palette=['green', 'red'])
plt.title('[0]: Patients those who do not have a heart disease [1]: Patients those who do have a heart disease')


plt.figure(figsize=(18, 10))
sns.countplot(x="age", hue="target", data=dataset, palette=['green', 'red'])
plt.legend(["Does not have heart disease", "Have heart disease"])
plt.title("Heart Diesease for ages of patients")
plt.xlabel("age")
plt.ylabel("frequency")
plt.plot()


plt.figure(figsize=(18, 10))
sns.heatmap(corr, annot=True)
plt.plot()




# Logistic Regression Model
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)


LogisticRegression_Model = LogisticRegression()
LogisticRegression_Model.fit(x_train, y_train)


LogisticRegression_Predictions = LogisticRegression_Model.predict(x_test)
LogisticRegression_Predictions


LogisticRegression_confusion_matrix = confusion_matrix(y_test, LogisticRegression_Predictions)
LogisticRegression_confusion_matrix


LogisticRegression_AccuracyScore = accuracy_score(y_test, LogisticRegression_Predictions)
LogisticRegression_AccuracyScore




# Decision Tree Classifier
DecisionTreeClassifier_Model = DecisionTreeClassifier()
DecisionTreeClassifier_Model.fit(x_train, y_train)


DecisionTreeClassifier_Predcitions = DecisionTreeClassifier_Model.predict(x_test)


DecisionTreeClassifier_confusion_matrix = confusion_matrix(y_test, DecisionTreeClassifier_Predcitions)
DecisionTreeClassifier_confusion_matrix


DecisionTreeClassifier_AccuracyScore = accuracy_score(y_test, DecisionTreeClassifier_Predcitions)
DecisionTreeClassifier_AccuracyScore


dot_data = StringIO()


export_graphviz(DecisionTreeClassifier_Model, out_file=dot_data)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
dot_data.getvalue()

import graphviz
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
os.environ["PATH"] += os.pathsep + r'C:\Users\user\Anaconda3\Library\bin\graphviz'

Image(graph.create_png())




# Random Forest Classifer
RandomForestClassifier_Model = RandomForestClassifier()
RandomForestClassifier_Model.fit(x_train, y_train)


RandomForestClassifier_Predcitions = RandomForestClassifier_Model.predict(x_test)


RandomForestClassifier_confusion_matrix = confusion_matrix(y_test, RandomForestClassifier_Predcitions)
RandomForestClassifier_confusion_matrix


RandomForestClassifier_AccuracyScore = accuracy_score(y_test, RandomForestClassifier_Predcitions)
RandomForestClassifier_AccuracyScore


dot_data = StringIO()


export_graphviz(RandomForestClassifier_Model.estimators_[0], out_file=dot_data)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
dot_data.getvalue()


Image(graph.create_png())




matrix_dictionary = { 'logistic_regression': LogisticRegression_confusion_matrix, 'decision_tree': DecisionTreeClassifier_confusion_matrix, 'random_forest': RandomForestClassifier_confusion_matrix }
matrix_dictionary


for label, matrix in matrix_dictionary.items():
    plt.title(label=label)
    sns.heatmap(matrix, annot=True)
    plt.show()


LogisticRegression_classification_report = classification_report(y_test, LogisticRegression_Predictions)
LogisticRegression_classification_report


DecisionTreeClassifier_classification_report = classification_report(y_test, DecisionTreeClassifier_Predcitions)
DecisionTreeClassifier_classification_report


RandomForestClassifier_classification_report = classification_report(y_test, RandomForestClassifier_Predcitions)
RandomForestClassifier_classification_report


from sklearn.metrics import precision_recall_fscore_support as score
precision,recall,fscore,support=score(y_test,LogisticRegression_Predictions,average='macro')
print('Precision : {}'.format(precision))
print('Recall    : {}'.format(recall))
print('F-score   : {}'.format(fscore))
print('Support   : {}'.format(support))


LogisticRegression_classification_report = classification_report(y_test, LogisticRegression_Predictions, output_dict=True )

LRmacro_precision =  LogisticRegression_classification_report['macro avg']['precision'] 
LRmacro_recall = LogisticRegression_classification_report['macro avg']['recall']    
LRmacro_f1 = LogisticRegression_classification_report['macro avg']['f1-score']
LRaccuracy = LogisticRegression_classification_report['accuracy']


DecisionTreeClassifier_classification_report = classification_report(y_test, DecisionTreeClassifier_Predcitions, output_dict=True )

DTCmacro_precision =  DecisionTreeClassifier_classification_report['macro avg']['precision'] 
DTCmacro_recall = DecisionTreeClassifier_classification_report['macro avg']['recall']    
DTCmacro_f1 = DecisionTreeClassifier_classification_report['macro avg']['f1-score']
DTCaccuracy = DecisionTreeClassifier_classification_report['accuracy']


RandomForestClassifier_classification_report = classification_report(y_test, RandomForestClassifier_Predcitions, output_dict=True )

RFCmacro_precision =  RandomForestClassifier_classification_report['macro avg']['precision'] 
RFCmacro_recall = RandomForestClassifier_classification_report['macro avg']['recall']    
RFCmacro_f1 = RandomForestClassifier_classification_report['macro avg']['f1-score']
RFCaccuracy = RandomForestClassifier_classification_report['accuracy']


dictionary0 = {'logistic_regression': [LRmacro_precision, LRmacro_recall, LRmacro_f1, LRaccuracy], 'decision_tree': [DTCmacro_precision, DTCmacro_recall, DTCmacro_f1, DTCaccuracy], 'random_forest': [RFCmacro_precision, RFCmacro_recall, RFCmacro_f1, RFCaccuracy ]}
pd.DataFrame.from_dict(dictionary0, orient='index')

df0 = pd.DataFrame.from_dict(dictionary0, orient='index', columns=['precision', 'recall', 'f1-score', 'accuracy'])
df0


dictionary = { 0 : [LRmacro_precision, LRmacro_recall, LRmacro_f1, LRaccuracy, 'logistic_regression'], 1 : [DTCmacro_precision, DTCmacro_recall, DTCmacro_f1, DTCaccuracy, 'decision_tree'], 2 : [RFCmacro_precision, RFCmacro_recall, RFCmacro_f1, RFCaccuracy, 'random_forest' ]}
pd.DataFrame.from_dict(dictionary, orient='index')

df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['precision', 'recall', 'f1-score', 'accuracy', 'classifier'])
df


labels = ['precision', 'recall', 'f1-score', 'accuracy']
lr = df0.iloc[0].tolist()
dt = df0.iloc[1].tolist()
rf = df0.iloc[2].tolist()

##lr = [0.8475670307845085, 0.8571428571428571, 0.8513258765866534, 0.8571428571428571]
##dt = [0.7183268858800773,0.7303571428571429,0.7114634146341463,0.7142857142857143]
##rf = [0.8144927536231884,0.8321428571428571,0.8109033125534775,0.8131868131868132]

x = np.arange(len(labels))  # The label locations
width = 0.35  # The width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, lr, width, label='logistic_regression')
rects2 = ax.bar(x - width/2, dt, width, label='decision_tree') # If we change "x - width/2" to "x + width/2" decision_tree's oranged-coloured bar disappears, which means it hides behind random_forest bar.
rects3 = ax.bar(x + width/2, rf, width, label='random_forest')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Distribution of Classifiers Metrics/Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


df0.plot(kind='bar', figsize=(16, 16))
plt.legend(['precision', 'recall', 'f1-score', 'accuracy'])
plt.title("Distribution of Classifiers' Metrics/Scores")
plt.xlabel("Classifiers Names")
plt.ylabel("Scores")
plt.plot()


classification_report_dictionary = { 'logistic_regression': LogisticRegression_classification_report, 'decision_tree': DecisionTreeClassifier_classification_report, 'random_forest': RandomForestClassifier_classification_report }
classification_report_dictionary


for label, report in classification_report_dictionary.items():
    plt.title(label=label)
    sns.countplot(report, annot=True)
    plt.show()


scores_dictionary = { 'logistic_regression': [LogisticRegression_AccuracyScore], 'decision_tree': [DecisionTreeClassifier_AccuracyScore], 'random_forest': [RandomForestClassifier_AccuracyScore] }


scores_df = pd.DataFrame(scores_dictionary)
scores_df


scores_df.plot(kind='bar', figsize=(10, 10))
plt.legend(["Logistic Regressor","Decision Tree Classifier","Random Forest Classifier"])
plt.title("Districution of Classifiers Accuracy Scores")
plt.xlabel("Classifiers Names")
plt.ylabel("Accuracy Scores")
plt.plot()

