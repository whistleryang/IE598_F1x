#import data and divide it to attribute and target, train and test
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('/Users/whistler/Desktop/MachineLearning/Treasury Squeeze test - DS1.csv', header=0)
X = df.iloc[:, 2:11]
y = df.iloc[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=1, 
                                                    stratify = y)

#fit KNN classifier to data, get scores for different K
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range = range(1, 26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

#plot scores for K in a line chart
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(1,len(scores)+1), scores)
plt.xlabel('Number of K')
plt.ylabel('Scores')
plt.title('Scores for Different K')
plt.show()

#fit DT classifier to data
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',
                              max_depth=3, 
                              random_state=1)
tree.fit(X_train, y_train)

#draw the result of DT
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=['False',
                                        'True'],
                           feature_names= ['price_crossing',
                                           'price_distortion',
                                           'roll_start',
                                           'roll_heart',
                                           'near_minus_next,',
                                           'ctd_last_first',
                                           'ctd1_percent',
                                           'delivery_cost',
                                           'delivery_ratio'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('/Users/whistler/Desktop/MachineLearning/tree.png')

#print request
print("My name is Taiyu Yang")
print("My NetID is: taiyuy2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
