# "# Data details sorted by column\n",
# "Data taken: productOnline.csv\n",
# "1. brand\n",
# "2. product name\n",
# "3. class (target for classification)\n",
# "- A = highest bought\n",
# "- B = high bought\n",
# "- C = medium bought\n",
# "- D = low bought\n",
# "4. procesor\n",
# "5. CPU\n",
# "6. Ram\n",
# "7. Ram_type\n",
# "8. ROM\n",
# "9. ROM_Type\n",
# "10. GPU\n",
# "11. display_size\n",
# "12. OS\n",
# "13. warranty\n",
# "(1-2, 4-13 = x, 3 = output)"

# "ให้นิสิตใช้ขั้นตอนวิธีต้นไม้ตัดสินใจเรียนรู้จากชุดข้อมูล productOnline วัดประสิทธิภาพ และแสดงผลภาพต้นไม้"\
    
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib as plt

def importdata(url, names):
    product_df = pd.read_csv(
        url, names=names,
        sep=',')
    # Displaying dataset information
    print("Dataset Length: ", len(product_df))
    print("Dataset Shape: ", product_df.shape)
    print("Dataset: ", product_df.head())
    
    return product_df

def splitdataset(balance_data):
    # Separating the targe variable
    X = balance_data.values
    Y = balance_data["class"].values

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100
    )
    
    return X, Y, X_train, X_test, y_train, y_test
    
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, X_test, y_train):
    # Creating the classifier object
    clf_entropy = DecisionTreeClassifier(criterion="entropy",
                                      random_state=100, max_depth=3, min_samples_leaf=5)
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Function to make predictions
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:", y_pred)
    return y_pred

# Placeholder function for cal_accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          metrics.confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
          metrics.accuracy_score(y_test, y_pred)*100)
    print("Report : ",
          metrics.classification_report(y_test, y_pred))

def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15,10))
    plt.plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()
    
if __name__ == "__main__":
    url = "productOnline.csv"
    names = ["brand", "product name", "class", "processor", "CPU"]
    data = importdata("productOnline.csv")
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    
    # Visualizing the Decision Trees
    plot_decision_tree(clf_gini, ['X1', 'X2', 'X3', 'X4'], ['A', 'B', 'C', 'D'])
    plot_decision_tree(clf_entropy, ['X1', 'X2', 'X3', 'X4'], ['A', 'B', 'C', 'D'])
    
    # Operational Phase
    print("Results Using Gini Index:")
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    
    print("Results Using Entropy:")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
