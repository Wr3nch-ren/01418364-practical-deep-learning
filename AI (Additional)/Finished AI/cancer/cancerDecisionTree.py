import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def importdata(url):
    product_df = pd.read_csv(
        url)
    # Displaying dataset information
    print("Dataset Length: ", len(product_df))
    print("Dataset Shape: ", product_df.shape)
    print("Dataset: \n", product_df.head())

    return product_df

def splitdataset(balance_data):
    # Separating the targe variable
    X = balance_data.iloc[:, :-1]
    Y = balance_data["CA level"].values
    
    print(X)
    print(Y)
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100
    )
    
    return X, Y, X_train, X_test, y_train, y_test
    
def train_using_gini(X_train, X_test, y_train):
    
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, class_weight='balanced')#, max_depth=3, min_samples_leaf=5)
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, X_test, y_train):
    # Creating the classifier object
    clf_entropy = DecisionTreeClassifier(criterion="entropy",
                                      random_state=100, class_weight='balanced')#, max_depth=3, min_samples_leaf=5)
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
    
if __name__ == "__main__":
    url = "cancer.csv"
    data = importdata(url)
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    
    # Label Encoding
    le = LabelEncoder()
    
    clf_gini = train_using_gini(X_train.apply(le.fit_transform), X_test.apply(le.fit_transform), y_train)
    clf_entropy = train_using_entropy(X_train.apply(le.fit_transform), X_test.apply(le.fit_transform), y_train)
    
    # Visualizing the Decision Trees for both Gini Index and Entropy
    from sklearn.tree import export_graphviz
    from six import StringIO
    from IPython.display import Image
    import pydotplus
    dot_data = StringIO()
    export_graphviz(clf_gini, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=X.columns, class_names=['A','B','C','D'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('cancer_gini.png')
    Image(graph.create_png())
    
    dot_data = StringIO()
    export_graphviz(clf_entropy, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=X.columns, class_names=['A','B','C','D'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('cancer_entropy.png')
    Image(graph.create_png())
    
    # Operational Phase
    print("Results Using Gini Index:")
    y_pred_gini = prediction(X_test.apply(le.fit_transform), clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    
    print("Results Using Entropy:")
    y_pred_entropy = prediction(X_test.apply(le.fit_transform), clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)