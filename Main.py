# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def loadData():
    # Load dataset
    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
    # shape (rows = num of examples, cols= num of attributes)
    print(dataset.shape)
    # head (print the 20 first examples)
    print(dataset.head(20))
    # descriptions (prints some measurements such as mean, std, etc)
    print(dataset.describe())
    # class distribution (prints how many instances are from each class)
    print(dataset.groupby('class').size())
    # Univariate plots
    # box and whisker plots
    '''
    This type of graph is used to show the shape of the distribution,
    its central value, and its variability.
     In a box and whisker plot: the ends of the box are the upper and lower 
     quartiles, so the box spans the interquartile range. the median is marked
      by a vertical line inside the box 
    '''
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()
    # histograms
    dataset.hist()
    plt.show()
    #Multivariate Plots
    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()


def main():
    loadData()

if __name__ == "__main__":
    main()

