## Final Project On Wine Quality Dataset 
pip install pandas
import pandas as pd

data = pd.read_csv("/Users/lozamengistu/Downloads/winequality-white.csv")



```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

Training using the Raw Data


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# preprocess the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
data = pd.read_csv(url, delimiter=';')

# split the data 
X = data.drop('quality', axis=1)
y = data['quality']

#training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train a Random Forest regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# predictions 
y_pred = model.predict(X_test)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

```

    Mean Squared Error: 0.34775581632653063


Training using PCA 


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Download and preprocess the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
data = pd.read_csv(url, delimiter=';')


X = data.drop('quality', axis=1)
y = data['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)


model = RandomForestRegressor(random_state=42)
model.fit(X_train_pca, y_train)


X_test_pca = pca.transform(X_test)

y_pred = model.predict(X_test_pca)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

```

    Mean Squared Error: 0.5852212244897959



```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Download and preprocess the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
data = pd.read_csv(url, delimiter=';')


X = data.drop('quality', axis=1)
y = data['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


kpca = KernelPCA(n_components=2, kernel='rbf')
X_train_kpca = kpca.fit_transform(X_train)


model = RandomForestRegressor(random_state=42)
model.fit(X_train_kpca, y_train)


X_test_kpca = kpca.transform(X_test)


y_pred = model.predict(X_test_kpca)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

```

    Mean Squared Error: 0.6508529423082209



```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Download and preprocess the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
data = pd.read_csv(url, delimiter=';')


X = data.drop('quality', axis=1)
y = data['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)


model = RandomForestRegressor(random_state=42)
model.fit(X_train_tsne, y_train)


X_test_tsne = tsne.fit_transform(X_test)
y_pred = model.predict(X_test_tsne)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

    Mean Squared Error: 0.8755918367346938



```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Download and preprocess the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
data = pd.read_csv(url, delimiter=';')

# Split the data 
X = data.drop('quality', axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors regressor on the raw data
model = KNeighborsRegressor()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

```

    Mean Squared Error: 0.6393877551020408



```python
import os
print(os.getcwd())

```

    /Users/lozamengistu



```python
data = pd.read_csv("/Users/lozamengistu/Downloads/winequality-white.csv")

```


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# URL link to the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# Load the dataset from the URL
data = pd.read_csv(url, delimiter=';')

# Split the data into features (X) and target variable (y)
X = data.drop('quality', axis=1)  # Assuming 'quality' is the target column name
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier with the RBF kernel
svm = SVC(kernel='rbf')

# Train the SVM classifier on the training data
svm.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_svm = svm.predict(X_test)

# Initialize the KNN classifier with a chosen value of k
knn = KNeighborsClassifier(n_neighbors=5)  # Setting k=5, you can adjust this value as needed

# Train the KNN classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_knn = knn.predict(X_test)

# Evaluate the accuracy of the SVM classifier
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy of SVM:", accuracy_svm)

# Evaluate the accuracy of the KNN classifier
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of KNN:", accuracy_knn)

```

    Accuracy of SVM: 0.44285714285714284
    Accuracy of KNN: 0.4826530612244898



```python
import pandas as pd

# Load the wine quality dataset
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')

# Print the column names
print(data.columns)

```

    Index(['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'], dtype='object')



```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the wine quality dataset
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')

# Plot histograms for each column
for column in data.columns:
    plt.hist(data[column], bins=10, edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(column))
    plt.show()


```


    
![png](output_16_0.png)
    



