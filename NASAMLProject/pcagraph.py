""" Anthony Guzman
    ITP-449
    Assignment number: FINAL PROJECT
    My script fr determing the number of principal components in main.py 
    There is no need to run this code multiple times in my main script if the result is going to be the same each time, so i decided to write it here
    From this graph, I have determined that the best number of components to use is 3.
"""
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV,train_test_split
import matplotlib.pyplot as plt
df_kepler_raw = pd.read_csv("cumulative_2023.11.07_13.44.30.csv",skiprows=41)
df_kepler = df_kepler_raw.copy()
df_kepler = df_kepler.drop_duplicates()

# Extract the nessary variables from the data set
df_kepler = df_kepler.drop(columns="koi_disposition")

columns_to_drop = [col for col in df_kepler.columns if "err" in col]
df_kepler.drop(columns=columns_to_drop, inplace=True)
df_kepler = df_kepler.dropna()


X = df_kepler.drop(columns='koi_pdisposition')
y = df_kepler["koi_pdisposition"]
#print(df_kepler)

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=567)

pca = PCA()
pca.fit(X_train)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# Plot to visualize

plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA: Variance Explained')
plt.grid(True)
plt.savefig("PCAgraph.png")




