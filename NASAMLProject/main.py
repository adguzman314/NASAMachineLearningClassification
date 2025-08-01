""" Anthony Guzman
    ITP-449
    Assignment number: FINAL PROJECT
    My attempt at craeting the best clasifer for the NASA Kelpler Telescope data
    NOTE: I determined the number of Principal Components to use in a separate script titled "pcagraph.py"
    This was done to save runtime on this script, as the runtime is already long as is. I encourage you to give it a look.
"""
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV,train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import math

def main():
    # ==== DATA WRANGLING =========
    df_kepler_raw = pd.read_csv("cumulative_2023.11.07_13.44.30.csv",skiprows=41) # The first 41 rows are metadata about the columns
    df_kepler = df_kepler_raw.copy()
    df_kepler = df_kepler.drop_duplicates()
    df_kepler = df_kepler.drop(columns="koi_disposition")
    columns_to_drop = [col for col in df_kepler.columns if "err" in col]#Cycle through column names and delete any whose name contains string "err"
    df_kepler.drop(columns=columns_to_drop, inplace=True)
    df_kepler = df_kepler.dropna()

    # Create Training/testing splits
    X = df_kepler.drop(columns='koi_pdisposition')
    y = df_kepler["koi_pdisposition"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=567)

    #Transform the dataset
    scaler=StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Cross Validate models (Must find best model and best hyperparams)
        # Create one optimal NON-PCA model 
        # Creeate one optimal PCA model
        # Compare the two 

    k = int(math.sqrt(X_train.shape[0]))# For KNN specifically
    pipe = Pipeline([
                ('estimator', KNeighborsClassifier()),  # Placeholder for the estimators
        ])


    # create a list of estimators and their hyperparams
    estimator_list = [{
                    'estimator':[KNeighborsClassifier()],# give or take 5 combinations
                    'estimator__n_neighbors': range(k, int(k*1.5))#the square root of num training samples)
    },
        {	# 310 combin
                    'estimator': [DecisionTreeClassifier()],
                    'estimator__max_depth': range(3, 19),
                    'estimator__criterion': ['entropy', 'gini'],
                    'estimator__min_samples_leaf': range(1,11)
    },
    {
        'estimator':[LogisticRegression()]# Just the one
    },
    {# 12 combination
        'estimator':[svm.SVC()],
        'estimator__kernel': ['rbf'],
        'estimator__C' : [0.01,0.1,1,10],
        'estimator__gamma':[1,10,100]

    }
    ]
    #There is a total of about 320 hyperparam combinations, for n_iter, we shall set it to 35

    # ==== NON-PCA Model ====

    rscv = RandomizedSearchCV(# Create a randomized search of some hyperparameter combinations of all the models
    pipe,  # Pipeline object
    param_distributions=estimator_list, # collection of estimators and hyperparams
    scoring='accuracy', # scoring metric
    n_iter=35 # The number of combinations to be tested across all models. For the sake of runtime, we will only test about 10%+
    )
    
    rscv.fit(X_train, y_train)#Fit the models
    best_model_npca = rscv.best_estimator_  # ← this is the tuned model NON PCA
    y_pred_npca = best_model_npca.predict(X_test)
    acc_npca = accuracy_score(y_test, y_pred_npca)

    # === PCA Dataset Model=== 
    # === Repeat Same steps but with PCA dataset ====


    pca = PCA(n_components=3) # determined the amount of components to use in "pcagraph.py"
    X_train_pca = pd.DataFrame(pca.fit(X_train_scaled).transform(X_train_scaled), index=X_train.index)
    X_test_pca = pca.transform(X_test_scaled)

    rscv_pca = RandomizedSearchCV(
    pipe,  
    param_distributions=estimator_list, # collection of estimators and hyperparams
    scoring='accuracy', # scoring metric
    n_iter=35 # The number of combinations to be tested across all models. For the sake of runtime, we will only test about 10%+
    )

    rscv_pca.fit(X_train_pca, y_train)
    best_model_pca = rscv_pca.best_estimator_
    y_pred_pca = best_model_pca.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)

    print(f"Non-PCA Model Accuracy: {acc_npca:.4f}")
    print(f"PCA Model Accuracy:     {acc_pca:.4f}")

    print("\nNon-PCA Classification Report:")
    print(classification_report(y_test, y_pred_npca))

    print("\nPCA Classification Report:")
    print(classification_report(y_test, y_pred_pca))

    optimal_model = None
    if acc_pca > acc_npca:
        optimal_model = best_model_pca
    else:
        optimal_model = best_model_npca

    # Extract which model is inside the pipeline
    chosen_model_name = optimal_model.named_steps['estimator'].__class__.__name__
    print("Best model type:", chosen_model_name)

    # Choose grid_of_params based on which model was deemed best

    if chosen_model_name == 'KNeighborsClassifier':
        grid_of_params = {
            'estimator__n_neighbors': range(k, int(k*1.5))
        }

    elif chosen_model_name == 'DecisionTreeClassifier':
        grid_of_params = {
            'estimator__max_depth': range(3, 19),
            'estimator__min_samples_leaf': range(1, 11),
            'estimator__criterion': ['gini', 'entropy']
        }

    elif chosen_model_name == 'SVC':
        grid_of_params = {
            'estimator__C': [0.01,0.1, 1, 10],
            'estimator__gamma': [1, 10, 100],
            'estimator__kernel': ['rbf']
        }


    gscv = GridSearchCV(estimator=optimal_model, param_grid=grid_of_params)
    if optimal_model == best_model_pca:
        gscv.fit(X_train_pca, y_train)
    else:
        gscv.fit(X_train,y_train)

    final_model = gscv.best_estimator_

    if optimal_model == best_model_pca:
        y_pred_final_model = final_model.predict(X_test_pca)
        acc_final_model = accuracy_score(y_test, y_pred_final_model)
    else:
        y_pred_final_model = final_model.predict(X_test)
        acc_final_model = accuracy_score(y_test, y_pred_final_model)

    # === CONFUSION MATRIX =====
    
    cm = confusion_matrix(y_test, y_pred_final_model)
    labels = gscv.best_estimator_.classes_
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(16, 9)) 
    cm_disp.plot(ax=ax)
    ax.set_title('Confusion Matrix for Final Optimized Model')
    fig.suptitle("", fontsize=16)
    fig.savefig("ConfusionMatrix.png")

    print("\nFinal Model Classification Report:")
    print(classification_report(y_test, y_pred_final_model))
    # We will use permutation importance to determine the most important varaible
    res = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=0)
    importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': res.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

    # ===== Questions ========
"""
    1.) PCA did not improve on the results of the model.
    2.) Using PCA risks losing variability that can help a model increase its ability to correctly
     classify records. It seems that in setting n_compoments to 3, we have done exactly that, and made it more difficult for the model
    to do its job. 
    3) Looking at the Final Model Classification Report, we can see that the model shows roughlyl equal precision and recall score for both the
    CANDIDATE and FALSE POSITIVE classes. Measuring precision is like asking how trustworthy a positive prediction is. For both classes, the precision is quite similar.
    Recall on the other hand measures teh ability to correctly find all instances of a label. For this, the final model
    was better at detecting CANDIDATE cases than FALSE POSITIVE cases by about 0.11.
    4) koi_prad (the radius of the object) seems to be the most infulential variable in the final model
    5) The radius of the celestial object is likely so impactful as a predriction variable due to the fact that in order for an object to be considered a planet, 
    it must be "big enough" so as to differitinate it from other flying rocks and objects in the cosmos. (See Pluto's demotion to a swarf planet in 2006).
    Obvioisly size is not the only prerequisite for being a planet, there could be stars and other massive objects in space, but there are multiple smaller objects
    that are considered too small to be official planets (such as the millions of obejcts in the Kuiper Belt), so filtering by size is likely a good place to start when determing
    whether something is an objet or not.
"""







if __name__ == '__main__':
    main()
