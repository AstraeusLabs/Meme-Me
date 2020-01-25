def PCA(trainingSET, testingSET):
    '''
    pca for dimensionality reduction
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(trainingSET)
    
    from sklearn.preprocessing import StandardScaler
    
    features = ['length', 'adult', 'violence', 'racy']
    
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,['pass']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                               , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
        
    finalDf = pd.concat([principalDf, df[['pass']]], axis = 1)

    finalDf.to_csv(trainingSET[:-4]+"PCA"+".csv", index=False)

    #FILE 2
    df = pd.read_csv(testingSET)
    features = ['length', 'adult', 'violence', 'racy']
    
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,['pass']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                               , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
        
    finalDf = pd.concat([principalDf, df[['pass']]], axis = 1)

    finalDf.to_csv(testingSET[:-4]+"PCA"+".csv", index=False)
    print(finalDf)
                               
    return neuralNetPCA(trainingSET[:-4]+"PCA"+".csv",testingSET[:-4]+"PCA"+".csv")

def classify(trainingSet, testingSet):
    '''
    build svm model with pca data set
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    training = pd.read_csv(trainingSet)
    
    testing = pd.read_csv(testingSet)

    
    X_train, X_test, y_train, y_test = training.drop('pass', axis=1), testing[['principal component 1', 'principal component 2', 'principal component 3']], training['pass'], testing['pass']
    
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    
    y_pred = svclassifier.predict(X_test)
    print(y_pred)
    import statistics
    final = statistics.mode(y_pred)
    
    return(final)

def classify2(trainingSet, testingSet):
    '''
    Build SVM model
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    training = pd.read_csv(trainingSet)
    
    testing = pd.read_csv(testingSet)
    #training.drop('id',axis=1)
    #testing.drop('id',axis=1)
    X_train, X_test, y_train, y_test = training.drop('pass', axis=1), testing[['length', 'adult', 'violence', 'racy']], training['pass'], testing['pass']
    
    from sklearn.svm import SVC

    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    
    y_pred = svclassifier.predict(X_test)
    print(y_pred)
    import statistics
    final = statistics.mode(y_pred)
    
    return(final)

def classifyForest(trainingSet, testingSet):
    '''
    build Random Forest Regression classifier model
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    training = pd.read_csv(trainingSet)
    
    testing = pd.read_csv(testingSet)

    X_train, X_test, y_train, y_test = training.drop('pass', axis=1), testing[['length', 'adult', 'violence', 'racy']], training['pass'], testing['pass']
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    import statistics
    final = statistics.mode(y_pred)
    
    return(final)

def knearest(trainingSet, testingSet):
    '''
    build knn classificiation model 
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    training = pd.read_csv(trainingSet)
    
    testing = pd.read_csv(testingSet)
    #training.drop('id',axis=1)
    #testing.drop('id',axis=1)
    X_train, X_test, y_train, y_test = training.drop('pass', axis=1), testing[['length', 'adult', 'violence', 'racy']], training['pass'], testing['pass']
    
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    print(y_pred)

    import statistics
    final = statistics.mode(y_pred)

    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
    except:
        print("Error in analysis")
    
    return(final)

def neuralNet(trainingSet, testingSet):
    '''
    Build Neural Net Model
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    training = pd.read_csv(trainingSet)
    
    testing = pd.read_csv(testingSet)
    #training.drop('id',axis=1)
    #testing.drop('id',axis=1)
    X_train, X_test, y_train, y_test = training.drop('pass', axis=1), testing[['length', 'adult', 'violence', 'racy']], training['pass'], testing['pass']
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())

    from joblib import dump, load
    import time

    dump(mlp, 'MLPClassifier'+str(time.time())+'.joblib')

    predictions = mlp.predict(X_test)


    import statistics
    final = statistics.mode(predictions)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test,predictions))
    except:
        print("Error in analysis")
    
    return(final)

def neuralNetPCA(trainingSet, testingSet):
    '''
    Build Neural Net Model with PCA dataset
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    training = pd.read_csv(trainingSet)
    
    testing = pd.read_csv(testingSet)
    #training.drop('id',axis=1)
    #testing.drop('id',axis=1)
    X_train, X_test, y_train, y_test = training.drop('pass', axis=1), testing[['principal component 1', 'principal component 2', 'principal component 3']], training['pass'], testing['pass']    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())

    from joblib import dump, load
    import time

    dump(mlp, 'MLPClassifierPCA'+str(time.time())+'.joblib')

    predictions = mlp.predict(X_test)


    import statistics
    final = statistics.mode(predictions)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test,predictions))
    except:
        print("Error in analysis")
    
    return(final)

def neuralNet(trainingSet, testingSet, model):
    '''
    Runs Neural Net Classificiation With a premade model
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    training = pd.read_csv(trainingSet)
    
    testing = pd.read_csv(testingSet)
    #training.drop('id',axis=1)
    #testing.drop('id',axis=1)
    X_train, X_test, y_train, y_test = training.drop('pass', axis=1), testing[['length', 'adult', 'violence', 'racy']], training['pass'], testing['pass']
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    from joblib import dump, load

    mlp = load(model) 

    predictions = mlp.predict(X_test)


    import statistics
    final = statistics.mode(predictions)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test,predictions))
    except:
        print("Error in analysis")
    
    return(final)


#PCA("finalMemesTrain.csv", "finalMemesTest.csv")

#classify2("finalMemesTrain1.csv", "finalMemesTest1.csv")
#print(PCA("finalMemesTrain1.csv", "finalMemesTest1.csv"))
print(neuralNet("finalMemesTrain1.csv", "finalMemesTest1.csv", "MLPClassifier-67.joblib"))
