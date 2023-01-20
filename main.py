import sys
def block1():
    #required Libraries
    import pandas as pd
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    #Preprocess training set ### 
    #splits gender and image name out of the training data and forms independent dataframe out of them
    data=pd.read_csv('Datasets/celeba/labels.csv')

    labelsTrain = data["\timg_name\tgender\tsmiling"].str.split(pat="\t", n=-1, expand=True)
    labelsTrain.drop(columns =[0,3], inplace = True)
    labelsTrain.columns = ["img_name"  , "gender"]
    labelsTrain = labelsTrain.astype({'gender': 'int32'})
    labelsTrain = labelsTrain.astype({'img_name': 'string'})
    labelsTrain["gender"] = labelsTrain["gender"].replace(-1, 0)
    #Preprocess testing set###
    #splits gender and image name out of the testing data and forms independent dataframe out of them
    data=pd.read_csv('Datasets/celeba_test/labels.csv')

    labelsTest = data["\timg_name\tgender\tsmiling"].str.split(pat="\t", n=-1, expand=True)
    labelsTest.drop(columns =[0,3], inplace = True)
    labelsTest.columns = ["img_name"  , "gender"]
    labelsTest = labelsTest.astype({'gender': 'int32'})
    labelsTest = labelsTest.astype({'img_name': 'string'})
    labelsTest["gender"] = labelsTest["gender"].replace(-1, 0)

    #Load training set################################
    #reads image data from training set ,converts to grayscale, flattens it , and stores it into imageTrain array 
    imagesTrain = np.zeros((5000, 218, 178))

    for i in range(0,5000):
        image = cv.imread('Datasets/celeba/img/' + labelsTrain["img_name"][i])
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        imagesTrain[i] = image
        
    imagesTrain = imagesTrain.reshape(5000,38804)    
    imagesTrain = pd.DataFrame(imagesTrain) 


    #Load Testing set################################
    #reads image data from Testing set ,converts to grayscale, flattens it , and stores it into imageTrain array 
    imagesTest = np.zeros((1000, 218, 178))

    for i in range(0,1000):
        image = cv.imread('Datasets/celeba_test/img/' + labelsTest["img_name"][i])
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        imagesTest[i] = image
        
    imagesTest = imagesTest.reshape(1000,38804)   

    imagesTest = pd.DataFrame(imagesTest)

    #scale coversion
    scaler = MinMaxScaler()

    imagesTrain_scaled = scaler.fit_transform(imagesTrain)
    imagesTest_scaled = scaler.transform(imagesTest)

    #PCA conversion
    pca = PCA(n_components = 500)

    imagesTrain_pca = pca.fit_transform(imagesTrain_scaled)
    imagesTest_pca = pca.transform(imagesTest_scaled)

    imagesTrain_pca = pd.DataFrame(imagesTrain_pca) 
    imagesTest_pca = pd.DataFrame(imagesTest_pca)
    
    #stores pca transformed image data into x_train and x_test respectively then joins labelsTrain and x_train into singular dataframe
    x_train = pd.DataFrame(imagesTrain_pca) 
    x_test = pd.DataFrame(imagesTest_pca)
    x_train = pd.concat([labelsTrain,x_train],axis=1, join='inner')

    #logistic regression#######
    model = LogisticRegression(C=0.012742749857031334, max_iter=50, solver='sag')
    model.fit(x_train.iloc[:,2:], labelsTrain['gender'])
    print("\n")
    print('Model score on testing data: ')
    print(model.score(x_test,labelsTest['gender']))
    lll= input() 


def block2():
     #required Libraries
    import pandas as pd
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import tree
    from sklearn.tree import plot_tree
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    #Preprocess training set
    #splits smiling and image name out of the training data and forms independent dataframe out of them
    data=pd.read_csv('Datasets/celeba/labels.csv')

    labelsTrain = data["\timg_name\tgender\tsmiling"].str.split(pat="\t", n=-1, expand=True)
    labelsTrain.drop(columns =[0,2], inplace = True)
    labelsTrain.columns = ["img_name"  , "smiling"]
    labelsTrain = labelsTrain.astype({'smiling': 'int32'})
    labelsTrain = labelsTrain.astype({'img_name': 'string'})

    #Preprocess testing set
    #splits smiling and image name out of the testing data and forms independent dataframe out of them
    data=pd.read_csv('Datasets/celeba_test/labels.csv')

    labelsTest = data["\timg_name\tgender\tsmiling"].str.split(pat="\t", n=-1, expand=True)
    labelsTest.drop(columns =[0,2], inplace = True)
    labelsTest.columns = ["img_name"  , "smiling"]
    labelsTest = labelsTest.astype({'smiling': 'int32'})
    labelsTest = labelsTest.astype({'img_name': 'string'})

    #Load training set################################
    #reads image data from training set ,converts to grayscale, flattens it , and stores it into imageTrain array 
    imagesTrain = np.zeros((5000, 218, 178))

    for i in range(0,5000):
        image = cv.imread('Datasets/celeba/img/' + labelsTrain["img_name"][i])
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        imagesTrain[i] = image
        
    imagesTrain = imagesTrain.reshape(5000,38804)    
    imagesTrain = pd.DataFrame(imagesTrain) 


    #Load Testing set################################
    #reads image data from Testing set ,converts to grayscale, flattens it , and stores it into imageTrain array 
    imagesTest = np.zeros((1000, 218, 178))

    for i in range(0,1000):
        image = cv.imread('Datasets/celeba_test/img/' + labelsTest["img_name"][i])
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        imagesTest[i] = image
        
    imagesTest = imagesTest.reshape(1000,38804)   

    imagesTest = pd.DataFrame(imagesTest)

    #decision tree##### 
    model = tree.DecisionTreeClassifier(max_depth=6,criterion = 'gini')
    model.fit(imagesTrain,labelsTrain['smiling'])
    print("\nDecision Tree scored on testing set: ")
    print(model.score(imagesTest,labelsTest['smiling']))

    #random forest###

    model = RandomForestClassifier(criterion='entropy', max_depth=5, n_estimators=140, n_jobs=-1)
    model.fit(imagesTrain,labelsTrain['smiling'])
    print("\nRandom Forest scored on testing set: ")
    print(model.score(imagesTest,labelsTest['smiling']))


def block3():
    #required Libraries
    import pandas as pd
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    
    #Preprocess training set
    #splits face_shape and image name out of the training data and forms independent dataframe out of them
    data=pd.read_csv('Datasets/cartoon_set/labels.csv')

    labelsTrain = data["\teye_color\tface_shape\tfile_name"].str.split(pat="\t", n=-1, expand=True)
    labelsTrain.drop(columns =[0,1], inplace = True)
    labelsTrain.columns = [ "face_shape" , "img_name"]
    labelsTrain = labelsTrain.astype({'face_shape': 'int32'})
    labelsTrain = labelsTrain.astype({'img_name': 'string'})

    #Preprocess testing set
    #splits face_shape and image name out of the testing data and forms independent dataframe out of them
    data=pd.read_csv('Datasets/cartoon_set_test/labels.csv')

    labelsTest = data["\teye_color\tface_shape\tfile_name"].str.split(pat="\t", n=-1, expand=True)
    labelsTest.drop(columns =[0,1], inplace = True)
    labelsTest.columns = [ "face_shape" , "img_name"]
    labelsTest = labelsTest.astype({'face_shape': 'int32'})
    labelsTest = labelsTest.astype({'img_name': 'string'})

    #Load training set################################
    #reads image data from training set ,resises it
    imagesTrain = np.zeros((10000, 150 , 150, 3)) 

    for i in range(0,10000):
        image = cv.imread('Datasets/cartoon_set/img/' + labelsTrain["img_name"][i])
        image = image[75:420,95:405] # cut 95 from left / right sides , 75 from top, 80 from the bottom
        image = cv.resize(image, dsize=(150 , 150), interpolation=cv.INTER_CUBIC)

        imagesTrain[i] = image



    #Load Testing set################################
    #reads image data from testing set ,resises it
    imagesTest = np.zeros((2500, 150 , 150, 3))

    for i in range(0,2500):
        image = cv.imread('Datasets/cartoon_set_test/img/' + labelsTest["img_name"][i])
        image = image[75:420,95:405]
        image = cv.resize(image, dsize=(150 , 150), interpolation=cv.INTER_CUBIC)

        imagesTest[i] = image

    #required Libraries
    import tensorflow as tf
    from tensorflow import keras
    from keras import Sequential
    from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
    print("num of gpus available: ",len(tf.config.experimental.list_physical_devices('GPU')))
    from keras.utils import normalize,to_categorical
    from keras.preprocessing.image import ImageDataGenerator
    tf.config.run_functions_eagerly(True)
    #needed for tf to run



    #normalises image data and one hot encodes the labels
    labelsTrain = labelsTrain['face_shape']
    labelsTest = labelsTest['face_shape']

    imagesTrain = normalize(imagesTrain,axis=1)
    imagesTest = normalize(imagesTest,axis=1)

    labelsTrain = to_categorical(labelsTrain)
    labelsTest = to_categorical(labelsTest)

    #creates generator functions
    datagen_train = ImageDataGenerator()
    #needed to save memory

    datagen_Test = ImageDataGenerator()


    #CNN architecture definition
    model = Sequential()

    model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(150,150,3)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

    model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

    model.add(Flatten())

    model.add( Dense(128,activation='relu') )
    model.add( Dense(64,activation='relu') )
    model.add( Dense(5,activation='softmax') )

    #prints model summary
    model.summary()

    #sets CNN model parameters
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    #Start CNN fit function
    history = model.fit( 
        datagen_train.flow(imagesTrain, labelsTrain, batch_size=100),
        epochs=8,
        validation_data=datagen_Test.flow(imagesTest, labelsTest, batch_size=100),
    )

def block4():
    #required Libraries
    import pandas as pd
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score


    #Preprocess training set
    #splits eye_color and image name out of the training data and forms independent dataframe out of them
    data=pd.read_csv('Datasets/cartoon_set/labels.csv')

    labelsTrain = data["\teye_color\tface_shape\tfile_name"].str.split(pat="\t", n=-1, expand=True)
    labelsTrain.drop(columns =[0,2], inplace = True)
    labelsTrain.columns = [ "eye_color" , "img_name"]
    labelsTrain = labelsTrain.astype({'eye_color': 'int32'})
    labelsTrain = labelsTrain.astype({'img_name': 'string'})

    #Preprocess testing set
    #splits eye_color and image name out of the testing data and forms independent dataframe out of them
    data=pd.read_csv('Datasets/cartoon_set_test/labels.csv')

    labelsTest = data["\teye_color\tface_shape\tfile_name"].str.split(pat="\t", n=-1, expand=True)
    labelsTest.drop(columns =[0,2], inplace = True)
    labelsTest.columns = [ "eye_color" , "img_name"]
    labelsTest = labelsTest.astype({'eye_color': 'int32'})
    labelsTest = labelsTest.astype({'img_name': 'string'})


    #Load training set################################
    #reads image data from training set ,resises it, flattens it , and stores it into imageTrain array 
    imagesTrain = np.zeros((10000, 50 , 50, 3)) 

    for i in range(0,10000):
        image = cv.imread('Datasets/cartoon_set/img/' + labelsTrain["img_name"][i])
        image = image[230:290,180:320] # cut 95 from left / right sides , 75 from top, 80 from the bottom
        image = cv.resize(image, dsize=(50 , 50), interpolation=cv.INTER_CUBIC)
        imagesTrain[i] = image
        
    imagesTrain = imagesTrain.reshape(10000,7500)    
    imagesTrain = pd.DataFrame(imagesTrain) 


    #Load Testing set################################
    #reads image data from Testing set ,resises it, flattens it , and stores it into imageTest array 
    imagesTest = np.zeros((2500, 50 , 50, 3))

    for i in range(0,2500):
        image = cv.imread('Datasets/cartoon_set_test/img/' + labelsTest["img_name"][i])
        image = image[230:290,180:320]
        image = cv.resize(image, dsize=(50 , 50), interpolation=cv.INTER_CUBIC)
        imagesTest[i] = image
        
        
    imagesTest = imagesTest.reshape(2500,7500)   
    imagesTest = pd.DataFrame(imagesTest)

    #Convolves flattened data of imageTrain and ImageTest with 1D 3 size filter and then stores it into x_train and x_test array respectively
    x_train=np.zeros((10000,7502))
    for i in range(0,10000):
        x_train[i]=np.convolve(imagesTrain.iloc[i,:],[0.066,-5.599,5.566])
        
    x_test=np.zeros((2500,7502))
    for i in range(0,2500):
        x_test[i]=np.convolve(imagesTest.iloc[i,:],[0.066,-5.599,5.566])

    x_train = pd.DataFrame(x_train) 
    x_test = pd.DataFrame(x_test)

    #runs knn algorithem for n_neighbor values between 100 and 1000 in steps of 100
    for i in range(100,1000,100):
        model = KNeighborsClassifier(n_neighbors=i, weights = 'distance' , n_jobs=-1)
        model.fit(x_train,labelsTrain['eye_color'])
        print("\n accuracy for knn algorithem on "+str(i)+" nearest neighbors is: ")
        print(model.score(x_test,labelsTest['eye_color']))



while True:
    print("Please select which task you would like to run: for A1 type A1 , for A2 type A2 , for B1 Type B1 , for B2 Type B2 , for exit type exit ")
    lll=input()
    if lll == "A1":
        block1()
    elif lll == "A2":
        block2()
    elif lll == "B1":
        block3()
    elif lll == "B2":
        block4()
    elif lll=="exit":
        sys.exit()
