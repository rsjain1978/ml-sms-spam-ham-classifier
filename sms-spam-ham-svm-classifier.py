import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def loadDataset():
    #load spam data file
    df = pd.read_csv('spam.csv', encoding="latin-1")
    df = df.rename(columns={'v1':'label', 'v2':'sms'})
    cols=['label','sms']
    df=df[cols]
    df = df.dropna(axis=0, how='any')

    return df

# This function shows how the Spam and Ham messages are distributed.
def printClassDistributionGraph():

    df = loadDataset()

    #Split dataset into X & Y
    Y=df['label']
    X=df['sms']

    sns.countplot(x=Y)
    plt.xlabel('Message Classes')
    plt.show()

def printFrequentWordsDistribution():
    print ('Not implemented yet')

def printSVMModelPerformance (rf, modelScore, train_X, actualY, predictedY):

    print ('    ')
    print ('********************* Classifier Performance Report On Test Data ***********************')
    
    print ('Model Score is ', modelScore)
                                        
    #print the confusion matrix
    c_matrix = confusion_matrix(actualY, predictedY)
    print (c_matrix)

    print ('Accuracy score is',accuracy_score(actualY, predictedY))
    print ('Recall score is', recall_score(actualY, predictedY, average='weighted'))
    print ('Precision store is', precision_score(actualY, predictedY, average='weighted'))
    print ("F1 score is", f1_score(actualY, predictedY, average='weighted'))

    #print the classification report
    #print (classification_report(actualY, predictedY))

# pre process the data. this would include
# 1. get all words from the message
# 2. convert words to lowercase
# 3. filter out all english stop words from the messages
# 4. create an array of all processed words.
import re
from nltk.corpus import stopwords

def preProcessData(data):
    print ('Pre-processing data..')
    countMessage = data['label'].size

    processedSMS = []
    for i in range(0, countMessage):
        #print (data['label'][i])
        sms = data['label'][i]
        letters_only=re.sub("[^a-zA-Z]"," ",sms)
        words=letters_only.lower().split()
        stops=set(stopwords.words("english"))
        filtered_words=[w for w in words if not w in stops]
        processedSMS.append(" ".join(filtered_words))

    return countMessage, processedSMS

def runSVMClassifier(trainX, testX, trainY, testY):
    #Use countvectorizer to convert the cleaned messages into an array of numbers. this workds as follows:
    # Assuming three messages:
    # msg1 = "Hello world"
    # msg2 = "world is flat"
    # msg3 = "Say hello to the world which is flat and says hello"
    # 
    # Now, 
    # In message1, there are two words, Hello and World, count of each is 1  
    # In message2, there are three words, count of each is 1
    # In message3, there are 11 workds & count of hello is 2 and others are 1
    # 
    # CountVectorizer would create an array which would have all count of all words at each sentence. 
    print ('Converting text into count vectors')
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer="word", max_features=5000)
    trainX_cv = cv.fit_transform(trainX)
    testX_cv = cv.fit_transform(testX)

    print ('Training the model')
    classifier = svm.SVC(C=1.0, kernel='linear', random_state=1, gamma=1)
    model = classifier.fit (trainX_cv, trainY)
    modelScore = model.score(testX_cv, testY)

    print ('Testing the model')
    predictedY = model.predict(testX_cv)

    printSVMModelPerformance(classifier, modelScore, trainX_cv, testY, predictedY)

    return modelScore

from sklearn.model_selection import KFold
def runClassifier(validation_method):

    #load the data
    df = loadDataset()

    #preprocess data set
    count, processedSMS = preProcessData(df)

    #Create new dataframe column
    df["processed_sms"]=processedSMS
    columns=["processed_sms","label"]
    df=df[columns]

    #convert the labels to 0 and 1 values
    #df.loc[df['label']=='ham','label']=0
    #df.loc[df['label']=='spam','label']=1

    if (validation_method==0):
        
        kFolds = 5
        kf = KFold(n_splits=kFolds, shuffle=True, random_state=10)

        runningModelScore = 0
        for train_index, test_index in kf.split(df['processed_sms']):

            trainX = df['processed_sms'].iloc[train_index]
            testX = df['processed_sms'].iloc[test_index]
            trainY = df['label'].iloc[train_index]
            testY = df['label'].iloc[test_index]

            score = runSVMClassifier(trainX, testX, trainY, testY)
            runningModelScore = runningModelScore + score
                    
        print ('Average model score is ', runningModelScore/5)
        
    if (validation_method==1):

        trainX = df['processed_sms'][:5000]
        testX = df['processed_sms'][5001:count]
        trainY = df['label'][:5000]
        testY = df['label'][5001:count]

        runSVMClassifier(trainX, testX, trainY, testY)
        

def processChoice(choice):

    choice = int (choice)
    if (choice==1):
        printClassDistributionGraph()
    if (choice ==2):
        printFrequentWordsDistribution()
    if (choice ==3):
        runClassifier(validation_method=0)
    if (choice ==4):
        runClassifier(validation_method=1)
    else :
        print ('Invalid choice')

#Display execution options to the user and process accordingly
print ('Please press 1 if you want to see the distribution of classes')
print ('Please press 2 if you want to see the plot distribution of frequent words')
print ('Please press 3 if you want to run the SVM classifier using cross validation')
print ('Please press 4 if you want to run the SVM classifier using hold out')

mode = input("Please enter your choice: ")
processChoice(mode)