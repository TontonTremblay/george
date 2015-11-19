import pandas as pd 
import numpy as np 
import tensorflow as tf
import seaborn as sns
import os



def getDataTest(data,shuffle=True):
    if shuffle:
        data = data.reindex(np.random.permutation(data.index))
    data = data.values
    X = data[:,:-2]
    y = data[:,-1]
    #Change the y to a 2dimensionnal 
    ytemp = []
    for i in y:
        if i is 0:
            ytemp.append([1,0])
        else:
            ytemp.append([0,1])
    y = np.array(ytemp)
    return [X,y]

def getDataTrainTest(data,shuffle=True):

    X,y = getDataTest(data,shuffle)    
    t = 0.15 #keep 15 percent for testing
    x_train = X[t*len(X):]
    x_test = X[:t*len(X)]
    y_train = y[t*len(y):]
    y_test = y[:t*len(y)]
    return (x_train,y_train,x_test,y_test)


def Fit(data,debug=False):
    #2 is hard coded but it should be the number of unique things in y or xy[1]
    sess = tf.InteractiveSession()
    numbClasses = 2
    numbDim= len(data[0][0])


    # Parameters
    learning_rate = 0.0005
    training_epochs = 8
    batch_size = 100
    display_step = 100

    #Network Parameters
    n_hidden_1 = 300
    n_hidden_2 = 150
    n_hidden_3 = 75
    n_input = numbDim 
    n_classes = numbClasses 

    # Create model
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    def multilayer_perceptron(_X, _weights, _biases):
        layer_1 = tf.nn.relu(tf.matmul(_X, _weights['h1']) + _biases['b1']) #Hidden layer with RELU activation
        layer_2 = tf.nn.relu(tf.matmul(layer_1, _weights['h2']) + _biases['b2']) #Hidden layer with RELU activation
        layer_3 = tf.nn.relu(tf.matmul(layer_2, _weights['h3']) + _biases['b3']) #Hidden layer with RELU activation
        return tf.matmul(layer_3, weights['out']) + biases['out']

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    value_accuracy = 0

    xy = data
    init = tf.initialize_all_variables()


    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(xy[0])/batch_size)
        for i in range(total_batch):
            batch = [xy[0][i:i+batch_size],xy[1][i:i+batch_size]]
            sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})
            avg_cost += sess.run(cost, feed_dict={x: batch[0], y: batch[1]})/total_batch

        if debug is True and epoch % display_step == 0:
           print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
    # Test trained model
    if debug is True and len(xy)>2:
        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        value_accuracy = sess.run(accuracy,feed_dict={x: xy[2], y: xy[3]})
        print value_accuracy


    # for i in range(len(xy[2])):


    values_predicted = [None for i in range(3)]
    values_predicted[0] = sess.run(tf.argmax(pred,1) ,feed_dict={x:xy[2]})
    values_predicted[1] = sess.run(pred ,feed_dict={x:xy[2]})
    values_predicted[2] = sess.run(tf.nn.softmax(pred) ,feed_dict={x:xy[2]})

      

    return pred,x,value_accuracy,values_predicted


def Predict(predData,data):
    a = []
    # x = tf.placeholder("float", [None,260])
    pred = predData[0]
    x=predData[1]

    for d in data:
        r = [None for i in range(3)]
        r[0] = sess.run(tf.argmax(pred,1) ,feed_dict={x:d})
        r[1] = sess.run(pred ,feed_dict={x:d})
        r[2] = sess.run(tf.nn.softmax(pred) ,feed_dict={x:d})
        a.append(r)
    return a 

def test():
    df = pd.read_csv('../Data/seq_data_300.csv')
    xy = getDataTrainTest(df,False)
    print xy[0][0]
    print df.s299.unique()
    predData = Fit(xy,debug=True)

def CheckFamillyClan():
    dfTrain = pd.read_csv('../Data/seq_data_train_no_clan.csv')
    dfTest = pd.read_csv('../Data/seq_data_test_clan.csv')
    xy = getDataTest(dfTrain) + getDataTest(dfTest,False)
    predData = Fit(xy,debug=True)

    for i in range(len(predData[3][0])):
        print xy[3][i],predData[3][0][i]

def SeedCheckingWithNTrains(folder='', n = 100):
    
    dftest = pd.read_csv('../Data/'+folder+'seed_seq_test_no_clan.csv')
    dftrain = pd.read_csv('../Data/'+folder+'seed_seq_train_no_clan.csv')
    # xy = getDataTest(dftrain) + getDataTest(dftest)

    # predData = Fit(xy,debug=True)
    # for row in dftest.values:
    #     r = Predict(predData,[[row[:-2]]])
    #     print 'real',row[-1],":",r[0][0]
    dist = []
    for i in range(n):
        xy = getDataTest(dftrain) + getDataTest(dftest)

        dist.append(Fit(xy,debug=True)[2])
        print dist
    sns.distplot(dist)
    path_out = os.path.join('..', 'Results', 'res_seed.pdf')
    sns.plt.savefig(path_out)  

def CheckClanResults(folder=''):

    dftest = pd.read_csv('../Data/'+folder+'seq_data_test_clan.csv')
    dftrain = pd.read_csv('../Data/'+folder+'seq_data_train_no_clan.csv')
    xy = getDataTest(dftrain) + getDataTest(dftest,False)
    predData = Fit(xy,debug=True)

    for i in range(len(predData[3][0])):
        if i %2 == 0:
            print "---"
        print dftest.iloc[i].values[-2],xy[3][i],predData[3][0][i],predData[3][2][i]

def CleanData(DataName,output = ''):
    #Start by looking at the sequence that have intersection/part of 
    #of a clan 
    df = pd.read_csv(DataName)
    a = []
    d = {}
    with open('../Data/glm_intersections.txt','r') as inf:
        for line in inf:
            l = eval(line)
            a.append(l)
            d[l[0]]=1
            d[l[1]]=1
    dataTrain = []
    dataTest = []

    for i in range(len(df)):
        if df.iloc[i][-2] in d:
            dataTest.append(df.iloc[i].values)
        else:
            dataTrain.append(df.iloc[i].values)

    #This should be smarter, but it is late and I feel lazy
    dt = [] 
    for v in a:
        s0,s1 = v 
        for i in dataTest:
            if s0 == i[-2]:
                dt.append(i)
                break
        for i in dataTest:
            if s1 == i[-2]:
                dt.append(i)
                break


    dataTrain = np.array(dataTrain)
    dataTest = np.array(dt)
    dfTrain = pd.DataFrame(dataTrain,columns=df.columns)
    dfTrain.to_csv('../Data/'+output+'seq_data_train_no_clan.csv',index_label=False)
    dfTest = pd.DataFrame(dataTest,columns=df.columns)
    dfTest.to_csv('../Data/'+output+'seq_data_test_clan.csv',index_label=False)

    #data set with the seeds 
    #find the seeds
    def FindSeeds(seeds,name):
        with open(name,'r') as inf:
            for line in inf:
                l = line.split(' ')
                if len(l[0])>5:
                    seeds[l[0]] = 1
            return seeds

    seeds = {}
    seeds = FindSeeds(seeds,'../Data/RF00083.stockholm.txt')
    seeds = FindSeeds(seeds,'../Data/RF00128.stockholm.txt')
    df = dfTrain
    dataTrain = []
    dataTest = []

    for i in range(len(df)):
        if df.iloc[i][-2] in seeds:
            dataTest.append(df.iloc[i].values)
        else:
            dataTrain.append(df.iloc[i].values)

    dataTrain = np.array(dataTrain)
    dataTest = np.array(dataTest)

    dfTrain = pd.DataFrame(dataTrain,columns=df.columns)
    dfTrain.to_csv('../Data/'+output+'seed_seq_train_no_clan.csv',index_label=False)

    dfTest = pd.DataFrame(dataTest,columns=df.columns)
    dfTest.to_csv('../Data/'+output+'seed_seq_test_no_clan.csv',index_label=False)
# SeedCheckingWithNTrains()
# CheckFamillyClan()
# test()
# CleanData('../Data/seq_data_300.csv','seq_complete/')
# SeedCheckingWithNTrains('seq_complete/',n = 20)
CheckClanResults('seq_complete/')
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier() 