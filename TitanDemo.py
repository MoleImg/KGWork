# coding=UTF-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import Series, DataFrame

data_train = pd.read_csv("Titanic/train.csv")
# print(data_train)
data_test = pd.read_csv("Titanic/test.csv")
# print(data_test)

'''
Data visualization
'''

fig = plt.figure()
fig.set(alpha=0.2)

# People
plt.subplot2grid((2,3), (0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title("Surviver distribution(1 for survived)")
plt.ylabel("Count")

# P Class
plt.subplot2grid((2,3), (0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title("P-class")
plt.ylabel("Count")

# Age vs. survival
plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.xlabel("If survived")
plt.ylabel("Age")
plt.title("Age distribution of if survived")

# Age vs. p-class
plt.subplot2grid((2,3), (1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Age distribution of each p-class")
plt.legend(("1st","2nd","3rd"), loc='best')

# Embarked
plt.subplot2grid((2,3), (1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.ylabel("Counts")
plt.title("People distribution of embarked")

# p-class vs. survival
plt.subplot(221)
survived_count = data_train.Pclass[data_train.Survived == 1].value_counts()
unsurvived_count = data_train.Pclass[data_train.Survived == 0].value_counts()
df = pd.DataFrame({"Survived": survived_count, "Unsurvived": unsurvived_count})
print(survived_count)
print(unsurvived_count)
print(df)
df.plot(kind='bar', stacked=True)
plt.xlabel("p-class")
plt.ylabel("Counts")
plt.title("P-class vs. survival")
#
# # gender vs. survival
# plt.subplot(222)
# survived_count = data_train.Sex[data_train.Survived == 1].value_counts()
# unsurvived_count = data_train.Sex[data_train.Survived == 0].value_counts()
# df = pd.DataFrame({"Survived": survived_count, "Unsurvived": unsurvived_count})
# # print(survived_count)
# # print(unsurvived_count)
# # print(df)
# df.plot(kind='bar', stacked=True)
# plt.xlabel("Gender")
# plt.ylabel("Counts")
# plt.title("Gender vs. survival")
#
# # Embarked vs. survival
# plt.subplot(223)
# survived_count = data_train.Embarked[data_train.Survived == 1].value_counts()
# unsurvived_count = data_train.Embarked[data_train.Survived == 0].value_counts()
# df = pd.DataFrame({"Survived": survived_count, "Unsurvived": unsurvived_count})
# # print(survived_count)
# # print(unsurvived_count)
# # print(df)
# df.plot(kind='bar', stacked=True)
# plt.xlabel("Embarked")
# plt.ylabel("Counts")
# plt.title("Embarked vs. survival")

# Cabin
# print(data_train.Cabin.value_counts())
# survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df = pd.DataFrame({"Cabin": survived_cabin, "Nocabin": survived_nocabin})
# # print(survived_cabin)
# # print(survived_nocabin)
# # print(df)
# df.plot(kind='bar')
# plt.xlabel("Cabin")
# plt.ylabel("Counts")


plt.show()

# '''
# Data pre-processing
# '''
#
#
# from sklearn.ensemble import RandomForestRegressor
# def set_missing_ages(df):
#     '''
#     Using random forest to fill up the vacant ages
#     '''
#     age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
#     known_age = age_df[age_df.Age.notnull()].as_matrix()
#     unknown_age = age_df[age_df.Age.isnull()].as_matrix()
#
#     age_y = known_age[:,0]
#     age_x = known_age[:,1:]
#     # fitting
#     rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
#     rfr.fit(age_x, age_y)
#
#     predict_age = rfr.predict(unknown_age[:,1::])
#
#     df.loc[(df.Age.isnull()), 'Age'] = predict_age
#
#     return df, rfr
#
# def set_Cabin_type(df):
#     '''
#     Convert Cabin into two categories
#     '''
#     df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
#     df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
#     return df
#
# '''
# Missing feature filling and converting
# '''
# data_train, rfr = set_missing_ages(data_train)
# data_train = set_Cabin_type(data_train)
# # print(data_train)
#
# '''
# Category into numerical value
# '''
# dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
# # print(dummies_Cabin)
# dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
# dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
# dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
#
# df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Pclass, dummies_Sex], axis=1)
# df.drop(['Cabin', 'Embarked', 'Sex', 'Pclass', 'Ticket', 'Name'], axis=1, inplace=True)
# # print(df)
#
# '''
# Normalization
# '''
# import sklearn.preprocessing as preprocessing
# scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(df['Age'])
# df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
# fare_scale_param = scaler.fit(df['Fare'])
# df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
#
# # print(df)
#
# '''
# select the features
# '''
# train_df = df.filter(regex=r'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# train_np = train_df.as_matrix()
# # print(train_df)
#
# '''
# Training & validation data
# '''
# from sklearn.cross_validation import train_test_split
# train_y = train_np[:,0]
# train_y = train_y.reshape((train_y.shape[0], 1))
# train_x = train_np[:,1:]
#
# train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)
# # print(np.shape(train_y))
# # print(train_y)
# # print(np.shape(train_x))
#
#
# '''
# Testing
# '''
#
# '''
# Missing procsessing
# '''
# data_test.loc[(data_test.Fare.isnull()), 'Fare' ] = 0
#
# tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
# null_age = tmp_df[data_test.Age.isnull()].as_matrix()
#
# X = null_age[:, 1:]
# predictedAges = rfr.predict(X)
# data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
# data_test = set_Cabin_type(data_test)
#
# '''
# Category to numerical
# '''
# dummies_Cabin = pd.get_dummies(data_test.Cabin, prefix='Cabin')
# dummies_Sex = pd.get_dummies(data_test.Sex, prefix='Sex')
# dummies_Pclass = pd.get_dummies(data_test.Pclass, prefix='Pclass')
# dummies_Embarked = pd.get_dummies(data_test.Embarked, prefix='Embarked')
# # Merge
# df_test = pd.concat([data_test, dummies_Cabin, dummies_Pclass, dummies_Sex, dummies_Embarked], axis=1)
# df_test.drop(['Cabin', 'Embarked', 'Sex', 'Pclass', 'Ticket', 'Name'], axis=1, inplace=True)
#
# '''
# Normalization
# '''
# scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(df_test['Age'])
# df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
# fare_scale_param = scaler.fit(df_test['Fare'])
# df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
#
# # print(df)
#
# '''
# select features
# '''
# test_df = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# test_np = test_df.as_matrix()
#
# '''
# Testing data
# '''
# # print(test_df)
# test_x = test_np
#
# def accuracy(predictions, labels):
#     tmp = np.sum(np.abs(predictions - labels))
#     return ((np.ones((1)) - tmp / predictions.shape[0]))
#
# '''
# Modeling
# sklearn.linear_model.LogisticRegression
# '''
#
# from sklearn import linear_model
# # Logistic regression
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# # clf = linear_model.SGDClassifier(loss='log')
# # Training
# clf.fit(train_x, train_y)
# # Validating
# lr_results = clf.predict(valid_x)
# # mean_acc = clf.score(valid_x, valid_y)
# # print(lr_results.shape)
# lr_results = lr_results.reshape((lr_results.shape[0], 1))
# mean_acc = accuracy(lr_results, valid_y)
# print("Validating accuracy: %.5f" % (mean_acc))
# Testing
# predictions = clf.predict(test_x)
# test_result = pd.DataFrame({"PassengerId": data_test['PassengerId'].as_matrix(), "Survived": predictions.astype(np.int32)})
# # test_result.to_csv("Titanic/test_predictions6.csv", index=False)
#
# df_ana = pd.DataFrame({"Features": list(train_df.columns)[1:], "Coefs": list(clf.coef_.T)})
# # print(df_ana)
#
# '''
# Cross validation
# '''
# from sklearn import cross_validation
# scores = cross_validation.cross_val_score(clf, train_x, train_y, cv=5)
# print(scores)
# print(scores.mean())


# '''
# Modeling
# Bagging
# '''
# from sklearn.ensemble import BaggingRegressor
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8,
#                                bootstrap=True, bootstrap_features=False, n_jobs=-1)
# # Training
# bagging_clf.fit(train_x, train_y)
# # Validating
# bagging_results = bagging_clf.predict(valid_x)
# # print(bagging_results.shape)
# bagging_results = bagging_results.reshape((bagging_results.shape[0], 1))
# print("Validating mean accuracy: %.5f" % (accuracy(bagging_results, valid_y)))

'''
Modeling
deep network by tensorflow
'''

# Deep network
import tensorflow as tf

def accuracy(predictions, labels):
    tmp = np.sum(np.abs(predictions - labels))
    return (100.0 * (np.ones((1)) - tmp / predictions.shape[0]))

def add_hidden_layer(data, units, activation=tf.nn.relu):
    # layer_weights = tf.Variable(tf.truncated_normal([input_num, output_num], stddev=0.1, dtype=tf.float32))
    # layer_biases = tf.Variable(tf.constant(1.0, shape=[output_num], dtype=tf.float32))
    # return tf.nn.relu(tf.matmul(data, layer_weights) + layer_biases)
    # initialization of weights using Xavier initializer
    initializer = tf.contrib.layers.xavier_initializer()
    hidden = tf.layers.dense(data, units, activation=activation, kernel_initializer=initializer)
    # fc=tf.layers.batch_normalization(hidden, training=is_training)
    return hidden

def add_out_layer(data, units):
    # layer_weights = tf.Variable(tf.truncated_normal([input_num, output_num], stddev=0.1, dtype=tf.float32))
    # layer_biases = tf.Variable(tf.constant(1.0, shape=[output_num], dtype=tf.float32))
    # return tf.matmul(data, layer_weights) + layer_biases
    return tf.layers.dense(data, units, activation=None)

def get_batch(data_x, data_y, batch_size):
    batch_n = data_x.shape[0] // batch_size
    for i in range(batch_n):
        batch_x = data_x[i*batch_size:(i+1)*batch_size,:]
        batch_y = data_y[i*batch_size:(i+1)+batch_size,:]
        yield batch_x, batch_y

graph = tf.Graph()
with graph.as_default():
    tf_x = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))
    tf_y = tf.placeholder(tf.int32, shape=(None, train_y.shape[1]))
    # tf_test_x = tf.constant(test_x, dtype=tf.float32)

    # hidden_layer_weights = tf.Variable(tf.truncated_normal([train_x.shape[1], 10], stddev=0.1, dtype=tf.float32))
    # hidden_layer_biases = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32))
    # out_layer_weights = tf.Variable(tf.truncated_normal([10, train_y.shape[1]], stddev=0.1, dtype=tf.float32))
    # out_layer_biases = tf.Variable(tf.constant(1.0, shape=[train_y.shape[1]], dtype=tf.float32))
    # hidden = tf.nn.relu(tf.matmul(tf_x, hidden_layer_weights)+hidden_layer_biases)
    # logits = tf.matmul(hidden, out_layer_weights) + out_layer_biases

    hidden = add_hidden_layer(tf_x, 10)
    logits = add_out_layer(hidden, 1)
    y_pred = tf.round(tf.nn.sigmoid(logits))

    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=tf_y, logits=logits))
    # loss = tf.reduce_mean(tf.reduce_sum(tf.pow((train_pred - tf_train_y),2)))
    global_step = tf.Variable(0)
    learning_rate = 0.05
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    acc_ref = tf.metrics.accuracy(labels=tf_y, predictions=y_pred)[1] # return (accuracy, update_op), 2 local variables

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

x_collect = []
train_loss_collect = []
train_acc_collect = []
train_acc_collect_ref = []
valid_loss_collect = []
valid_acc_collect = []
valid_acc_collect_ref = []
epoch_num = 2000
batch_size = 16
batch_num = int(train_x.shape[0] / batch_size)



with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    for epoch in range(epoch_num):    # total training time
        for steps in range(batch_num):    # mini-batch gradient descent
            batch_data = train_x[steps*batch_size:(steps+1)*batch_size,:]
            batch_labels = train_y[steps*batch_size:(steps+1)*batch_size,:]
            feed_dict = {tf_x:batch_data, tf_y:batch_labels}
            _, l, predictions, acc = sess.run([optimizer, loss, y_pred, acc_ref], feed_dict=feed_dict)

        if (epoch%50) == 0:
            # Training
            print('Training loss at epoch %d/%d: %f' % (epoch, epoch_num, l))
            print('Training accuracy at step %d/%d: %.5f' % (epoch, epoch_num, accuracy(predictions, batch_labels)))
            print('Training accuracy_ref at step %d/%d: %.5f' % (epoch, epoch_num, acc))
            x_collect.append(epoch)
            train_loss_collect.append(l)
            train_acc_collect.append(accuracy(tf.round(predictions).eval(), batch_labels))
            train_acc_collect_ref.append(acc)

            # Validation
            feed_dict = {tf_x:valid_x, tf_y:valid_y}
            _, lv, pred_v, acc_v = sess.run([optimizer, loss, y_pred, acc_ref], feed_dict=feed_dict)
            print('Validating loss at epoch %d/%d: %f' % (epoch, epoch_num, lv))
            print('Valadating accuracy at step %d/%d: %.5f' % (epoch, epoch_num, accuracy(pred_v, valid_y)))
            print('Validating accuracy_ref at step %d/%d: %.5f' % (epoch, epoch_num, acc_v))
            # x_collect.append(epoch)
            valid_loss_collect.append(lv)
            valid_acc_collect.append(accuracy(tf.round(pred_v).eval(), valid_y))
            valid_acc_collect_ref.append(acc_v)


    # Test
    # test_pred_data = sess.run(y_pred, feed_dict={tf_x:test_x})
    # test_pred_data = test_pred_data.reshape((test_x.shape[0],))
    # test_result = pd.DataFrame({"PassengerId":data_test['PassengerId'].as_matrix(), "Survived":test_pred_data})
    # test_result.to_csv("Titanic/test_predictions7.csv", index=False)
    # print('Testing accuracy: %.2f' % (accuracy(test_pred_data, test_y1)))
    print('Testing predictions finished!')

    # saver = tf.train.Saver()
    # saver.save(sess, "./Titanic/tfModel.ckpt")
# Visualization
plt.cla()
plt.subplot(231)
plt.plot(x_collect, train_loss_collect)
plt.title("Loss vs. training")
plt.subplot(232)
plt.plot(x_collect, train_acc_collect)
plt.title("Accuracy vs. training")
plt.subplot(233)
plt.plot(x_collect, train_acc_collect_ref)
plt.title("Accuracy_ref vs. training")
plt.subplot(234)
plt.plot(x_collect, valid_loss_collect)
plt.title("Loss vs. validating")
plt.subplot(235)
plt.plot(x_collect, valid_acc_collect)
plt.title("Accuracy vs. validating")
plt.subplot(236)
plt.plot(x_collect, valid_acc_collect_ref)
plt.title("Accuracy_ref vs. validating")
plt.show()

