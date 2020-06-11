import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression as Mymodel
from sklearn.ensemble import AdaBoostClassifier as Mymodel1

from sklearn.preprocessing import minmax_scale

import time

import preprocess as ap

#preprocess train and test data to get data without sensitive labels
train_data, train_label, test_data, test_label = ap.preprocess('adult_data.txt', 'adult_test.txt')

total_size = len(train_label)
total_size_test = len(test_label)

train_data_no_sensitive = train_data.drop(columns = ['race', 'gender'])
test_data_no_sensitive = test_data.drop(columns = ['race', 'gender'])

#transform data into one hot vectors
train_data_1hot = ap.get_1hot(train_data_no_sensitive)
test_data_1hot = ap.get_1hot(test_data_no_sensitive)

# Fitting scaler (not really necessary)
scaler = StandardScaler()
scaler.fit(train_data_1hot)  
train_data_scaled = scaler.transform(train_data_1hot)
test_data_scaled = scaler.transform(test_data_1hot)

#implementing GREEDY
print('Implementing Greedy...')
remaining_data = train_data_scaled
hack_data = remaining_data[-2:] #see hack_label below

remaining_label = train_label
hack_label = remaining_label[-2:] #SGD blackbox trainer always needs labels of >=2 classes
length_rem = len(remaining_label)
#lists to collect greedy classifiers and corresponding weights
classifiers = []
weights = np.array([])
count = 1

#train greedy step by step
while length_rem > 0:
    classifiers.append(Mymodel(random_state = 0, solver = 'newton-cg'))
    clf = classifiers[-1]
    data_size = len(remaining_label)

    if len(np.unique(remaining_label)) > 1:
        clf.fit(remaining_data, remaining_label)
        predicted = clf.predict(remaining_data)
        remaining_indices = np.not_equal(predicted, remaining_label)
        if count == 1:
            agree_indices = np.equal(predicted, remaining_label)
            agree_data = remaining_data[agree_indices]
            agree_label = remaining_label[agree_indices]
        remaining_data = remaining_data[remaining_indices]
        remaining_label = remaining_label[remaining_indices]
        length_rem = len(remaining_label)
    else:
        remaining_data = np.append(remaining_data, hack_data, axis = 0)
        remaining_label = remaining_label.append(hack_label)
        clf.fit(remaining_data, remaining_label)
        length_rem = 0
        
    current_weight = data_size - length_rem
    weights = np.append(weights, current_weight)

    count = count + 1
print('Done.')

weights = weights/sum(weights)

greedy_overall_acc_train = ap.accuracy_randomized(classifiers, weights, train_data_scaled, train_label)
greedy_overall_acc_test = ap.accuracy_randomized(classifiers, weights, test_data_scaled, test_label)

#implementing seqPAV
num_iter = 1000
print('Implementing SeqPAV with num_iter = %d' %num_iter)
points_tally = np.ones(len(train_label))
#have a list of classifiers (per time step)
seqpav = []
seqpav_weights2 = []

count = 1
t1 = time.time()
while count <= num_iter: 
    if count % 100 == 0:
        print('Iteration # %d' % count)
    sample_weights = minmax_scale(1 / points_tally, feature_range = (0.1, 10))
    #find current best classifier
    seqpav.append(Mymodel(random_state = 0, solver = 'newton-cg'))
    clf = seqpav[-1]
    clf.fit(train_data_scaled, train_label, sample_weight = sample_weights)
    #update weights
    prediction = clf.predict(train_data_scaled)
    #convert prediction vs labels to 0,1
    update = np.zeros(len(prediction))
    agreement = np.equal(prediction, train_label)
    update[agreement] = 1
    #update tally and move to next round
    w = sum(np.multiply((1/points_tally),update))
    seqpav_weights2.append(w)
    points_tally = points_tally + update
    count = count + 1

t2 = time.time()
print('Done. Time taken = %.2f' %(t2-t1))

#seqpav_weights = np.ones(len(seqpav)) / len(seqpav)
seqpav_weights = seqpav_weights2 / sum(seqpav_weights2)
seqpav_overall_acc_train = ap.accuracy_randomized(seqpav, seqpav_weights, train_data_scaled, train_label)
seqpav_overall_acc_test = ap.accuracy_randomized(seqpav, seqpav_weights, test_data_scaled, test_label)

#Analysis
print('----------Analysis------------')
print("Total number of train data-points = %d" % total_size)
print("Total number of test data-points = %d" % total_size_test)
print("Overall train accuracy of ERM = %.5f " % classifiers[0].score(train_data_scaled, train_label))
print("Overall test accuracy of ERM = %.5f " % classifiers[0].score(test_data_scaled, test_label))
print("Overall train accuracy of Greedy = %.5f " % greedy_overall_acc_train)
print("Overall test accuracy of Greedy = %.5f " % greedy_overall_acc_test)
print("Support of greedy (number of classifiers) = %d" % len(weights))
print("and their weights are:")
print(weights)
print("Number of iteration for SeqPAV = %d" % num_iter)
print("Overall train accuracy of SeqPAV = %.5f " % seqpav_overall_acc_train)
print("Overall test accuracy of SeqPAV = %.5f " % seqpav_overall_acc_test)

erm = classifiers[0]
per = Mymodel1(random_state = 0)
per.fit(train_data_scaled, train_label)
per_preds = per.predict(test_data_scaled)
erm_preds = erm.predict(test_data_scaled)
    
agree_inds = np.equal(erm_preds, test_label)
correct_data = test_data_scaled[agree_inds]
correct_label = test_label[agree_inds].to_numpy()
correct_inds = list(range(len(correct_label)))
disagree_inds = np.not_equal(erm_preds, test_label)
wrong_data = test_data_scaled[disagree_inds]
wrong_label = test_label[disagree_inds].to_numpy()
wrong_inds = list(range(len(wrong_label)))

test_size = len(test_label)
correct_size = len(correct_label)
wrong_size = len(wrong_label)

test_label = test_label.to_numpy()
seqpav_pred = np.zeros(len(test_label))
for i in range(num_iter):
    pred = seqpav[i].predict(test_data_scaled)
    acc = list(map(lambda x,y: float(x == y), pred, test_label))
    seqpav_pred = seqpav_pred + seqpav_weights[i] * np.array(acc)
seqpav_pred = seqpav_pred

greedy_pred = np.zeros(len(test_label))
for i in range(len(classifiers)):
    pred = classifiers[i].predict(test_data_scaled)
    acc = list(map(lambda x,y: float(x == y), pred, test_label))
    greedy_pred = greedy_pred + (weights[i] * np.array(acc))

erm_pred = classifiers[0].predict(test_data_scaled)
erm_pred = list(map(lambda x,y: float(x == y), erm_pred, test_label))

per_pred = per.predict(test_data_scaled)
per_pred = list(map(lambda x,y: float(x == y), per_pred, test_label))

arg_inds = np.argsort(seqpav_pred)
seqpav_acc = seqpav_pred[arg_inds]
greedy_acc = greedy_pred[arg_inds]
erm_acc = np.zeros(len(erm_pred))
per_acc = np.zeros(len(per_pred))
for i in range(len(erm_pred)):
    erm_acc[i] = erm_pred[arg_inds[i]]
    per_acc[i] = per_pred[arg_inds[i]]

seqpav_plot = np.zeros(len(test_label))
greedy_plot = np.zeros(len(test_label))
erm_plot = np.zeros(len(test_label))
per_plot = np.zeros(len(test_label))
for i in range(len(test_label)):
    seqpav_plot[i] = sum(seqpav_acc[0:i+1])/(i+1)
    greedy_plot[i] = sum(greedy_acc[0:i+1])/(i+1)
    erm_plot[i] = sum(erm_acc[0:i+1])/(i+1)
    per_plot[i] = sum(per_acc[0:i+1])/(i+1)

opt_x = np.array([])
opt_plot = np.array([])
for i in range(len(test_label)):
    if (i > 0 and i % 400 == 0) or i == len(test_label)-1:
        inds = arg_inds[0:i+1]
        data = test_data_scaled[inds]
        label = test_label[inds]
        opt = Mymodel(random_state = 0, solver = 'newton-cg')
        opt.fit(data, label)
        opt_score = opt.score(data, label)
        #print(opt_score)
        opt_acc = ((i+1) / test_size) * opt_score * opt_score
        opt_plot = np.append(opt_plot, opt_acc)
        opt_x = np.append(opt_x, i)

    

x = list(range(len(test_label)))
plt.plot(x, erm_plot, color = 'red', linestyle = ':', label = "LR")
plt.plot(x, per_plot, color = 'brown', linestyle = 'dashdot', label = "AdaBoost")
plt.plot(x, greedy_plot, color = 'blue', linestyle = 'dashed', label = "Greedy")
plt.plot(x, seqpav_plot, color = 'green', linestyle = 'solid', label = "hPF")
plt.plot(opt_x ,opt_plot, color = 'magenta', linestyle = 'solid', marker = 'o', label = "PF Lower Bound")
plt.legend(loc="upper left")
plt.grid(axis = 'y')
plt.ylabel('Accuracy on the set of points {0,1,...,x}')
plt.xlabel('Points ordered by their hPF scores.')
plt.show()

erm = classifiers[0]
erm_pred_labels = erm.predict(test_data_scaled)
erm_pred_labels = np.array(list(map(lambda x: int(x == '>50K'), erm_pred_labels)))
erm_pred_proba = np.array(erm.predict_proba(test_data_scaled))
erm_pred_scores = np.array(list(map(lambda x,y: x[y], erm_pred_proba, erm_pred_labels)))
erm_pred_cor = np.array(erm_pred)
sort_scores = np.multiply(2*erm_pred_cor - 1, erm_pred_scores)

arg_inds = np.argsort(sort_scores)
seqpav_acc = seqpav_pred[arg_inds]
greedy_acc = greedy_pred[arg_inds]
erm_acc = np.zeros(len(erm_pred))
per_acc = np.zeros(len(per_pred))
for i in range(len(erm_pred)):
    erm_acc[i] = erm_pred[arg_inds[i]]
    per_acc[i] = per_pred[arg_inds[i]]

seqpav_plot = np.zeros(len(test_label))
greedy_plot = np.zeros(len(test_label))
erm_plot = np.zeros(len(test_label))
per_plot = np.zeros(len(test_label))
for i in range(len(test_label)):
    seqpav_plot[i] = sum(seqpav_acc[0:i+1])/(i+1)
    greedy_plot[i] = sum(greedy_acc[0:i+1])/(i+1)
    erm_plot[i] = sum(erm_acc[0:i+1])/(i+1)
    per_plot[i] = sum(per_acc[0:i+1])/(i+1)

opt_x = np.array([])
opt_plot = np.array([])
for i in range(len(test_label)):
    if (i > 0 and i % 400 == 0) or i == len(test_label)-1:
        inds = arg_inds[0:i+1]
        data = test_data_scaled[inds]
        label = test_label[inds]
        opt = Mymodel(random_state = 0, solver = 'newton-cg')
        opt.fit(data, label)
        opt_score = opt.score(data, label)
        #print(opt_score)
        opt_acc = ((i+1) / test_size) * opt_score * opt_score
        opt_plot = np.append(opt_plot, opt_acc)
        opt_x = np.append(opt_x, i)

        
x = list(range(len(test_label)))
plt.plot(x, erm_plot, color = 'red', linestyle = ':', label = "LR")
plt.plot(x, per_plot, color = 'brown', linestyle = 'dashdot', label = "AdaBoost")
plt.plot(x, greedy_plot, color = 'blue', linestyle = 'dashed', label = "Greedy")
plt.plot(x, seqpav_plot, color = 'green', linestyle = 'solid', label = "hPF")
plt.plot(opt_x ,opt_plot, color = 'magenta', linestyle = 'solid', marker = 'o', label = "PF Lower Bound")
plt.legend(loc="upper left")
plt.grid(axis = 'y')
plt.ylabel('Accuracy on the set of points {0,1,...,x}')
plt.xlabel('Points ordered by their LR scores.')
plt.show()

per_pred_labels = per.predict(test_data_scaled)
per_pred_labels = np.array(list(map(lambda x: int(x == '>50K'), per_pred_labels)))
per_pred_proba = np.array(per.predict_proba(test_data_scaled))
per_pred_scores = np.array(list(map(lambda x,y: x[y], per_pred_proba, per_pred_labels)))
per_pred_cor = np.array(per_pred)
sort_scores = np.multiply(2*per_pred_cor - 1, per_pred_scores)

arg_inds = np.argsort(sort_scores)
seqpav_acc = seqpav_pred[arg_inds]
greedy_acc = greedy_pred[arg_inds]
erm_acc = np.zeros(len(erm_pred))
per_acc = np.zeros(len(per_pred))
for i in range(len(erm_pred)):
    erm_acc[i] = erm_pred[arg_inds[i]]
    per_acc[i] = per_pred[arg_inds[i]]

seqpav_plot = np.zeros(len(test_label))
greedy_plot = np.zeros(len(test_label))
erm_plot = np.zeros(len(test_label))
per_plot = np.zeros(len(test_label))
for i in range(len(test_label)):
    seqpav_plot[i] = sum(seqpav_acc[0:i+1])/(i+1)
    greedy_plot[i] = sum(greedy_acc[0:i+1])/(i+1)
    erm_plot[i] = sum(erm_acc[0:i+1])/(i+1)
    per_plot[i] = sum(per_acc[0:i+1])/(i+1)

opt_x = np.array([])
opt_plot = np.array([])
for i in range(len(test_label)):
    if (i > 0 and i % 400 == 0) or i == len(test_label)-1:
        inds = arg_inds[0:i+1]
        data = test_data_scaled[inds]
        label = test_label[inds]
        opt = Mymodel(random_state = 0, solver = 'newton-cg')
        opt.fit(data, label)
        opt_score = opt.score(data, label)
        #print(opt_score)
        opt_acc = ((i+1) / test_size) * opt_score * opt_score
        opt_plot = np.append(opt_plot, opt_acc)
        opt_x = np.append(opt_x, i)

x = list(range(len(test_label)))
plt.plot(x, erm_plot, color = 'red', linestyle = ':', label = "LR")
plt.plot(x, per_plot, color = 'brown', linestyle = 'dashdot', label = "AdaBoost")
plt.plot(x, greedy_plot, color = 'blue', linestyle = 'dashed', label = "Greedy")
plt.plot(x, seqpav_plot, color = 'green', linestyle = 'solid', label = "hPF")
plt.plot(opt_x ,opt_plot, color = 'magenta', linestyle = 'solid', marker = 'o', label = "PF Lower Bound")
plt.legend(loc="upper left")
plt.grid(axis = 'y')
plt.ylabel('Accuracy on the set of points {0,1,...,x}')
plt.xlabel('Points ordered by their AdaBoost scores.')
plt.show()
