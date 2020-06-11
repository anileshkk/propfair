import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as Mymodel
from sklearn.ensemble import AdaBoostClassifier as Mymodel1
from sklearn.preprocessing import minmax_scale
import time
import preprocess as ap

FRACTION = 0.25

#preprocess train and test data to get data without sensitive labels
train_data, train_label, test_data, test_label = ap.compas_preprocess('compas-scores-two-years.csv', True)

total_size = len(train_label)
total_size_test = len(test_label)

train_data_no_sensitive = train_data.drop(columns = ['race', 'sex'])
test_data_no_sensitive = test_data.drop(columns = ['race', 'sex'])

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
hack_data = remaining_data[0:2] #see hack_label below

remaining_label = train_label
hack_label = remaining_label[0:2] #SGD blackbox trainer always needs labels of >=2 classes
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
num_iter = 500
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
    seqpav.append(Mymodel(random_state = 0, solver = 'newton-cg', max_iter = 2000))
    #seqpav.append(Mymodel1(random_state = 0))
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
per_order = False
per_order = True
if per_order:
    erm_preds = per_preds
    
agree_inds = np.equal(erm_preds, test_label)
correct_data = test_data_scaled[agree_inds]
correct_label = test_label[agree_inds].to_numpy()
correct_inds = list(range(len(correct_label)))
disagree_inds = np.not_equal(erm_preds, test_label)
wrong_data = test_data_scaled[disagree_inds]
wrong_label = test_label[disagree_inds].to_numpy()
wrong_inds = list(range(len(wrong_label)))

erm_accs = []
seqpav_accs = []
greedy_accs = []
opt_accs = []
per_accs = []
test_size = len(test_label)
correct_size = len(correct_label)
wrong_size = len(wrong_label)
l = int(0.25 * test_size)

for t in range(11):
    i = t + 0
    print("i = %d" %i)
    samples_wrong_0 = l - int((i / 10) * l)
    samples_correct_0 = int((i / 10) * l)
    samples_correct, samples_wrong = ap.sample_sizes(l, correct_size, wrong_size, samples_correct_0, samples_wrong_0)
    erm_acc = 0
    seqpav_acc = 0
    greedy_acc = 0
    opt_acc = 0
    per_acc = 0
    for j in range(10):
        print("j = %d" %j) 
        inds = correct_inds.copy()
        np.random.shuffle(inds)
        correct_sample_inds = inds[0:samples_correct]
        correct_sample_data = correct_data[correct_sample_inds]
        correct_sample_label = correct_label[correct_sample_inds]
        inds = wrong_inds.copy()
        np.random.shuffle(inds)
        wrong_sample_inds = inds[0:samples_wrong]
        wrong_sample_data = wrong_data[wrong_sample_inds]
        wrong_sample_label = wrong_label[wrong_sample_inds]
        data = np.append(correct_sample_data, wrong_sample_data, axis = 0)
        label = np.append(correct_sample_label, wrong_sample_label, axis = 0)
        opt = Mymodel(random_state = 0, solver = 'newton-cg')
        opt.fit(data, label)
        opt_score = opt.score(data, label)
        opt_acc = opt_acc + (l / test_size) * opt_score * opt_score
        erm_acc = erm_acc + erm.score(data, label)
        per_acc = per_acc + per.score(data, label)
        seqpav_acc = seqpav_acc + ap.accuracy_randomized(seqpav, seqpav_weights, data, label)
        greedy_acc = greedy_acc + ap.accuracy_randomized(classifiers, weights, data, label)
    erm_acc = erm_acc / (j+1)
    seqpav_acc = seqpav_acc / (j+1)
    greedy_acc = greedy_acc / (j+1)
    opt_acc = opt_acc / (j+1)
    per_acc = per_acc / (j+1)
    erm_accs.append(erm_acc)
    seqpav_accs.append(seqpav_acc)
    greedy_accs.append(greedy_acc)
    opt_accs.append(opt_acc)
    per_accs.append(per_acc)
    
x = 0.1 * np.arange(0, 11, 1)
plt.plot(x, erm_accs, color = 'red', linestyle = 'solid', marker = 'o', label = "LR")
plt.plot(x, per_accs, color = 'brown', linestyle = ':', marker = 'D', label = "AdaBoost")
plt.plot(x, greedy_accs, color = 'blue', linestyle = 'dashed', marker = '^', label = "Greedy")
plt.plot(x, seqpav_accs, color = 'green', linestyle = 'dashdot', marker = 's', label = "hPF")
plt.plot(x ,opt_accs, color = 'magenta', linestyle = 'dotted', marker = '*', label = "PF Lower Bound")
plt.legend(loc="upper left")
plt.grid(axis = 'y')
plt.ylabel('Accuracy')
plt.xlabel('Fraction correctly classified by AdaBoost')
plt.title('Size of subsampled test set = 0.75 of actual test set.')
plt.show()
