import matplotlib.pyplot as plt
import complex_neuron_iris as c_nn 
import numpy as np
import math

path = "/Users/IBM_ADMIN/Documents/Complex-Valued-NN/iris_dataset/"
# initialize visualization
fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect="equal", adjustable='datalim', anchor='C'))
# ax.set_xlim((-4,4))
# ax.set_ylim((-4,4))
fig.set_dpi(100)

def create_pie(labels, tot_lab):
    # create Pie to represent neuron
    data = np.ones(4)
    labels = ['False', 'True'] * 2
    patches, texts, autotexts = ax.pie(data, autopct=lambda pct: int(pct * sum(data)/100), 
        labels=labels, textprops=dict(color="w"))
    
    # set Pie legends Title and loc
    ax.legend(patches, labels,
        title="Truth Values",
        loc="center left",
        bbox_to_anchor=(0.8, 0, 0.5, 1))

    # Set Legends color and Texts
    legnds = ax.get_legend()
    for i in range(tot_lab):
        patches[i-1].set_alpha(0.7)
        if i%2 == 0:
            autotexts[i].set_text('0')
            patches[i].set_color('#0cff0c')
            legnds.legendHandles[i].set_color('#0cff0c')
        else:
            patches[i].set_color('#0165fc')
            legnds.legendHandles[i].set_color('#0165fc')

    ax.set_title("Single Neuron:\nIris Dataset Classification Problem")
    plt.setp(autotexts, size=8, weight="bold")

# Load Iris Dataset
# converters = {
#     4: lambda s: 0 if s == 'Iris-setosa' else (1 if s == 'Iris-versicolor' else 2)
# }
dataset = np.loadtxt(fname=path + "iris_m.csv", dtype=float, delimiter=",")
print(dataset.shape)
np.random.shuffle(dataset)
train_set = []
test_set  = []
num_test  = int(np.ceil(len(dataset)/4))
print (num_test)
test_n = int(np.floor(num_test/3))

for i in range(3):
    temp = np.array(list(filter(lambda m: m[4] == i, dataset)))
    ts_set = temp[:test_n,:]
    tr_set = temp[test_n:,:]
    train_set.extend(tr_set)
    test_set.extend(ts_set)

test_set = np.array(test_set)
train_set = np.array(train_set)

minx = np.min(test_set)
maxx = np.max(test_set)
test_set = (test_set - minx) / (maxx - minx)

minx = np.min(train_set)
maxx = np.max(train_set)
train_set = (train_set - minx) / (maxx - minx)

print ('Train Set: ', len(train_set))
print ('Test Set: ', len(test_set))
# Initialize neural network
n_in = 4 # Sepal Length | Sepal Width | Petal Length | Petal Width
cat  = 3 # Iris Sentosa | Iris Versicolour | Iris Virginica
per  = 2
nn = c_nn.ComplexNeuron(n_in, cat, per)

# Train Neural Network
iter = 100
for i in range(iter):
    for train in train_set:
        nn.train(train[:n_in], int(train[n_in:]))

    print ('Iter {}: Test passed: {}'.format(i, nn.tc_passed))
    if nn.tc_passed == len(train_set):
        print ('Complex network successfully learned Iris Dataset')
        break
    nn.reset_tc_passed()

# Sample Test
num_tc = 0
for test in test_set:
    z, q = nn.query(test[:n_in])
    if q[0] == int(test[n_in:]):
        num_tc += 1

ts_len = len(test_set)
print ('{} out of {} successfully identified or {}% accuracy'.format(num_tc, ts_len, num_tc/ts_len * 100))



# xy, xytxt = c_nn.angle2_cartesian(np.angle(z), q)
# # raw_angle = plt.annotate('Raw Angle', xy=xy, xytext=xytxt,
# #         bbox=dict(boxstyle="round", fc='gray'), arrowprops=arrowprops)
# learned_angle = plt.annotate('Learned Angle', xy=xy, xytext=xytxt,
#         bbox=dict(boxstyle="round", fc='gold'), arrowprops=arrowprops)

# learned_point, = ax.plot(*xy, 'ro')
# # raw_point,     = ax.plot([], [], 'ro')



plt.show()
