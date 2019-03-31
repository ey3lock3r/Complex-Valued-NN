# simple neural network of one complex valued neuron
# extended to use a periodic activation function
# https://github.com/makeyourownneuralnetwork/complex_valued_neuralnetwork/blob/master/single_neuron-periodic-mnist.ipynb

import numpy
path = '/Users/IBM_ADMIN/Documents/Complex-Valued-NN/'
class neuralNetwork:
    
    def __init__(self, inputs, cats, periods):
        # number of inputs
        self.inputs = inputs
        
        # link weights matrix
        self.w = numpy.random.normal(0.0, pow(1.0, -0.5), (inputs + 1))
        self.w = numpy.array(self.w, ndmin=2, dtype='complex128')
        self.w += 1j * numpy.random.normal(0.0, pow(1.0, -0.5), (inputs + 1))
        
        # testing overrride
        #self.w = numpy.array([1.0 + 0.0j, 1.0 + 0.0j], ndmin=2, dtype='complex128')
        
        # number of output class categories
        self.categories = cats
        
        # todo periodicity
        self.periodicity = periods
        
        pass
    
    def z_to_class(self, z):
        # first work out the angle, but shift angle from [-pi/2, +pi.2] to [0,2pi]
        angle = numpy.mod(numpy.angle(z) + 2*numpy.pi, 2*numpy.pi)
        # from angle to category
        p = int(numpy.floor (self.categories * self.periodicity * angle / (2*numpy.pi)))
        p = numpy.mod(p, self.categories)
        return p

    def class_to_angles(self, c):
        # class to several angles due to periodicity, using bisector
        angles = (c + 0.5 + (self.categories * numpy.arange(self.periodicity))) / (self.categories * self.periodicity) * 2 * numpy.pi
        return angles
    
    def status(self):
        print ("w = ", self.w)
        print ("categories = ", self.categories)
        print ("periodicity = ", self.periodicity)
        pass

    def query(self, inputs_list):
        # add bias input
        inputs_list.append(1.0)
        
        # convert input to complex
        inputs = numpy.array(inputs_list, ndmin=2, dtype='complex128').T
        #print("inputs = \n", inputs)
        
        # combine inputs, weighted
        z = numpy.dot(self.w, inputs)
        #print("z = ", z)
        
        # map to output classes
        o = self.z_to_class(z)
        #print("output = ", o)
        #print ("")
        return o
    
    def train(self, inputs_list, target):
        # add bias input
        inputs_list.append(1.0)
        
        # convert inputs and outputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2, dtype='complex128').T

        # combine inputs, weighted
        z = numpy.dot(self.w, inputs)[0]
        
        # desired angle from trainging set
        # first get all possible angles
        desired_angles = self.class_to_angles(target)
        
        # potential errors errors
        errors =  numpy.exp(1j*desired_angles) - z
        # select smallest error
        e = errors[numpy.argmin(numpy.abs(errors))]
        
        # dw = e * x.T / (x.x.T)
        dw = (e * numpy.conj(inputs.T)) / (self.inputs + 1)
        #print("dw = ", dw)
        self.w += dw
        #print("new self.w = ", self.w )
        #print("test new self.w with query = ", self.query(inputs.T))
        #print("--")
    pass

# create instance of neural network
number_of_inputs = 784
categories = 10
periods = 3
n = neuralNetwork(number_of_inputs, categories, periods)

# load the mnist training data CSV file into a list
training_data_file = open(path + "mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# epochs is the number of times the training data set is used for training
epochs = 5
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        inputs = inputs.tolist()
        n.train(inputs, int(all_values[0]))
        pass
    pass

# load the mnist test data CSV file into a list
test_data_file = open(path + "mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []
# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    inputs = inputs.tolist()
    # query the network
    label = n.query(inputs)

    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)