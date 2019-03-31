# simple neural network of one complex valued neuron
# extended to use a periodic activation function
# https://github.com/makeyourownneuralnetwork/complex_valued_neuralnetwork/blob/master/single_neuron-periodic.ipynb
# http://makeyourownneuralnetwork.blogspot.com/2016/05/complex-valued-neural-networks.html

import numpy
class neuralNetwork:
    
    def __init__(self, inputs, cats, periods):
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

    def query(self, inputs_list, train=False):
        # add bias input
        if not train:
            inputs_list.append(1.0)
        
        # convert input to complex
        inputs = numpy.array(inputs_list, ndmin=2, dtype='complex128').T
        print("inputs = \n", inputs)
        
        # combine inputs, weighted
        z = numpy.dot(self.w, inputs)
        print("z = ", z)
        
        # map to output classes
        o = self.z_to_class(z)
        print("output = ", o)
        print ("")
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
        dw = (e * numpy.conj(inputs.T)) / (3)
        print("dw = ", dw)
        self.w += dw
        print("new self.w = ", self.w )
        print("test new self.w with query = ", self.query(inputs.T, train=True))
        print("--")
    pass

# create instance of neural network
number_of_inputs = 2
categories = 2
periods = 2

n = neuralNetwork(number_of_inputs, categories, periods)
n.status()

# train neural network - XOR
for i in range(10):
    n.train([-1.0, -1.0], 0)
    n.train([-1.0, 1.0], 1)
    n.train([1.0, -1.0], 1)
    n.train([1.0, 1.0], 0)
    pass

# query after training
n.query( [-1.0, -1.0] )
n.query( [-1.0, 1.0] )
n.query( [1.0, -1.0] )
n.query( [1.0, 1.0] )