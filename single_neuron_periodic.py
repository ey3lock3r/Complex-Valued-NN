# simple neural network of one complex valued neuron
# extended to use a periodic activation function
# https://github.com/makeyourownneuralnetwork/complex_valued_neuralnetwork/blob/master/single_neuron-periodic.ipynb
# http://makeyourownneuralnetwork.blogspot.com/2016/05/complex-valued-neural-networks.html
import numpy
import matplotlib.pyplot as plt

# initialize visualization
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='datalim', anchor='C')
#ax.set_xlim((-2,2))
#ax.set_ylim((-2,2))
circ = plt.Circle((0, 0), 1, color='#00ffff', alpha=1, fill=False)
ax.add_patch(circ)

def visualize(angle, txt='', color='black', sign='+'):
    x = numpy.cos(angle)
    y = numpy.sin(angle)
    ax.scatter(x,y)
    if sign == '+':
        plt.annotate(txt, xy=(x, y), xytext=(x+0.25, y+0.25),
            arrowprops=dict(facecolor=color, shrink=0.05),)
    else:
        plt.annotate(txt, xy=(x, y), xytext=(x-0.25, y-0.25),
            arrowprops=dict(facecolor=color, shrink=0.05),)

class neuralNetwork:
    
    def __init__(self, inputs, cats, periods):
        # link weights matrix
        self.w = numpy.random.normal(0.0, pow(1.0, -0.5), (inputs + 1))
        #print ('1st Weight transform: ',self.w)
        self.w = numpy.array(self.w, ndmin=2, dtype='complex128')
        #print ('2nd Weight transform: ',self.w)
        self.w += 1j * numpy.random.normal(0.0, pow(1.0, -0.5), (inputs + 1))
        #print ('3rd Weight transform: ',self.w)
        
        # testing overrride
        #self.w = numpy.array([1.0 + 0.0j, 1.0 + 0.0j], ndmin=2, dtype='complex128')
        
        # number of output class categories
        self.categories = cats
        
        # todo periodicity
        self.periodicity = periods

        self.tc_passed = 0
        
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
        visualize(numpy.angle(z), 'Learned', 'green')

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
        visualize(numpy.angle(z), 'Raw In', 'red')
        
        # desired angle from trainging set
        # first get all possible angles
        desired_angles = self.class_to_angles(target)
        for i in desired_angles:
            visualize(i, 'Target', 'yellow', '-')

        print ('Angles:',desired_angles)
        print ('Z: ', z)
        # potential errors errors
        errors =  numpy.exp(1j*desired_angles) - z
        print ('Errors: ',errors)
        # select smallest error
        e = errors[numpy.argmin(numpy.abs(errors))]
        print ('Error: ',e)
        
        # dw = e * x.T / (x.x.T)
        dw = e * numpy.conj(inputs.T) / 3
        print("dw = ", dw)
        self.w += dw
        print("new self.w = ", self.w )
        learned = self.query(inputs.T, train=True)
        print("test new self.w with query = ", learned)
        print("--")
        if learned == target:
            self.tc_passed += 1
    pass

    def reset_tc_passed(self):
        self.tc_passed = 0

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
    print ('Iter {}: Test passed: {}'.format(i, n.tc_passed))
    if n.tc_passed == 4:
        break
    
    n.reset_tc_passed()

# query after training
print('After Training')
n.query( [-1.0, -1.0] )
n.query( [-1.0, 1.0] )
n.query( [1.0, -1.0] )
n.query( [1.0, 1.0] )
#plt.show()