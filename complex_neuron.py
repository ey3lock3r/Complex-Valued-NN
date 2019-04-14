import matplotlib.pyplot as plt
import numpy as np
import math

# initialize visualization
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='datalim', anchor='C')
#ax.set_xlim((-2,2))
#ax.set_ylim((-2,2))
circ = plt.Circle((0, 0), 1, color='#00ffff', alpha=1, fill=False)
ax.add_patch(circ)

class ComplexNeuron():
    def __init__(self, inputn, cat, periods):
        self.input_num = inputn
        self.categories = cat
        self.periods = periods
        self.tc_passed = 0

        # link weights matrix
        self.w = np.random.normal(0.0, 1.0, (inputn + 1))
        self.w = np.array(self.w, ndmin=2, dtype='complex128')
        self.w += 1j * np.random.normal(0.0, 1.0, (inputn + 1))
        print ('Weights: ', self.w)
        self.out = {}
        self.map_out()
        print ('Out Mapping: ',self.out)

    def map_out(self):
        sections = self.categories * self.periods
        angle    = 2 * np.pi / sections
        h_angle  = angle / 2
        # angle + π to get the oposite angle
        for i in range(1, self.categories+1):
            out_dict = {}
            o_angle = angle * i
            t_angle = o_angle - h_angle
            # o_angle_i = o_angle + np.pi
            # t_angle_i = t_angle + np.pi
            out_dict['angle']  = [o_angle]
            out_dict['target'] = [t_angle]
            if self.periods == 2:
                for j in range(1, self.periods):
                    out_dict['angle'].append(o_angle + np.pi)
                    out_dict['target'].append(t_angle + np.pi)

                    # Plot lines
                    x = np.cos(o_angle + np.pi)
                    y = np.sin(o_angle + np.pi)
                    ax.plot([0,x], [0,y], lw=2, marker='o', markevery=[1])

            # out_dict['angle']  = [o_angle, o_angle_i]
            # out_dict['target'] = [t_angle, t_angle_i]
            self.out[i-1] = out_dict
            self.out[i-1] = out_dict
            #print ('Target Angle {}: {}, {}'.format(i-1, *self.out[i-1]['target']))

            # Plot lines
            x = np.cos(o_angle)
            y = np.sin(o_angle)
            ax.plot([0,x], [0,y], lw=2, marker='o', markevery=[1])
        
    def map_z(self, z):
        z = np.angle(z)
        while z < 0:
            z += 2 * np.pi

        print ('Angle: ', z)
        for i in range(self.periods):
            for j in range(self.categories):
                if z < self.out[j]['angle'][i]:
            # for j in self.out[i]['angle']:
            #     if z < j:
                    return j

    def query(self, in_list, visual=False):
        print ('++++++++++++++++ Q U E R Y ++++++++++++++++')
        in_list.append(1.0)
        input = np.array(in_list, ndmin=2, dtype='complex128')
        print ('Input: ',input)
        z = np.dot(input, self.w.T)[0]
        print ('Z: ',z)
        o = self.map_z(z)
        print ('Output: ',o)
        return z
    
    def train(self, in_list, target, visual=False):
        print ('++++++++++++++++ T R A I N ++++++++++++++++')
        in_list.append(1.0)
        input = np.array(in_list, ndmin=2, dtype='complex128')
        print ('Input: ',input)
        z = np.dot(input, self.w.T)[0]
        print ('Z: ',z)
        o = self.map_z(z)
        print ('Output: ',o)
        # if o == target:
        #     self.tc_passed += 1
        #     return z

        print ('Modify weights!')
        t_angle = np.array(self.out[target]['target'])
        print ('Target Angles: ', t_angle)
        #t_angle_2complex = complex(np.cos(t_angle) + 1j*np.sin(t_angle))

        errors =  np.exp(1j * t_angle) - z
        #print ('Errors: ', errors)
        e = errors[np.argmin(np.abs(errors))]
        print ('Error: ', e)
        dw = e * input / 3
        print ('DW: ',dw)
        self.w += dw
        
        #query after weight adjustment
        z1 = self.query(in_list[:2])
        o = self.map_z(z1)
        if o == target:
            self.tc_passed += 1
        
        return z
    
    def reset_tc_passed(self):
        self.tc_passed = 0

def visualize(angle, txt='', color='black', sign='*'):
    x = np.cos(angle)
    y = np.sin(angle)
    ax.scatter(x,y)
    if sign == '+':
        plt.annotate(txt, xy=(x, y), xytext=(x+0.25, y+0.25),
            arrowprops=dict(facecolor=color, shrink=0.05),)
    elif sign == '-':
        plt.annotate(txt, xy=(x, y), xytext=(x-0.25, y-0.25),
            arrowprops=dict(facecolor=color, shrink=0.05),)
    else:
        plt.annotate(txt, xy=(x, y), xytext=(x-0.75, y+0.25),
            arrowprops=dict(facecolor=color, shrink=0.05),)

n_in = 2
cat  = 2
per  = 2
iter = 100

nn = ComplexNeuron(n_in, cat, per)
# Train Neural Network
for i in range(iter):
    nn.train([-1.0, -1.0], 0)
    nn.train([-1.0, 1.0], 1)
    nn.train([1.0, -1.0], 1)
    nn.train([1.0, 1.0], 0)
    print ('Iter {}: Test passed: {}'.format(i, nn.tc_passed))
    if nn.tc_passed == 4:
        print ('Complex network successfully learned XOR')
        break
    
    nn.reset_tc_passed()

o = nn.train([-1,-1],0)
visualize(np.angle(o), 'Raw Angle', color='gray',sign='+')

# Sample Test
o = nn.query([-1, -1])
# o = nn.query([-1, 1])
# o = nn.query([1, -1])
# o = nn.query([1, 1])
# visualize(np.angle(o), 'Learned Angle', color='gold',sign='+')



# Sample Target
ang0 = nn.out[0]['target']
for i in ang0:
    visualize(i, 'Target Angle 0', color='green', sign='-')

ang1 = nn.out[1]['target']
for i in ang1:
    visualize(i, 'Target Angle 1', color='red')

plt.show()