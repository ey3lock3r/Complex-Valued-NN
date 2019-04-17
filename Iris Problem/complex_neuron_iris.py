
import numpy as np
import copy

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
        print ('Target Angles: ',self.out)

    def map_out(self):
        sections = self.categories * self.periods
        angle    = 2 * np.pi / sections
        h_angle  = angle / 2
        # angle + π to get the oposite angle
        for i in range(1, self.categories+1):
            out_dict = {}
            o_angle = angle * i
            t_angle = o_angle - h_angle

            out_dict['angle']  = [o_angle]
            out_dict['target'] = [t_angle]
            if self.periods == 2:
                for j in range(1, self.periods):
                    out_dict['angle'].append(o_angle + np.pi)
                    out_dict['target'].append(t_angle + np.pi)

            self.out[i-1] = out_dict
            self.out[i-1] = out_dict
        
    def map_z(self, z):
        z = np.angle(z)
        while z < 0:
            z += 2 * np.pi

        #print ('Angle: ', z)
        for i in range(self.periods):
            for j in range(self.categories):
                if z < self.out[j]['angle'][i]:
                    return j, i

    def query(self, in_list, visual=False):
        in_arr = copy.deepcopy(in_list)
        in_arr = np.append(in_arr, 1.0)
        input = np.array(in_arr, ndmin=2, dtype='complex128')
        z = np.dot(input, self.w.T)[0]
        o, q = self.map_z(z)
        # print ('++++++++++++++++ Q U E R Y ++++++++++++++++')
        # print ('Input: ',input)
        # print ('Z: ',z)
        # print ('Output: ',o)
        return z, (o, q)
    
    def train(self, in_list, target, visual=False):
        in_arr = copy.deepcopy(in_list)
        in_arr = np.append(in_arr, 1.0)
        input = np.array(in_arr, ndmin=2, dtype='complex128')[0]
        z = np.dot(input, self.w.T)[0]
        o, q = self.map_z(z)
        # if o == target:
        #     self.tc_passed += 1
        #     return z
        t_angle = np.array(self.out[target]['target'])
        #t_angle_2complex = complex(np.cos(t_angle) + 1j*np.sin(t_angle))
        errors =  np.exp(1j * t_angle) - z
        #print ('Errors: ', errors)
        e = errors[np.argmin(np.abs(errors))]
        #mul = np.multiply.reduce([e, input], dtype='complex128')
        dw = e * np.conj(input) / (self.input_num + 1)
        self.w += dw
        
        #query after weight adjustment
        z1, o1 = self.query(in_list)
        #o1, _ = self.map_z(z1)
        if o1[0] == target:
            self.tc_passed += 1
        
        # print ('++++++++++++++++ T R A I N ++++++++++++++++')
        # print ('Input: ',input)
        # print ('Z: ',z)
        # print ('Output: ',o)
        # print ('Target Angles: ', t_angle)
        # print ('Error: ', e)
        # print ('Weights: ',self.w)
        return z, (o, q)
    
    def reset_tc_passed(self):
        self.tc_passed = 0

def angle2_cartesian(angle, q, offset=1):
    x = np.cos(angle)
    y = np.sin(angle)

    xtxt = 0
    ytxt = 0
    offset *= 0.25
    if q[1] == 1:       # bottom half of circle
        ytxt = y - offset
        if q[0] == 0:
            xtxt = x - offset
        else:
            xtxt = x + offset
    else:               # top half of circle
        ytxt = y + offset
        if q[0] == 1:
            xtxt = x - (offset + 0.25)
        else:
            xtxt = x + offset
    return (x,y), (xtxt, ytxt)
