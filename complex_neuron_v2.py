# simple neuron using complex numbers for weights and input
# to learn the XOR problem, must use 2 periods, ie, dividing the circle into 2 sections so opposite angles have the same category
# The ideas are based from the link below, which I modified based on my understanding of how the neuron behaves
# I've extended this to include visualization of the neural activity
# https://github.com/makeyourownneuralnetwork/complex_valued_neuralnetwork/blob/master/single_neuron-periodic.ipynb
# http://makeyourownneuralnetwork.blogspot.com/2016/05/complex-valued-neural-networks.html
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import copy

# initialize visualization
fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect="equal", adjustable='datalim', anchor='C'))
# ax.set_xlim((-4,4))
# ax.set_ylim((-4,4))
fig.set_dpi(100)
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

            out_dict['angle']  = [o_angle]
            out_dict['target'] = [t_angle]
            if self.periods == 2:
                for j in range(1, self.periods):
                    out_dict['angle'].append(o_angle + np.pi)
                    out_dict['target'].append(t_angle + np.pi)

            self.out[i-1] = out_dict
            self.out[i-1] = out_dict

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
        for i in range(sections):
            patches[i-1].set_alpha(0.7)
            if i%2 == 0:
                autotexts[i].set_text('0')
                patches[i].set_color('#0cff0c')
                legnds.legendHandles[i].set_color('#0cff0c')
            else:
                patches[i].set_color('#0165fc')
                legnds.legendHandles[i].set_color('#0165fc')

        ax.set_title("Single Neuron:\nSolving XOR Problem")
        plt.setp(autotexts, size=8, weight="bold")
                        
    def map_z(self, z):
        z = np.angle(z)
        while z < 0:
            z += 2 * np.pi

        print ('Angle: ', z)
        for i in range(self.periods):
            for j in range(self.categories):
                if z < self.out[j]['angle'][i]:
                    return j, i

    def query(self, in_list, visual=False):
        print ('++++++++++++++++ Q U E R Y ++++++++++++++++')
        in_arr = copy.deepcopy(in_list)
        in_arr.append(1.0)
        input = np.array(in_arr, ndmin=2, dtype='complex128')
        print ('Input: ',input)
        z = np.dot(input, self.w.T)[0]
        print ('Z: ',z)
        o, q = self.map_z(z)
        print ('Output: ',o)
        return z, (o, q)
    
    def train(self, in_list, target, visual=False):
        print ('++++++++++++++++ T R A I N ++++++++++++++++')
        in_arr = copy.deepcopy(in_list)
        in_arr.append(1.0)
        input = np.array(in_arr, ndmin=2, dtype='complex128')
        print ('Input: ',input)
        z = np.dot(input, self.w.T)[0]
        print ('Z: ',z)
        o, q = self.map_z(z)
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
        self.w += dw
        print ('Weights: ',self.w)
        
        #query after weight adjustment
        # z1, _ = self.query(in_list[:2])
        # o1, _ = self.map_z(z1)
        # if o1 == target:
        #     self.tc_passed += 1
        
        return z, (o, q)
    
    def reset_tc_passed(self):
        self.tc_passed = 0

def visualize(angle, q, offset=1):
    x = np.cos(angle)
    y = np.sin(angle)
    #ax.scatter(x,y, facecolor='red')
    # xtxt = -1.7
    # ytxt = 1.2 * offset
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

# Initialize neural network
n_in = 2
cat  = 2
per  = 2
nn = ComplexNeuron(n_in, cat, per)

# define arrow annotations for raw and learned angle
arrowprops = dict(arrowstyle="-|>",
    color='black',
    shrinkA=5, shrinkB=5,
    patchA=None,
    patchB=None,
    connectionstyle="angle,angleA=-90,angleB=180,rad=5",
    )
raw_angle = plt.annotate('', xy=(0,0))
learned_angle = plt.annotate('', xy=(0,0))

# define Input Texts and Input Data
bbox = dict(boxstyle="square", fc='w', ec='black')
txt_handler = [
    plt.text(-2, 0.6, "Training Data:", weight="semibold", size='large'),
    plt.text(-1.8, 0.35, '', bbox=bbox, weight="semibold", size='xx-large', family='monospace'),
    plt.text(-1.8, 0, '', bbox=bbox, weight="semibold", size='xx-large', family='monospace'),
    plt.text(-2, -0.4, "Target Output:", weight="semibold", size='large'),
    plt.text(-1.8, -0.65, '', bbox=bbox, weight="semibold", size='xx-large', family='monospace')]

Train_d = [
    ['Train',[-1, -1], 0],
    ['Train',[-1,  1], 1],
    ['Train',[ 1, -1], 1],
    ['Train',[ 1,  1], 0],
    ['Query',[-1, -1]],
    ['Query',[-1,  1]],
    ['Query',[ 1, -1]],
    ['Query',[ 1,  1]]
]
cnt = 0
t_len = len(Train_d)
test = False
learned_point, = ax.plot([], [], 'ro')
raw_point,     = ax.plot([], [], 'ro')
def updatefig(data):
    global txt_handler, nn, Train_d, cnt, t_len, raw_angle, learned_angle, test, arrowprops, learned_point, raw_point
    bbox = dict(boxstyle="square", fc='w', ec='black')
    bbox_r = dict(boxstyle="square", fc='red', ec='black')
    bbox_g = dict(boxstyle="square", fc='green', ec='black')

    if Train_d[cnt][0] == 'Train':
        if test:
            print ('Data: ', Train_d[cnt][1])
            z, q = nn.query(Train_d[cnt][1])
            xy, xytxt = visualize(np.angle(z), q)
            # set and show learned angle and point
            learned_point.set_data(xy)
            learned_point.set_visible(True)
            learned_angle = plt.annotate('Learned Angle', xy=xy, xytext=xytxt,
                bbox=dict(boxstyle="round", fc='gold'), arrowprops=arrowprops)

            txt_handler[4].set_text(q[0])
            if q[0] != Train_d[cnt][2]:
                txt_handler[4].set_bbox(bbox_r)
            else:
                txt_handler[4].set_bbox(bbox_g)

            test = False

            cnt += 1
            if cnt == t_len:
                cnt = 0

            return [*txt_handler, learned_angle , raw_angle, learned_point, raw_point]
        else:
            z, q = nn.train(Train_d[cnt][1], Train_d[cnt][2])
            xy, xytxt = visualize(np.angle(z), q, -1.2)
            # set raw angle and point
            raw_point.set_data(xy)
            raw_point.set_visible(True)
            raw_angle = plt.annotate('Raw Angle', xy=xy, xytext=xytxt,
                bbox=dict(boxstyle="round", fc='gray'), arrowprops=arrowprops)
            
            # set input texts
            txt_handler[0].set_text('Training Data')
            txt_handler[3].set_text('Target Output')
            txt_handler[1].set_text(Train_d[cnt][1][0])
            txt_handler[2].set_text(Train_d[cnt][1][1])
            txt_handler[4].set_text(q[0])
            if q[0] != Train_d[cnt][2]:
                txt_handler[4].set_bbox(bbox_r)
            else:
                txt_handler[4].set_bbox(bbox_g)

            # hide learned angle and point
            learned_angle.remove()
            learned_point.set_visible(False)
            test = True

            return [*txt_handler, raw_angle, learned_point, raw_point]
    else:
        z, q = nn.query(Train_d[cnt][1])
        xy, xytxt = visualize(np.angle(z), q)
        # set learned angle
        learned_point.set_data(xy)
        learned_point.set_visible(True)
        learned_angle = plt.annotate('Learned Angle', xy=xy, xytext=xytxt,
            bbox=dict(boxstyle="round", fc='gold'), arrowprops=arrowprops)

        # set input texts
        txt_handler[0].set_text('After Training')
        txt_handler[3].set_text('Learned Output')
        txt_handler[1].set_text(Train_d[cnt][1][0])
        txt_handler[2].set_text(Train_d[cnt][1][1])
        txt_handler[4].set_text(q[0])
        txt_handler[4].set_bbox(bbox)
        
        # hide raw angle
        raw_angle.set_visible(False)
        raw_point.set_visible(False)

        cnt += 1
        if cnt == t_len:
            cnt = 0
        
        return [*txt_handler, learned_angle, learned_point, raw_point]

ani = animation.FuncAnimation(fig, updatefig, interval=5000, blit=True, repeat=False)
#ani.save("single_neuron.mp4")
plt.show()