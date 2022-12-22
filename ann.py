'''
Author: J. Rafid Siddiqui
Azad-Academy
jrs@azaditech.com
https://www.azaditech.com

'''
#==================================================================

from utils import *
import random

class ANN:
    
    inputs = None
    params = None
    outputs = None
    activations = None
    num_params = 13
    num_layers = 2
    Y = np.array([[1]])
    Y_hat = np.array([[1]])
    grad = None
    
    def __init__(self): 
        self.inputs = np.array([[1,0.5,0.5]])
        self.params = np.random.rand(self.num_params)
        self.outputs = [None]*self.num_layers
        self.activations = [None]*self.num_layers
        self.grad = np.random.rand(self.num_params)
    
    def cost(self,nnparams,L):
    
        #W1,W2 = weights2matrices(nnparams)
        self.params = nnparams
        J = self.feed_forward_step(L)
        
        return J

    def Gradient(self,nnparams,L):

        #W1,W2 = weights2matrices(nnparams)
        self.params = nnparams
        W1_grad, W2_grad, delta = self.back_prop_step(L)
        
        #W1 = (W1-0.85*W1_grad).flatten()
        #W2 = (W2-0.85*W2_grad).flatten()
        
        #self.params = np.concatenate((W1,W2))
            
        G = np.concatenate((W1_grad.flatten(),W2_grad.flatten()))

        self.grad = G
        
        return G

    def feed_forward_step(self, L=1):
        
        X = self.inputs
        Y = self.Y
        W1,W2 = weights2matrices(self.params)

        z1 =  X @ np.transpose(W1)
        a1 = sigmoid(z1)
        a11 = np.concatenate((np.ones((a1.shape[0],1)),a1),axis=1)
        z2 = a11 @ np.transpose(W2)
        a2 = sigmoid(z2)

        m = X.shape[0]

        Jreg = (L/(2*m))*(np.sum(np.sum(W1[:,2:]**2,axis=1)) + np.sum(np.sum(W2[:,2:]**2,axis=1)))
        J = (-1/m*np.sum(np.sum(Y*np.log(a2)+(1-Y)*np.log(1-a2),axis=1))) + Jreg
        
        self.outputs = [z1,z2]
        self.activations = [a1,a2]
        self.Y_hat = a2

        return J            
            

    def back_prop_step(self,L=1):

        X = self.inputs
        Y = self.Y
        W1,W2 = weights2matrices(self.params)

        a1 = self.activations[0]
        a2 = self.activations[1]
        z1 = self.outputs[0]
        z2 = self.outputs[1]

        delta2 = a2 - Y
        #delta1 = (delta2 @ W2)*sigmoid_grad(z2)
        delta1 = (delta2 @ W2[:,1:])*sigmoid_grad(z1)
        
        Delta1 = np.zeros(W1.shape)
        Delta2 = np.zeros(W2.shape)
        
        a11 = np.concatenate((np.ones((a1.shape[0],1)),a1),axis=1)
        
        m = X.shape[0]
        for i in range(0,m):
            
            Delta2 = Delta2 + np.transpose(delta2[i,:]).reshape(1,1) @ a11[i,:].reshape(1,4)
            Delta1 = Delta1 + np.transpose(delta1[i,:]).reshape(3,1) @ X[i,:].reshape(1,3)
        
        W1_grad = 1/m * (Delta1 + L*np.concatenate((np.zeros((W1.shape[0],1)),W1[:,1:]),axis=1))
        W2_grad = 1/m * (Delta2 + L*np.concatenate((np.zeros((W2.shape[0],1)),W2[:,1:]),axis=1))
        

        return W1_grad,W2_grad,[delta1,delta2]
    
    def predict(self,X):

        num_pts = X.shape[0]
        Y = np.zeros(num_pts)
        for i in range(0,num_pts):
        
            self.inputs = np.array([X[i,:]])
            self.feed_forward_step()
            Y[i] = self.activations[1]
            set_values(self.inputs[0,1],self.inputs[0,2],self.Y[0,0],self.outputs[0],self.outputs[1],self.activations[0],self.activations[1])

        return Y



#============================================================================================
# GUI Controls and Display
#============================================================================================
axes_annotations = []

xy_coords1 = [(240,410),(240,620),(755,235),(900,235),(777,535),(918,535),(785,850),(930,850),(1310,540),(1690,540),(1700,710),(1750,70),(1610,290),(360,100) ] #pixel coordinates of neurons + backprops
xy_coords2 = [(705,85),(520,300),(585,390),(730,400),(600,505),(555,600),(750,705),(630,790),(555,840),(1325,395),(1210,480),(1140,563),(1227,655)] #coordinates of weights
wcolors = ['#7F0000','#7F0000','#7F0000','#007F7F','#007F7F','#007F7F','#658D00','#658D00','#658D00','#404040','#404040','#404040','#404040','#404040','#404040']
wrotations=[0,25,45,0,-20,15,0,-45,-35,0,-55,0,0,0,50]
values = np.zeros(len(xy_coords1))

ann = ANN()  


fig = None
ax = None

#==================================================================================================

def set_values(x1,x2,y,z1,z2,a1,a2,delta=None):
    
    global values
    
    values[0] = x1
    values[1] = x2
    values[10] = y
    
    values[9] = a2    # Y_hat value
    values[8] = z2
    values[2] = z1[:,0]
    values[3] = a1[:,0]
    values[4] = z1[:,1]
    values[5] = a1[:,1]
    values[6] = z1[:,2]
    values[7] = a1[:,2]
    
    if(delta is not None):
        values[11] = delta[0]
        values[12] = delta[1]
        values[13] = delta[2]
    
    

    
def get_values():    
        
    X = ann.inputs
    Y = ann.Y
    
    z1 = ann.outputs[0]
    a1 = ann.activations[0]
    z2 = ann.outputs[1]
    a2 = ann.activations[1]
        
    return X,Y,z1,z2,a1,a2


feed_forward_btn = widgets.Button(
    value=True,
    description='Feed Forward',
    disabled=True,
    button_style='',
    tooltip='Press to run a feed-forward step on the input values',
    )

feed_forward_btn.style.button_color = '#90ee90'

back_prop_btn = widgets.Button(
    value=True,
    description='BackPropagation',
    disabled=True,
    button_style='',
    tooltip='Press to run a Backpropagation step on the input values',
    )
back_prop_btn.style.button_color = '#066ff9'

init_btn = widgets.Button(
    value=True,
    description='Initialize',
    disabled=False,
    tooltip='Press to Initialize with Random weights',
    #icon='fa-step-forward'
    )
init_btn.style.button_color = '#066ff9'

reset_btn = widgets.Button(
    value=True,
    description='Reset',
    disabled=True,
    tooltip='Press to reset and re-initialize weights',
    )
reset_btn.style.button_color = '#c05454'


input1 = widgets.FloatSlider(
    name="X1:",
    value=0.5,
    min=-4,
    max=4,
    step=0.01,
    bar_color='red'
)

input2 = widgets.FloatSlider(
    name="X2",
    value=0.5,
    min=-4,
    max=4,
    step=0.01,
    bar_color='green'
)

class_label = widgets.Dropdown(
    options=[('0', 0), ('1', 1)],
    value=1,
    description='Class Label:',
    width='30%'
)

def initialize_params(nnparams):
    
    global axes_annotations,values
    
    axes_annotations.clear()
    values = np.zeros(len(xy_coords1))
    values[0]=values[1]=0.5
    values[10]=1

    for i in range(0,len(xy_coords1)):
            
        lbl = ax.annotate(f'{values[i]:.2f}', xy=xy_coords1[i], textcoords='data', size=10, weight='bold') #annotating values
        axes_annotations.append(lbl)
            
    for i in range(0,len(xy_coords2)):
        
        #Annotating the weights
        txt = f'{nnparams[i]:.3f}'
        
        lbl = ax.annotate(txt, xy=xy_coords2[i], textcoords='data', size=8, weight='bold',color=wcolors[i],rotation=wrotations[i]) #annotating values
        axes_annotations.append(lbl)
    
    feed_forward_btn.disabled = False
    init_btn.disabled = True        

            
def update_params(nnparams):
    
    
    for i in range(0,len(xy_coords1)):

        txt = f'{values[i]:.2f}'
        if(i>10):
            txt = f'{values[i]:.4f}'
        ax = axes_annotations[i]
        ax.set_text(txt) # update annotations for values
            
    for j in range(0,len(xy_coords2)):
        
        txt = f'{nnparams[j]:.3f}'
        
        ax = axes_annotations[len(xy_coords1)+j]
        ax.set_text(txt) # update the annotations for weights
        
def reset_params():
    
    global ann, fig
    
    ann = ANN()
    
    #fig, ax = plt.subplots(frameon=False,figsize=(12,6))  #Plotting the diagram
    ax.clear()
    img = plt.imread("ANN-small.png")
    bg_img = ax.imshow(img)
    plt.axis('off')
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    
    initialize_params(ann.params)
    
    feed_forward_btn.disabled = False
    back_prop_btn.disabled = True
    reset_btn.disabled = True
    input1.disabled = False
    input2.disabled = False
    class_label.disabled = False
    
        
def update_input(x1=input1.value,x2=input2.value,y=class_label.value):
   
    values[0] = x1
    values[1] = x2
    values[10] = y
    
    ann.inputs = np.array([[1,x1,x2]])
    ann.Y = np.array([[y]])

    if(axes_annotations):
    
        update_params(ann.params)            
        
    fig.canvas.draw()      
    
    #print("X1={},X2={},Y={}".format(x1,x2,y))            
    
def feed_forward_update(x1=input1.value,x2=input2.value,y=class_label.value):
   
    global axes_annotations
            
    #X = ann.inputs
    #Y = ann.Y
    
    J = ann.feed_forward_step()
    
        
    set_values(ann.inputs[0,1],ann.inputs[0,2],ann.Y[0,0],ann.outputs[0],ann.outputs[1],ann.activations[0],ann.activations[1])
    
    if(axes_annotations):
    
        update_params(ann.params)    
        
        
    fig.canvas.draw()   
    
    feed_forward_btn.disabled = True
    back_prop_btn.disabled = False
    reset_btn.disabled = False
    input1.disabled = True
    input2.disabled = True
    class_label.disabled = True
    #print("FeedForward: z1={},z2={},a1={},a2={},J={}".format(z1,z2,a1,a2,J))
    

def back_prop_update():

    #X = ann.inputs
    #Y = ann.Y
    W1,W2 = weights2matrices(ann.params)
    
    W1_grad, W2_grad, D = ann.back_prop_step()
    
    W1 = (W1-0.85*W1_grad).flatten()
    W2 = (W2-0.85*W2_grad).flatten()
    
    ann.params = np.concatenate((W1,W2))
    
    d1 = D[0][0,0]
    d2 = D[1][0,0]
    #if(d1.shape[0]>1):
        #d1 = np.sum(d1,axis=1)[1]/d1.shape[0]
    #if(d2.shape[0]>1):
    #    d2 = np.sum(d2,axis=1)/d2.shape[0]
    
    set_values(ann.inputs[0,1],ann.inputs[0,2],ann.Y[0,0],ann.outputs[0],ann.outputs[1],ann.activations[0],ann.activations[1],[d1,d2,W1_grad[0,1]])
    
    update_params(ann.params)
    
    feed_forward_btn.disabled = False
    back_prop_btn.disabled = True
    reset_btn.disabled = False
    
    #print("BackProp: z1={},z2={},a1={},a2={},delta={}".format(z1,z2,a1,a2,delta))
        

upd_inputs = lambda x: update_input(input1.value,input2.value,class_label.value)
ffd_update = lambda x: feed_forward_update(input1.value,input2.value,class_label.value)
bp_update = lambda x: back_prop_update()
reset_click = lambda x: reset_params()
init_click = lambda x: initialize_params(ann.params)

class_label.observe(upd_inputs, names='value')
input1.observe(upd_inputs, names='value')
input2.observe(upd_inputs, names='value')

feed_forward_btn.on_click(ffd_update)
back_prop_btn.on_click(bp_update)
reset_btn.on_click(reset_click)
init_btn.on_click(init_click)


form_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between'
)

form_items = [VBox([input1,input2,class_label]),HBox([init_btn,feed_forward_btn,back_prop_btn,reset_btn])]
form = Box(form_items, layout=Layout(
    display='flex',
    flex_flow='column',
    border='solid 2px',
    align_items='center',
    width='80%'
))

def show_network():
    global fig, ax
    plt.close('all')
    fig, ax = plt.subplots(frameon=False,figsize=(12,6))  #Plotting the diagram
    img = plt.imread("ANN-small.png")
    bg_img = ax.imshow(img)
    plt.axis('off')
    #fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False


