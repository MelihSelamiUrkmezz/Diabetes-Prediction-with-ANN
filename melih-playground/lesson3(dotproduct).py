inputs=[1,
        2,
        3,
        2.5]
weights=[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]
biasses=[2,3,0.5]
output=[]
# for neuron_weights,biasses in zip(weights,biasses):
#     neuron_output=0
#     for input,weight in zip(inputs,neuron_weights):
#         neuron_output+=input*weight
#     neuron_output+=biasses
#     output.append(neuron_output)
# print(output)

# ---------------
import numpy as np

summary =np.dot(weights,inputs) + biasses

print(summary)