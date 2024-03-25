#Derin öğrenmenin temeli ağırlıkları ve bias'ı iyi belirlemekten geçer.

inputs=[1,2,3,2.5]
weights=[[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
biasses=[2,3,0.5]
output=[0,0,0]

for i in range(0,3):
    for x in range(0,len(inputs)):
        output[i]+=inputs[x]*weights[i][x]
    output[i]+=biasses[i]
    
print(output)    
