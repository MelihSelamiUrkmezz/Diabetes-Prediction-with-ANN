# Her bir nöronun kendine özel bias'ı vardır.
# Her bir girdinin kendine özel ağırlığı vardır. Random başlayabilir.
# Girdiler ya gerçekten bir girdi katmanı olabilir, bir önceki nöronun çıktı katmanı da olabilir.
# Bias sayısının girdi sayısıyla bir ilişkisi yoktur.

 

inputs= [3.4,5.2,2.8]
weights= [2.1,4.3,1.2]
bias = 5
output = 0
for i in range(0,3):
    print(i)
    output+=inputs[i]*weights[i]
    
output=output+bias

print("Output:"+str(output))