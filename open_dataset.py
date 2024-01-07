from scipy.io import arff
from IPython import embed
# 读取.arff文件
data, meta = arff.loadarff('EthanolConcentration.arff')#1751*3
# embed()
# 打印数据集信息
print(meta)
embed()
for instance in data:
    print(instance)
    embed()





