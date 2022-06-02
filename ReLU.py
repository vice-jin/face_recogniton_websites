import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np




# 绘制ReLU函数
fig =  plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
x = np.linspace(-10,10)
y = np.where(x<0,0,x) # 小于0输出0，大于0输出y
plt.xlim(-11,11)
plt.ylim(-11,11)
 
ax = plt.gca() # 获得当前axis坐标轴对象
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
 
ax.xaxis.set_ticks_position('bottom') # 指定下边的边作为x轴
ax.yaxis.set_ticks_position('left') # 指定左边的边为y轴
 
ax.spines['bottom'].set_position(('data',0)) # 指定data 设置的bottom（也就是指定的x轴）绑定到y轴的0这个点上
ax.spines['left'].set_position(('data',0))  # 指定y轴绑定到x轴的0这个点上
 
plt.plot(x,y,label = 'ReLU',linestyle='-',color='r')
plt.legend(['ReLU'])
 
plt.show()


      