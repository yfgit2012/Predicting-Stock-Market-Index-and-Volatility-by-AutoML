# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:55:08 2022

@author: fbai_
"""


import matplotlib.pyplot as plt
import pandas as pd


Title = ['SPX return prediction by RF (2018-2020)', 'SPX return prediction by AB (2018-2020)',  
         'SPX return prediction by XB (2018-2020)', 'SPX return prediction by SVM (2018-2020)',  
         'SPX return prediction by LSTM (2018-2020)', 'SPX return prediction by AG (2018-2020)'] 


df1 = pd.read_csv('./results/'+Title[0]+'.csv')
df2 = pd.read_csv('./results/'+Title[1]+'.csv')
df3 = pd.read_csv('./results/'+Title[2]+'.csv')
df4 = pd.read_csv('./results/'+Title[3]+'.csv')
df5 = pd.read_csv('./results/'+Title[4]+'.csv')
df6 = pd.read_csv('./results/'+Title[5]+'.csv')

#plt.figure(figsize=(20,10))

#plt.subplot(2,3, 1)
plt.figure()
plt.title(Title[0])
plt.plot(df1['actual_return'], label="label")
plt.plot(df1['pred_return'],label="prediction")
plt.xlabel("Data points")
plt.ylabel("SPX return")
plt.text(-5, -6, 'RMSE=1.04, Signal_accuracy=71.2%', fontsize = 10)
plt.legend(loc="upper left")
plt.savefig('./results/'+Title[0]+'.png')
plt.show()

#plt.subplot(2, 3, 2)
plt.figure()
plt.title(Title[1])
plt.plot(df2['actual_return'], label="label")
plt.plot(df2['pred_return'],label="prediction")
plt.xlabel("Data points")
plt.ylabel("SPX return")
plt.text(-5, -6, 'RMSE=1.04, Signal_accuracy=70.4%', fontsize = 10)
plt.legend(loc="upper left")
plt.savefig('./results/'+Title[1]+'.png')
plt.show()


#plt.subplot(2, 3, 3)
plt.figure()
plt.title(Title[2])
plt.plot(df3['actual_return'], label="label")
plt.plot(df3['pred_return'],label="prediction")
plt.xlabel("Data points")
plt.ylabel("SPX return")
plt.text(-5, -6, 'RMSE=0.87, Signal_accuracy=63.5%', fontsize = 10)
plt.legend(loc="upper left")
plt.savefig('./results/'+Title[2]+'.png')
plt.show()


#plt.subplot(2, 3, 4)
plt.figure()
plt.title(Title[3])
plt.plot(df4['actual_return'], label="label")
plt.plot(df4['pred_return'],label="prediction")
plt.xlabel("Data points")
plt.ylabel("SPX return")
plt.text(-5, -6, 'RMSE=1.03, Signal_accuracy=60.2%', fontsize = 10)
plt.legend(loc="upper left")
plt.savefig('./results/'+Title[3]+'.png')
plt.show()


#plt.subplot(2, 3, 5)
plt.figure()
plt.title(Title[4])
plt.plot(df5['actual_return'], label="label")
plt.plot(df5['pred_return'],label="prediction")
plt.xlabel("Data points")
plt.ylabel("SPX return")
plt.text(-5, -6, 'RMSE=1.17, Signal_accuracy=47.8%', fontsize = 10)
plt.legend(loc="upper left")
plt.savefig('./results/'+Title[4]+'.png')
plt.show()


#plt.subplot(2, 3, 6)
plt.figure()
plt.title(Title[5])
plt.plot(df6['actual_return'], label="label")
plt.plot(df6['pred_return'],label="prediction")
plt.xlabel("Data points")
plt.ylabel("SPX return")
plt.text(-5, -6, 'RMSE=1.00, Signal_accuracy=68.4%', fontsize = 10)
plt.legend(loc="upper left")
plt.savefig('./results/'+Title[5]+'.png')
plt.show()


