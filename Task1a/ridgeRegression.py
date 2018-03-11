import numpy as np
from sklearn import linear_model
from numpy import genfromtxt

data = genfromtxt('train.csv', delimiter=',')
y=data[1:,1]
x=data[1:,2:]
print('size of y is ',y.size,'  and of x is ',x.size)
coefflambda = [0.1, 1, 10, 100, 1000]
#Ridge regression on first part
# reg=linear_model.Ridge (alpha = 0.1)
# reg.fit(x[:49,:],y[:49])
# y_pred = reg.predict(x[50:,:])
# y_error = np.sum(np.power((y[50:]-y_pred),2))/y_pred.size
# print(y_error)
y_pred_temp = np.zeros(450)
y_RMSE = np.zeros((5,10))
x_reg_temp = np.zeros((50,10))
y_reg_temp = np.zeros(50)
x_pred_temp = np.zeros((450,10))
y_real_temp = np.zeros(450)
y_est_temp = np.zeros(450)
for i in range(0,5):
    reg = linear_model.Ridge (alpha = coefflambda[i])
    for j in range(0,10):
        # if j == 0:
        #     x_reg_temp = x[:50,:]
        #     y_reg_temp = y[:50]
        #     x_pred_temp = x[50:,:]
        #     y_real_temp = y[50:]

        # if j == 1:
        #     x_reg_temp = x[50:199,:]
        #     y_reg_temp = y[50:199]
        #
        #     x_pred_temp = np.concatenate((x[:49,:],x[100:,:]),axis=0)
        #     y_real_temp = np.concatenate((y[:49],y[100:]),axis=0)
            # if i ==0:
            #     print('size of first array: ', y[0:50].size, 'size of second array: ', y[101:].shape, 'size of both together: ', y_real_temp.size)
            #     print('size of reg: ',y_reg_temp.size)
        # if j == 2:
        #     x_reg_temp = x[100:149,:]
        #     y_reg_temp = y[100:149]
        #     x_pred_temp = np.concatenate((x[:49,:],x[150:,:]),axis=0)
        #     y_real_temp = np.concatenate((y[:49],y[150:]),axis=0)
        # if j == 3:
        #     x_reg_temp = x[150:199,:]
        #     y_reg_temp = y[150:199]
        #     x_pred_temp = np.concatenate((x[:149,:],x[200:,:]),axis=0)
        #     y_real_temp = np.concatenate((y[:149],y[200:]),axis=0)
        # if j == 4:
        #     x_reg_temp = x[200:249,:]
        #     y_reg_temp = y[200:249]
        #     x_pred_temp = np.concatenate((x[:199,:],x[250:,:]),axis=0)
        #     y_real_temp = np.concatenate((y[:199],y[250:]),axis=0)
        # if j == 5:
        #     x_reg_temp = x[250:299,:]
        #     y_reg_temp = y[250:299]
        #     x_pred_temp = np.concatenate((x[:249,:],x[300:,:]),axis=0)
        #     y_real_temp = np.concatenate((y[:249],y[300:]),axis=0)
        # if j == 6:
        #     x_reg_temp = x[300:349,:]
        #     y_reg_temp = y[300:349]
        #     x_pred_temp = np.concatenate((x[:299,:],x[350:,:]),axis=0)
        #     y_real_temp = np.concatenate((y[:299],y[350:]),axis=0)
        # if j == 7:
        #     x_reg_temp = x[350:399,:]
        #     y_reg_temp = y[350:399]
        #     x_pred_temp = np.concatenate((x[:349,:],x[400:,:]),axis=0)
        #     y_real_temp = np.concatenate((y[:349],y[400:]),axis=0)
        # if j == 8:
        #     x_reg_temp = x[400:449,:]
        #     y_reg_temp = y[400:449]
        #     x_pred_temp = np.concatenate((x[:399,:],x[450:,:]),axis=0)
        #     y_real_temp = np.concatenate((y[:399],y[450:]),axis=0)
        # if j == 9:
        #     x_reg_temp = x[450:500,:]
        #     y_reg_temp = y[450:500]
        #     x_pred_temp = x[:500,:]
        #     y_real_temp = y[:500]
        x_reg_temp = x[50*j:(50*(j+1)),:]
        y_reg_temp = y[50*j:(50*(j+1))]
        x_pred_temp = np.concatenate((x[:(50*j),:],x[50*(j+1):,:]),axis=0)
        y_real_temp = np.concatenate((y[:(50*j)],y[50*(j+1):]), axis=0)
        # print('j is :', j)
        # print('size of real is: ', y_real_temp.size)
        # print('j:', j,'size of x_reg_temp ', x_reg_temp.size,'y_reg_temp: ', y_reg_temp.size,'x_pred_temp:  ',x_pred_temp.size,' y_real_temp ',y_real_temp.size)

        reg.fit(x_pred_temp,y_real_temp)
        y_est_temp = reg.predict(x_reg_temp)
        # print('size of estimate is:', y_est_temp.size)
        y_RMSE[i,j] = np.sqrt(np.sum(np.power(y_reg_temp-y_est_temp,2))/y_est_temp.size)

min_indices = np.argmin(y_RMSE, axis=1)
min_value = np.amin(y_RMSE, axis=1)
print(y_RMSE)
print(min_value)
print('at location')
print(min_indices)
