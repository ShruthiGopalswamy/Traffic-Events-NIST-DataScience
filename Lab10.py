#!/usr/bin/python

# imports
import sys
import pandas as pd
import numpy as np
from sklearn import linear_model
#import matplotlib.pyplot as plt
#%matplotlib inline

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)
if len(sys.argv) < 2:
	print 'Arguments missing. Add file path.'
	sys.exit()
	
path = sys.argv[1]

str_ts = '%s/timestamp.tsv'%path
str_flow = '%s/flow.tsv'%path
str_prob = '%s/prob.tsv'%path


data_flow = pd.read_csv(str_flow, sep='\t', header=None)
data_prob = pd.read_csv(str_prob, sep='\t', header=None)
data_time = pd.read_csv(str_ts, sep='\t', header=None)

data_flow[data_flow < 0] = np.nan
data_prob[data_prob < 0] = np.nan
data_flow.fillna(method='ffill', inplace=True)
data_prob.fillna(method='ffill', inplace=True)

zone=pd.DataFrame(index=range(0,len(data_flow)))

timeShiftUp=data_time[1:len(data_time)][[0]]
timeShiftUp=timeShiftUp.reset_index(drop=True)
data_time['laneUp']=timeShiftUp
#Shift Down
timeShiftDown=data_time[0:len(data_time)-1][[0]]
dfnew=pd.DataFrame({0:0},index=[0])
timeShiftDown=pd.concat([dfnew,timeShiftDown]).reset_index(drop=True)
#flowShiftDown=flowShiftDown.ix[:-1]
data_time['laneDown']=timeShiftDown
data_time=data_time.fillna(0)
nextTimeDiff=pd.DataFrame(pd.to_datetime(data_time['laneUp'])-pd.to_datetime(data_time[0])).astype('timedelta64[s]')
data_time['nextDiff']=nextTimeDiff
prevTimeDiff=pd.DataFrame(pd.to_datetime(data_time[0])-pd.to_datetime(data_time['laneDown'])).astype('timedelta64[s]')
data_time['prevDiff']=prevTimeDiff

for i in data_flow.columns :
    reg = linear_model.LinearRegression()
    cols=[col for col in data_flow.columns if col not in [i]]
    #data_flow_filter=data_flow[cols]
    #print(cols)
    reg.fit (data_flow[cols], data_flow[data_flow.columns[i]])
    #reg.coef_, reg.intercept_
    lane=pd.DataFrame(index=range(0,len(data_flow)), columns=['M1','c1'])
    lane['M1']=0
    lane['c1']=0
    j=0
    for k in cols:
        lane['M1']=lane['M1']+(reg.coef_[j]*data_flow[k])
        lane['c1']=lane['c1']+data_prob[k]
        j=j+1
    
    lane['M1']=lane['M1']+reg.intercept_
    lane['c1']=lane['c1']/len(cols)
    #Method1 ~ Method #1: Nearby Lanes + LR
    #Confidence ~
    #lane['c1']=data_prob[1]
    
    #Method2 ~ Method #2: Nearby Timestamps + Weighted Sum
    #Prediction ~ lane
    #Prob ~ Shift Up
    probShiftUp=data_prob[1:len(data_prob)][[i]]
    probShiftUp=probShiftUp.reset_index(drop=True)
    data_prob['laneUp']=probShiftUp
    #Prob ~ Shift Down
    probShiftDown=data_prob[0:len(data_prob)-1][[i]]
    dfnew=pd.DataFrame({i:0},index=[0])
    probShiftDown=pd.concat([dfnew,probShiftDown]).reset_index(drop=True)
    #probShiftDown=probShiftDown.ix[:-1]
    data_prob['laneDown']=probShiftDown
    data_prob=data_prob.fillna(0)
    #Flow ~ Shift Up
    flowShiftUp=data_flow[1:len(data_flow)][[i]]
    flowShiftUp=flowShiftUp.reset_index(drop=True)
    data_flow['laneUp']=flowShiftUp
    #Shift Down
    flowShiftDown=data_flow[0:len(data_flow)-1][[i]]
    dfnew=pd.DataFrame({i:0},index=[0])
    flowShiftDown=pd.concat([dfnew,flowShiftDown]).reset_index(drop=True)
    #flowShiftDown=flowShiftDown.ix[:-1]
    data_flow['laneDown']=flowShiftDown
    data_flow=data_flow.fillna(0)
    #lane['M2']=(data_prob['laneDown']/(data_prob['laneUp']+data_prob['laneDown']))*data_flow['laneDown']+(data_prob['laneUp']/(data_prob['laneDown']+data_prob['laneUp']))*data_flow['laneUp']
    
    ##################
    lane['M2']=(np.where(data_time['prevDiff']<=600, data_prob['laneDown'], 1e-10)*data_flow['laneDown']+np.where(data_time['nextDiff']<=600, data_prob['laneUp'], 1e-10)*data_flow['laneUp'])/((np.where(data_time['nextDiff']<=600, data_prob['laneUp'], 1e-10))+(np.where(data_time['prevDiff']<=600, data_prob['laneDown'], 1e-10)))
    ##################
    
    lane=lane.fillna(0)
    #Method2 ~ Method #2: Nearby Timestamps + Weighted Sum
    #Confidence ~ lane
    lane['c2']=data_prob.loc[:, ['laneUp', 'laneDown']].min(axis=1)
    data_flow.drop(['laneUp', 'laneDown'], axis=1, inplace=True)
    data_prob.drop(['laneUp', 'laneDown'], axis=1, inplace=True)
    
    #Method #3: Keep Measurement Unchanged
    #Prediction ~ lane
    #Confidence ~ lane
    lane['M3']=data_flow[i]
    lane['c3']=data_prob[i]
    zone[i]=((lane['M1']*lane['c1'])+(lane['M2']*lane['c2'])+(lane['M3']*lane['c3']))/(lane['c1']+lane['c2']+lane['c3'])
    zone=zone.round(0)

#Write to file
zone.to_csv(r'F:\Fall2016_DS\Lab10\1160_1.flow.txt', header=None, index=None, sep=' ', mode='w', float_format=None)