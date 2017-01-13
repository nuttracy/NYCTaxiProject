# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 19:02:48 2016

@author: lezhi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#####Question 1
#Programmatically download and load into your favorite analytical tool the trip data for September 2015.
#Report how many rows and columns of data you have loaded.

#path ='/home/lezhi/Submission'   #LINUX
path = '/Users/lezhi/Submission' #mac
os.chdir(path)

datafile = 'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv'
df0 = pd.read_csv(datafile, index_col=None, parse_dates=True)

(row,col) = df0.shape
print ("Question 1:")
print "# of rows and columns in raw data are:", row, "and", col, "."
df = df0
df = df.rename(columns = {'Trip_type ':'Trip_type'})

row, col = None, None

#%## preprocessing data
#consider trip less than 3 hours
#def dd2weeks(td):
#    return td.days/7
def dd2hours(td):
    return td.seconds/3600.

def dd2days(td):
    return td.days
    
def dd2mins(td):
    return td.seconds/60.

def dd2secs(td):
    return td.seconds

def preprocess_data(df):
    VendorID = ((df.VendorID == 1) | (df.VendorID == 2))
    Passenger_count = ((df.Passenger_count >= 1.0) & (df.Passenger_count <= 6.0))
    Trip_distance = ((df.Trip_distance > 0.0) & (df.Trip_distance <= 100.0))
    RateCodeID = (df.RateCodeID<=6)
    #Store_and_fwd_flag = df.Store_and_fwd_flag=='Y'
    Payment_type =(df.Payment_type <=6)
    Fare_amount = (df.Fare_amount >= 0.0)
    Extra = ((df.Extra == 0.0) | (df.Extra == 0.5) | (df.Extra == 1.0))
    MTA_tax = (df.MTA_tax == 0.5)
    improvement_surcharge = ((df.improvement_surcharge == 0.0) | (df.improvement_surcharge == 0.3) )
    Tip_amount = (df.Tip_amount >= 0.0)
    Tolls_amount = (df.Tolls_amount>=0.0)
    Total_amount = (df.Total_amount>=0.0)
    Trip_type =((df['Trip_type'] == 1)| (df['Trip_type'] == 0))
    
    df = df[VendorID & Passenger_count & Trip_distance & RateCodeID & Payment_type & 
        Fare_amount & Extra & MTA_tax & improvement_surcharge  & Tip_amount & 
        Tolls_amount & Total_amount & Trip_type]
        
    df = df.drop('Ehail_fee', 1) #drop Nan
    df = df.drop('Store_and_fwd_flag', 1) #drop unclear 
    
    #nyc bounding box
    nyclat2, nyclog1, nyclat1, nyclog2 = 40.917577, -74.25909, 40.477399, -73.700009
    #https://www.maptechnica.com/city-map/New+York/NY/3651000
    
    df = df[((df['Pickup_longitude']>=nyclog1) & 
                    (df['Pickup_longitude']<=nyclog2) &
                    (df['Pickup_latitude'] >=nyclat1) &
                    (df['Pickup_latitude']<=nyclat2))&
                    (df['Dropoff_longitude']>=nyclog1) & 
                    (df['Dropoff_longitude']<=nyclog2) &
                    (df['Dropoff_latitude'] >=nyclat1) &
                    (df['Dropoff_latitude']<=nyclat2)] 
                    
    nyclat2, nyclog1, nyclat1, nyclog2 = None, None, None, None 

    dftp1 = df['lpep_pickup_datetime'].apply(pd.to_datetime)
    dftd1 = df['Lpep_dropoff_datetime'].apply(pd.to_datetime)
    
    df['dhour'] = dftd1 - dftp1
    
    #consider trip less than 3 hours and with positive seconds record
    dhour = df['dhour'].apply(dd2hours)<=3
    dmin = (dftd1 - dftp1).apply(dd2mins) 
    dsec = (dftd1 - dftp1).apply(dd2secs)>0.0
    df = df[dhour & dsec]
    
    #consider trip with average speed less than 75 miiles/h
    df['aver_speed'] = df['Trip_distance'].div(dmin)*60
    speed_limit = (df['aver_speed'] <= 75)
    df = df[speed_limit]
    
    return df

df = preprocess_data(df)

#%%## Question 2
#Plot a histogram of the number of the trip distance ("Trip Distance").
#Report any structure you find and any hypotheses you have about that structure.

def hist_dist(df):
# Histogram of the number of the trip distance
    tripdist = df['Trip_distance']
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1 = tripdist.plot.hist(ax=ax1,bins=50, color='orange',bottom=0.1)
    ax1.set_ylabel('# of Trips', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax2 = tripdist.plot.hist(ax=ax2,bins=50, color='orange',bottom=0.1)
    ax2.set_yscale('log')
    ax2.set_xlabel('Trip Distance', fontsize=14)
    ax2.set_ylabel('# of Trips', fontsize=14)
    ax2.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()
    f.savefig('distance_hist.png')
    return df

df = hist_dist(df)

#%%## Question 3
def stat_dist(df):
    #pickup hour
    dfh1 = pd.DatetimeIndex(df['lpep_pickup_datetime'])
    df['phour'] = dfh1.hour
    
    mean_distp = df.groupby("phour").mean()
    med_distp = df.groupby("phour").median()
    dfout1 =mean_distp['Trip_distance'].to_frame()  
    dfout1.columns = ['dist_mean']
    dfout1.index.name = 'hour_pickup'
    dfout1['dist_med'] = med_distp['Trip_distance']
    
    
    #drop off hour
    dfh2 = pd.DatetimeIndex(df['Lpep_dropoff_datetime'])
    df['dhour'] = dfh2.hour
    
    mean_distd = df.groupby("dhour").mean()
    med_distd = df.groupby("dhour").median()
    dfout2 =mean_distd['Trip_distance'].to_frame()  
    dfout2.columns = ['dist_mean']
    dfout2.index.name = 'hour_dropoff'
    dfout2['dist_med'] = med_distd['Trip_distance']
    
    return df, dfout1, dfout2
    
#Report mean and median trip distance grouped by hour of day.
(df, dfout1, dfout2) = stat_dist(df)

outfile1 = 'TripDistPickup.csv'
dfout1.to_csv(outfile1, columns = dfout1.columns, index = True)

outfile2 = 'TripDistDropoff.csv'
dfout2.to_csv(outfile2, columns = dfout2.columns, index = True)


#We'd like to get a rough sense of identifying trips 
#that originate or terminate at one of the NYC area airports. 
#Can you provide a count of how many transactions fit this criteria, 
#the average fare, and any other interesting characteristics of these trips.

def stat_airport(df, position):
    #JFK airport bounding box
    airlog1, airlog2, airlat1, airlat2 = position[0], position[1], position[2], position[3]
    
    
    dfair = df[((df['Pickup_longitude']>=airlog1) & 
                    (df['Pickup_longitude']<=airlog2) &
                    (df['Pickup_latitude'] >=airlat1) &
                    (df['Pickup_latitude']<=airlat2))|
                    ((df['Dropoff_longitude']>=airlog1) & 
                    (df['Dropoff_longitude']<=airlog2) &
                    (df['Dropoff_latitude'] >=airlat1) &
                    (df['Dropoff_latitude']<=airlat2))] 
                    
    return dfair

position =  [-73.794694, -73.776283, 40.640668, 40.651381]
dfjfk = stat_airport(df, position)
(dfjr,dfjc) = dfjfk.shape                
#11761 transcations are around JFK. 
aver_fare = dfjfk['Total_amount'].mean()
print "# of transactions originated or terminated at JFK is:", dfjr,"which is", dfjr/float(len(df))*100, "% of all the transcations."
print "Average fare is:", aver_fare,"."
print "Average distance is:", dfjfk['Trip_distance'].mean()
#average fare: 
#duration
#tip


#%%## Question 4
#Build a derived variable for tip as a percentage of the total fare.
#Build a predictive model for tip as a percentage of the total fare. 
#Use as much of the data as you like (or all of it). We will validate a sample.
##########
def tip_hist(df):
    dftip = df[df.Payment_type == 1]
    #choose credit card payment
    dftip = dftip[(dftip.Tip_amount>=0)&(dftip.Total_amount>0)]
    dftip['tip_p'] = dftip['Tip_amount'].div(dftip.Total_amount)
    dftip = dftip[np.isfinite(dftip['tip_p'])]
    fig,ax = plt.subplots()
    
    hist, bins = np.histogram(dftip['tip_p'], bins=50)
    ax.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='orange')
    
    ax.set_title('Histogram of Tips', fontsize=14)
    ax.set_xlabel('Percentage of Tips', fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    valx = ax.get_xticks()
    valy = ax.get_yticks()
    
    ax.set_xticklabels(['{:2.2f}%'.format(x*100) for x in valx])
    ax.set_yticklabels(['{:2.2f}%'.format(x*100) for x in valy])
    ax.tick_params(labelsize=12)
    fig = plt.gcf() 
    plt.tight_layout()
    plt.show()
    fig.savefig('tips_hist.png')
    return dftip

dftip = tip_hist(df)

# Trips VS Tips Range
def tip_range(dftip):
    dftip['tip_bin'] = (dftip.tip_p*100./5).apply(int)*5
    upbound = 30
    grouped = dftip[['tip_bin', 'Trip_distance']].groupby('tip_bin')
    temp = grouped.agg(len)
    temp1 = temp[temp.index>=upbound].apply(sum)
    temp2 = pd.DataFrame([[upbound]+list(temp1.values)], columns=['tip_bin']+list(temp1.index.values))
    temp2 = temp2.set_index('tip_bin')
    temp3 = temp[temp.index<upbound]
    trip_count = pd.concat([temp3, temp2])
    idx1 = list(trip_count.index)
    idx2 = ['['+str(v)+','+str(idx1[k+1])+')' if k<len(idx1)-1 else '>='+str(v) for k, v in enumerate(idx1)]
    trip_count.index = idx2
    ax_count = trip_count.plot(kind = 'bar', color = 'orange')
    
    ax_count.tick_params(labelsize=12)
    ax_count.set_xlabel('Tips Range', fontsize=14)
    ax_count.set_ylabel('# of Trips', fontsize=14)
    ax_count.legend_.remove()
    fig = plt.gcf() 
    plt.tight_layout()
    plt.show()
    fig.savefig('tips_range.png', format='png')
    return dftip

dftip = tip_range(dftip)

# set tip_p > 30 into range [30, 100]
tip_bin = dftip.tip_bin>30
dftip[tip_bin] = 30

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import roc_curve, auc

df_train, df_test = train_test_split(dftip)

cols = ['VendorID', 'RateCodeID', 'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude', 
           'Passenger_count', 'Trip_distance', 'Fare_amount', 'Extra', 'MTA_tax', 'Tolls_amount', 'improvement_surcharge', 
           'Total_amount','Trip_type', 'dhour', 'phour']
index = np.arange(len(cols))

model = RandomForestClassifier(n_estimators=10)
model.fit(df_train[cols], df_train.tip_bin)


importances = model.feature_importances_

pickind = importances>importances.mean()
importances1 = importances[pickind]
ind1 = [i for i, x in enumerate(pickind) if x]
cols1 = list( cols[i] for i in ind1 )

plt.figure(figsize=(6 * 2, 6))
index = np.arange(len(importances1))
bar_width = 0.35
ax=plt.bar(index, importances1, color='Turquoise', alpha=0.5)
plt.xlabel('features', fontsize=14)
plt.ylabel('importance', fontsize=14)
plt.title('Top Features', fontsize=14)
plt.tick_params(labelsize=12)
plt.xticks(index + bar_width, cols1)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('importance.png', format='png')
plt.show()

#accuracy
pred = model.predict(df_test[cols])
pscore = metrics.accuracy_score(df_test.tip_bin, pred)
print "Test set accuracy is", pscore

pred_train = model.predict(df_train[cols])
pscore_train = metrics.accuracy_score(df_train.tip_bin, pred_train)
print "Training set accuracy is", pscore_train

#%% ##The following is to plot confusion matrices and AUC curves
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

from matplotlib import cm as cmap

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
 
label_cols = ['lpep_pickup_datetime', 'Lpep_dropoff_datetime']
int_cols = ['VendorID', 'RateCodeID', 'Passenger_count', 'Payment_type', 'Trip_type', \
            'dhour', 'phour']
feature_cols = ['VendorID','lpep_pickup_datetime','Lpep_dropoff_datetime','RateCodeID','Pickup_longitude', \
'Pickup_latitude','Dropoff_longitude','Dropoff_latitude','Passenger_count','Trip_distance','Fare_amount', \
'Extra','MTA_tax','Tip_amount','Tolls_amount','improvement_surcharge','Total_amount','Payment_type', \
'Trip_type','dhour','aver_speed','phour','tip_p']
class_col = 'tip_bin'

for col in label_cols:
    real_col = dftip[col].values
    le = LabelEncoder()
    le.fit(real_col)
    label_col = le.transform(real_col)
    dftip[col] = label_col    
    le, real_col, label_col = None, None, None

for col in int_cols:
    dftip[col] = dftip[col].astype(int)

data_features = dftip[feature_cols].values
data_class = dftip[class_col].values

cross_val = StratifiedShuffleSplit(data_class, n_iter=10, test_size=0.25, random_state=0)

scores = []
conf_matrices = []
precision_scores = {}
recall_scores = {}
pr_auc_scores = {}
fpr_scores = {}
tpr_scores = {}
roc_auc_scores = {}

for train_ind, test_ind in cross_val:
    data_features_train, data_class_train = data_features[train_ind], data_class[train_ind]
    data_features_test, data_class_test = data_features[test_ind], data_class[test_ind]
    model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    model.fit(data_features_train, data_class_train)
    
    if pr_auc_scores=={}:
        for c in model.classes_:
            precision_scores[c] = []
            recall_scores[c] = []
            pr_auc_scores[c] = [] 
            fpr_scores[c] = []
            tpr_scores[c] = []
            roc_auc_scores[c] = []
    
    test_score = model.score(data_features_test, data_class_test)
    scores.append(test_score)
    #Saving confusion matrix
    data_class_pred = model.predict(data_features_test)
    cm = confusion_matrix(data_class_test, data_class_pred)
    conf_matrices.append(cm)
    
    # Saving data for the P/R and ROC scores
    prob = model.predict_proba(data_features_test)
    for c in model.classes_:
        pr_idx = np.where(model.classes_==c)[0][0]
        precision, recall, _ = precision_recall_curve(data_class_test, \
                            prob[:, pr_idx], pos_label=c)
        precision_scores[c].append(precision)
        recall_scores[c].append(recall)
        pr_auc_scores[c].append(auc(recall, precision))
        fpr, tpr, _ = roc_curve(data_class_test, prob[:, pr_idx], pos_label=c)
        fpr_scores[c].append(fpr)
        tpr_scores[c].append(tpr)
        roc_auc_scores[c].append(auc(fpr, tpr))

print 'Accuracy mean: ' + str(np.mean(scores))
print 'Accuracy std: ' + str(np.std(scores))

# Plot confusion matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

classes = ['', 0, 5, 10, 15, 20, 25, 30]
lclasses = ['', '[0,5)', '[5,10)', '[10,15)', '[15,20)', \
        '[20,25)', '[25,30)', '[30,100)',]
first = True
cm = None

for cm_iter in conf_matrices:
    if first:
        cm = cm_iter.copy()
        first = False
    else:
        cm = cm + cm_iter

fig = plt.figure(figsize=(11,12))
ax = fig.add_subplot(111)

colorbar = ax.matshow(cm, cmap=cmap.Blues)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(colorbar, ticks=[50000, 150000, 250000, 350000, 450000, 550000,650000,750000], cax=cax)

ax.set_xlabel('Predicted tip bin', fontsize=16)
ax.set_ylabel('True tip bin', fontsize=16)

ax.set_xticklabels(lclasses)
ax.set_yticklabels(lclasses)

ax.tick_params(labelsize=16)
plt.show()
#fig = plt.gcf()
#plt.tight_layout()
fig.savefig('confusion.png', format='png')

#%%
# Plot PR curve, ROC curve
# A function for plotting a couple of AUC, one per class.

def plot_auc(a_scores, b_scores, auc_scores, messages):
    # First axis.
    a_scores_copy = {}
    a_scores_avg = {}
    
    max_length = 0
    
    for a in a_scores[classes[1]] + a_scores[classes[2]]:
        length = len(a)
        if length > max_length:
            max_length = length
    
    for c in classes[1:]:
        a_scores_copy[c] = []
        for a in a_scores[c]:
            length = len(a)
            
            if length < max_length:
                for i in range(length, max_length):
                    a = np.append(a, a[length - 1])
        
            a_scores_copy[c].append(a)
                
        a_scores_avg[c] = np.average(np.array(a_scores_copy[c]), axis=0)

    # Second axis.
    b_scores_copy = {}
    b_scores_avg = {}

    for c in classes[1:]:
        b_scores_copy[c] = []        
        for b in b_scores[c]:
            length = len(b)
            
            if length < max_length:
                for i in range(length, max_length):
                    b = np.append(b, b[length - 1])            
            b_scores_copy[c].append(b)
        
        b_scores_avg[c] = np.average(np.array(b_scores_copy[c]), axis=0)

    # Plotting.

    fig, ax = plt.subplots(4, 2, figsize=(12, 6))
    ax = ax.reshape(-1)
    i = 0

    for c in classes[1:]:
        a = a_scores_avg[c]
        b = b_scores_avg[c]
        auc_score = np.mean(auc_scores[c])

        if messages['title'] == 'ROC':
            ax[i].plot([0, 1], [0, 1], 'k--')
            ax[i].plot(a, b)
        else:
            print b.shape, a.shape
            ax[i].plot(b, a)

        ax[i].set_xlim([0.0, 1.0])
        ax[i].set_ylim([0.0, 1.0])
        
        if messages['title'] == 'ROC':
            ax[i].fill_between(a, b, alpha=0.5)
        else:
            ax[i].fill_between(b, a, alpha=0.5)
            
        ax[i].set_title(messages['title'] + ' curve (AUC = %0.2f) | Class "%s"' % (auc_score, c), fontsize=15)
        ax[i].set_xlabel(messages['x_label'], fontsize=15)
        ax[i].set_ylabel(messages['y_label'], fontsize=15)
        
        ax[i].tick_params(labelsize=12)
        
        i += 1

    fig.tight_layout()

msg = {'title': 'P/R', 'x_label': 'Recall', 'y_label': 'Precision'}
plot_auc(precision_scores, recall_scores, pr_auc_scores, msg)

msg1 = {'title': 'ROC', 'x_label': 'False positive rate', 'y_label': 'True positive rate'}
plot_auc(fpr_scores, tpr_scores, roc_auc_scores, msg1)