# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 19:02:48 2016

@author: lezhi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#%% ####Preprocessing
#Programmatically download and load into your favorite analytical tool the trip data for September 2015.
#Report how many rows and columns of data you have loaded.

#path ='/home/lezhi/Submission'   #LINUX
#path = '/Users/lezhi/Submission' #mac
#os.chdir(path)

datafile = 'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv'
df0 = pd.read_csv(datafile, index_col=None, parse_dates=True)

(row,col) = df0.shape
print ("Begin Preprocessing:")
print "# of rows and columns in raw data are:", row, "and", col, "."
df = df0
df = df.rename(columns = {'Trip_type ':'Trip_type'})

row, col = None, None

## preprocessing data
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
print ("End Preprocessing.")

#%% Plot inter/intra borough traffic using county shapefile
import shapefile as shp
from matplotlib.patches import Polygon
#from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib import path
from pylab import rcParams, legend
import pyproj

# http://geospatialpython.com/2011/09/reading-shapefiles-from-cloud.html
from StringIO import StringIO
from zipfile import ZipFile
from urllib2 import urlopen

cloudshape = urlopen('https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nybbwi_16c.zip')
memoryshape = StringIO(cloudshape.read())
zipshape = ZipFile(memoryshape)
shpname, dbfname, shxname, prjname, _= zipshape.namelist()
cloudshp = StringIO(zipshape.read(shpname))
cloudshx = StringIO(zipshape.read(shxname))
clouddbf = StringIO(zipshape.read(dbfname))

sf = shp.Reader(shp=cloudshp, shx=cloudshx, dbf=clouddbf)

NYSP1983 = pyproj.Proj(init="ESRI:102718", preserve_units=True)

min_x, max_x = -74.25909, -73.700009    # Longitude
min_y, max_y = 40.477399, 40.917577     # Latitude

#NYC Borough's shape.
fig = plt.figure(figsize=(11,12), frameon=False)
ax = fig.add_subplot(111)
ax.set_axis_off()
recs    = sf.records()
shapes  = sf.shapes()
Nshp    = len(shapes)

dic_rec = {}
for nshp in xrange(Nshp):
    ptchs   = []
    name = recs[nshp][1]
    dic_rec[name]=[]
    temp    = shapes[nshp].points
    pts     = np.array([NYSP1983(v[0], v[1], inverse=True) for v in temp])
    prt     = shapes[nshp].parts
    par     = list(prt) + [pts.shape[0]]
    for pij in xrange(len(prt)):
        ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
        dic_rec[name].append(path.Path(pts[par[pij]:par[pij+1]]))
    ax.add_collection(PatchCollection(ptchs,facecolor='none',edgecolor='k', linewidths=.5))
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
#ax.autoscale_view(True, True,True)
fig.savefig('nyc_boroughs.png', format='png', bbox_inches='tight', pad_inches=0)

data_plot = df
#samples = np.random.choice(data_plot.shape[0], 60000, replace=False)
samples = np.arange(data_plot.shape[0])
datasub = data_plot.iloc[samples, :]

ppx, ppy = 'Pickup_longitude', 'Pickup_latitude'
dfx, dfy = 'Dropoff_longitude', 'Dropoff_latitude'
Points_pp, Points_df = [], []
for idx, row in datasub.iterrows():
    x_pp, y_pp = row[ppx], row[ppy]
    x_df, y_df = row[dfx], row[dfy]
    Points_pp.append([x_pp, y_pp])
    Points_df.append([x_df, y_df])
len_pp, len_df = len(Points_pp), len(Points_df)
Points_pp = np.array(Points_pp).reshape(len_pp, 2)
Points_df = np.array(Points_df).reshape(len_df, 2)

datasub['Borough_pp'] = 'outside'
datasub['Borough_df'] = 'outside'
lst_labels = list(datasub.columns)
iloc_bpp = [k for k, v in enumerate(lst_labels) if v=='Borough_pp'][0]
iloc_bdf = [k for k, v in enumerate(lst_labels) if v=='Borough_df'][0]
for borough in dic_rec:
    print borough
    for p in dic_rec[borough]:
        res_pp = p.contains_points(Points_pp)
        res_df = p.contains_points(Points_df)
        temp = datasub.iloc[:, iloc_bpp]
        datasub.iloc[:, iloc_bpp] = [borough if res_pp[k] else v \
                     for k, v in enumerate(temp)]
        temp = datasub.iloc[:, iloc_bdf]
        datasub.iloc[:, iloc_bdf] = [borough if res_df[k] else v \
                     for k, v in enumerate(temp)]

colors = ['r', 'g', 'b', 'y', 'c']
bcolors = {v:colors[k] for k, v in enumerate(dic_rec.keys())}
bcolors['outside'] = 'm'
extent = min_x, max_x, min_y, max_y

# Plot the whole set
fig = plt.figure(figsize=(11,12))
ax1 = fig.add_subplot(111)
im = plt.imread('nyc_boroughs.png')
implot = plt.imshow(im, extent=extent)
plt.hold(True)
#use pickup locations.
px, py = datasub[ppx], datasub[ppy]
plt.scatter(px, py, marker='o', s=10, c='r', \
            alpha=0.5, edgecolor='none')
ax1.set_xlim(min_x, max_x)
ax1.set_ylim(min_y, max_y)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.tick_params(labelsize=12)
plt.show()
fig.savefig('all_traffic.png', format='png')

print "The boroughs are"
print dic_rec.keys()
for borough in bcolors:
    rcParams['legend.loc'] = 'best'
    fig = plt.figure(figsize=(11,12))
    ax1 = fig.add_subplot(111)
    im = plt.imread('nyc_boroughs.png')
    implot = plt.imshow(im, extent=extent)
    plt.hold(True)
    # Scatter plot for each borough
    df1 = datasub[datasub['Borough_pp']==borough]
    all_loc = float(df1.shape[0])
    df_all = {}
    df_all[borough] = df1[df1['Borough_df']==borough] # drop off locations
    for k, b in enumerate(bcolors.keys()):
        if b==borough:
            continue
        df_all[b] = df1[df1['Borough_df']==b]
    leg, txt = [], []
    for b in df_all:
        print b
        dft = df_all[b]
        px, py = dft[ppx], dft[ppy]
        ax2 = plt.scatter(px, py, s=10, c=bcolors[b], \
                    marker='o', edgecolor='none', alpha=0.5)
        leg.append(ax2)
        txt.append(b+': '+'{:.2f}'.format((dft.shape[0]/all_loc)*100.)+'%')
        plt.hold(True)
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    legend(leg, txt, markerscale=3, loc='upper center', bbox_to_anchor=(0.5, -0.08), \
           fancybox = True, shadow = True, ncol= 3, fontsize=12)
    ax1.set_title('Destination by Pickup Location: '+borough+'. Total pickups: '+str(int(all_loc)), \
                 fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.show()
    fig.savefig(borough+'_traffic.png', format='png')
    