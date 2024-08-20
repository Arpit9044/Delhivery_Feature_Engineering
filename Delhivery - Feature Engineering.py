#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 50)


# In[2]:


df=pd.read_csv(r"C:\Users\hp\Downloads\delhivery_data.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# - source_name & destination_name have some null values 

# In[5]:


df.describe()


# In[6]:


df.describe(include='object')


# In[7]:


df[df.duplicated()]


# - There are no duplicates in the data.

# In[8]:


df.isna().sum()


# In[9]:


df.isna().sum()


# In[10]:


df.dropna(how='any',inplace=True)
df.reset_index(drop=True)


# In[11]:


df.isna().sum()


# In[12]:


df.columns


# In[13]:


df['od_start_time']=pd.to_datetime(df['od_start_time'])
df['od_end_time']=pd.to_datetime(df['od_end_time'])
df[df['trip_uuid']=='trip-153741093647649320']


# In[14]:


create_segment_dict = {
    'data' : 'first',
    'trip_creation_time': 'first',
    'route_schedule_uuid' : 'first',
    'route_type' : 'first',
    
    'source_center' : 'first',

    'destination_center' : 'last',

    'od_start_time' : 'first',
    'od_end_time' : 'first',
    'start_scan_to_end_scan' : 'first',


    'actual_distance_to_destination' : 'last',
    'actual_time' : 'last',

    'osrm_time' : 'last',
    'osrm_distance' : 'last',
    
    'segment_actual_time' : 'sum',
    'segment_osrm_time' : 'sum' ,
    'segment_osrm_distance' : 'sum'
    }


# In[15]:


df1=df.groupby(['trip_uuid','source_name','destination_name']).agg(create_segment_dict)
df1 = df1.reset_index()
df1


# In[16]:


df1=df1.sort_values(by=['trip_uuid','source_name','destination_name','od_end_time'], ascending=True)
df1


# ### Calculate time taken between od_start_time and od_end_time and keep it as a feature. `
# 
# - od_time_diff_hour is matching with start_scan_to_end_scan

# In[17]:


df1['od_time_diff_hour'] = (df1['od_end_time'] - df1['od_start_time']).dt.total_seconds() / (60)
df1['od_time_diff_hour']


# In[18]:


df1


# Let us transform this data to a trip level information.

# In[19]:


trip_dict = { 'data' : 'first',
              'trip_creation_time': 'first',
              'route_schedule_uuid' : 'first',
              'route_type' : 'first',
   

              'source_center' : 'first',
              'source_name' : 'first',

              'destination_center' : 'last',
              'destination_name' : 'last',

              'start_scan_to_end_scan' : 'sum',
              'od_time_diff_hour': 'sum' ,


              'actual_distance_to_destination' : 'sum',
              'actual_time' : 'sum',
              'osrm_time' : 'sum',
              'osrm_distance' : 'sum',
    

              'segment_actual_time' : 'sum',
              'segment_osrm_distance' : 'sum',
              'segment_osrm_time' : 'sum',
}


# In[20]:


df2=df1.groupby('trip_uuid').agg(trip_dict).reset_index()
df2


# In[21]:


df2.drop(['start_scan_to_end_scan'],axis=1,inplace=True)
df2


# # Performing Hypothesis Testing on Actual and OSRM Time.

# In[51]:


df2['actual_distance_to_destination'].sample(1000).mean()


# In[52]:


df['osrm_distance'].sample(1000).mean()


# - We are able to see that actual distance is less than osrm distance.Let us check whether this difference is significant or not.

# In[53]:


actual_distance=df2['actual_distance_to_destination'].sample(1000)
osrm_distance=df2['osrm_distance'].sample(1000)


# In[54]:


#Ho----> actual_distance >= osrm_distance
#Ha----> actual_distance <  osrm_distance

from scipy.stats import ttest_ind
t_test,p_value=ttest_ind(actual_distance,osrm_distance,alternative='less')
p_value


# In[55]:


#Lets assume the significance level at 5%
alpha=0.05
if p_value<alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to Reject Null Hypothesis')


# Therefore, there is a significant difference between actual distance and osrm distance.Actual distance is less than what is predicted.

# # Performing Hypothesis Testing on Actual Time and OSRM Time.

# In[22]:


df2['actual_time'].sample(1000).mean()


# In[23]:


df['osrm_time'].sample(1000).mean()


# - We are able to see that actual time is more than osrm time.Let us check whether this difference is significant or not.

# In[24]:


actual_time=df2['actual_time'].sample(1000)
osrm_time=df2['osrm_time'].sample(1000)


# In[25]:


#Ho----> actual_time <= osrm_time
#Ha----> actual_time > osrm_time

from scipy.stats import ttest_ind
t_test,p_value=ttest_ind(actual_time,osrm_time,alternative='greater')
p_value


# In[26]:


#Lets assume the significance level at 5%
alpha=0.05
if p_value<alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to Reject Null Hypothesis')


# Actual time is significantly more than osrm time.

# # Feature Engineering

# - Extracting/Engineering new features from trip_creation_time

# In[27]:


df2['trip_creation_time'] =  pd.to_datetime(df2['trip_creation_time'])

df2['trip_year'] = df2['trip_creation_time'].dt.year
df2['trip_month'] = df2['trip_creation_time'].dt.month
df2['trip_hour'] = df2['trip_creation_time'].dt.hour
df2['trip_day'] = df2['trip_creation_time'].dt.day
df2['trip_week'] = df2['trip_creation_time'].dt.isocalendar().week
df2['trip_dayofweek'] = df2['trip_creation_time'].dt.dayofweek


# In[28]:


df2[['trip_year',
'trip_month',
'trip_hour',
'trip_day',
'trip_week',
'trip_dayofweek']]


# - Splitting source name and destination name to engineer features like "source_city","source_state","destination_city","destination_state"

# In[29]:


df2['source_name']=df2['source_name'].str.lower()
df2['destination_name']=df2['destination_name'].str.lower()


# In[30]:


def place2state(x):
    state=x.split('(')[1]
    return state[:-1]

def place2city(x):
    city=x.split('_')[0]
    return  city


# In[31]:


df2['source_state']=df2['source_name'].apply(lambda x : place2state(x))
df2['source_city']=df2['source_name'].apply(lambda x : place2city(x))


# In[32]:


df2['destination_state']=df2['destination_name'].apply(lambda x : place2state(x))
df2['destination_city']=df2['destination_name'].apply(lambda x : place2city(x))


# In[33]:


df2[['source_state', 'source_city','destination_state', 'destination_city']]


# In[60]:


df2['source_state'].value_counts()


# In[61]:


df2['source_city'].value_counts()


# In[62]:


df2['destination_state'].value_counts()


# In[63]:


df2['destination_city'].value_counts()


# # Outliers Detection and Treatment

# In[34]:


cols = ['actual_distance_to_destination', 'actual_time', 'osrm_time',
            'osrm_distance', 'segment_actual_time', 'segment_osrm_distance',
            'segment_osrm_time', 'od_time_diff_hour']


# In[35]:


df2[cols].boxplot(figsize=(25,8))


# In[36]:


Q1 = df2[cols].quantile(0.25)
Q3 = df2[cols].quantile(0.75)

IQR = Q3 - Q1


# In[37]:


df2 = df2[~((df2[cols] < (Q1 - 1.5 * IQR)) | (df2[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df2 = df2.reset_index(drop=True)


# In[38]:


df2


# In[39]:


df2[cols].boxplot(figsize=(25,8))


# - As we can see, all the outliers have been removed.

# # Scaling the numerical features to prevent biasness

# - MinMax Scaling or Normalization

# In[41]:


from sklearn.preprocessing import MinMaxScaler


# In[44]:


scaler_1=MinMaxScaler()
df2_normalized=pd.DataFrame(scaler_1.fit_transform(df2[cols]),columns=cols)
df2_normalized


# - Standardization

# In[47]:


from sklearn.preprocessing import StandardScaler
scaler_2=StandardScaler()
df2_standardized=pd.DataFrame(scaler_2.fit_transform(df2[cols]),columns=cols)
df2_standardized


# # Recommendations
# 
# ### There is a significant difference between actual and osrm time.The osrm time is less and hence the model could be corrected accordingly so that customers are able to see the correct osrm time.
# 
# ### Top States contributing to business:
#    ##### - Maharashtra
#    ##### - Karnataka
#    ##### - Haryana
#    ##### - Tamil Nadu
#    ##### - Telangana
#  
# ### Top Cities contributing to business:
#    ##### - Bengaluru
#    ##### - Gurgaon
#    ##### - Mumbai
#    ##### - Delhi
#    ##### - Bhiwandi
# 
#    ### - Delhivery should keep special focus on these states and cities.
# 
# ### Can we collect data in a better way so that it becomes easy for data analyst and scientists to analyst.

# In[ ]:




