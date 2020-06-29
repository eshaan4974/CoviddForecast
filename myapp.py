#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_new=pd.read_csv('https://covid.ourworldindata.org/data/ecdc/new_cases.csv')


# In[3]:


df_new


# In[4]:


df_final=df_new[['date','India']]


# In[5]:


df_final.rename(columns={'date':'Date','India':'ConfirmedCases'},inplace = True)


# In[6]:


df_j=df_final


# In[7]:


df_j


# In[8]:


df_final=df_final.set_index('Date')


# In[9]:


df_final


# In[10]:


df_final.dropna(inplace=True)


# In[11]:


df_final.ConfirmedCases.isnull().sum()


# In[12]:


df_final.ConfirmedCases=df_final.ConfirmedCases.astype(int)


# In[13]:


df_final.info()


# In[14]:


training_set = df_final['ConfirmedCases'].values
training_set


# In[15]:


training_set=training_set.reshape(-1,1)


# In[16]:


df_final.iloc[:,0].head(6)


# In[17]:


#Date Preprocessing
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating data structure with 30 timesteps 
X_train = []
y_train = []
for i in range(30,len(training_set)-1):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_scaled[[i,i+1], 0])
    
X_train, y_train = np.array(X_train) , np.array(y_train)   

# before reshaping
print(X_train.shape)
print(y_train.shape)
#Reshaping
# [batch_size, timesteps, input_dim(features) :  one result per step]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))


# In[18]:


#Date Preprocessing
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating data structure with 30 timesteps 
X_train = []
y_train = []
for i in range(105,len(training_set)-6):
    X_train.append(training_set_scaled[i-105:i, 0])
    y_train.append(training_set_scaled[[i,i+1,i+2,i+3,i+4,i+5],0])
    #   X_train.append(training_set_scaled[i-30:i, 0])
  #  y_train.append(training_set_scaled[[i,i+1], 0])
       
X_train, y_train = np.array(X_train) , np.array(y_train)   

# before reshaping
print(X_train.shape)
print(y_train.shape)
#Reshaping
# [batch_size, timesteps, input_dim(features) :  one result per step]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))


# In[19]:


len(training_set_scaled)


# In[20]:


X_train.shape


# In[21]:


y_train.shape


# In[22]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[23]:


from keras.optimizers import Adam
from keras.layers import Dense, RepeatVector, TimeDistributed, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D


# In[24]:


regressor = Sequential()

#Add first LSTM layer and Dropout regularisation
regressor.add(Conv1D(filters=4, kernel_size=1, activation='relu', input_shape = (X_train.shape[1], 1)))
regressor.add(Flatten())

regressor.add(RepeatVector(X_train.shape[1]))

regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))


#Adding second layer
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding third layer
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding fourth layer
#regressor.add(LSTM(units =50, return_sequences = True))
#regressor.add(Dropout(0.2))


#regressor.add(LSTM(units =50, return_sequences = True))
#regressor.add(Dropout(0.2))
#Adding fifth layer
regressor.add(LSTM(units =50))
regressor.add(Dropout(0.2))


#regressor.add(Dense(units = 16))
#regressor.add(Dense(activation='relu',units = 32))
#regressor.add(Dense(activation='relu',units = 64))

#Output layer
regressor.add(Dense(units = 6))
opt = Adam(learning_rate=0.05)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

#Training the model


# In[25]:


regressor.summary()


# In[26]:


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode='min', baseline=0.0080)


# In[27]:


regressor.fit(X_train, y_train, epochs = 100)


# In[28]:


real_confirmed_cases = df_final.iloc[105:,0].values
#print("real case",real_confirmed_cases,len(real_confirmed_cases))
X_test = []

for i in range(105,len(training_set)+1):
    X_test.append(training_set_scaled[i-105:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#X_test.append(training_set_scaled[:105])
#X_test = np.array(X_test)

subsequences = 1
#timesteps = X_test.shape[1]//subsequences
#X_test_series_sub = X_test.reshape((-1,30, subsequences))
#print('Train set shape', X_test_series_sub .shape)

predicted_confirmed_cases = regressor.predict(X_test )
predicted_confirmed_cases = sc.inverse_transform(predicted_confirmed_cases)
#print("predicted result",predicted_confirmed_cases,len(predicted_confirmed_cases))


# In[29]:


predicted_confirmed_cases[len(predicted_confirmed_cases)-1]


# In[30]:


plt.figure(figsize = (12,8))
plt.plot(real_confirmed_cases[-1:], color='c',marker = 'o', label = 'Real Confirmed Cases')
plt.plot(predicted_confirmed_cases[len(predicted_confirmed_cases)-2], color='g',marker = 'o', label = 'Predicted Number of Cases')
plt.title('Coronavirus Forecasting Trend')
plt.xlabel('Days')
plt.ylabel('Item Count')
plt.legend()
plt.show()


# In[31]:


df=df_final.iloc[105:,0:]


# In[32]:


predicted_confirmed_cases[len(predicted_confirmed_cases)-1]


# In[33]:


df['Prediction(1day)']=0
df['Prediction(2day)']=0
df['Prediction(3day)']=0
df['Prediction(4day)']=0
df['Prediction(5day)']=0
df['Prediction(6day)']=0


# In[34]:


for i in range(len(df)):
    df['Prediction(1day)'][i]=predicted_confirmed_cases[i][0]
    df['Prediction(2day)'][i]=predicted_confirmed_cases[i][1]
    df['Prediction(3day)'][i]=predicted_confirmed_cases[i][2]
    df['Prediction(4day)'][i]=predicted_confirmed_cases[i][3]
    df['Prediction(5day)'][i]=predicted_confirmed_cases[i][4]
    df['Prediction(6day)'][i]=predicted_confirmed_cases[i][5]


# In[35]:


df


# In[36]:


df_sis = df


# In[37]:


df_sis.reset_index(inplace=True)


# In[38]:


df_sis


# In[39]:


df.tail()


# In[40]:


date=df.Date[df.shape[0]-1]


# In[41]:


import datetime
date_str = date

format_str = '%Y-%m-%d' # The format
datetime_obj = datetime.datetime.strptime(date_str, format_str)


# In[42]:


data=pd.DataFrame(columns=['date','Predicted_Cases'])


# In[43]:


date=[]
cases=[]
for i in range(6):
    NextDay_Date = datetime_obj.date() + datetime.timedelta(days=i+1)
    print('for the',NextDay_Date,'the predicted number of cases :',int(predicted_confirmed_cases[df.shape[0]][i]))
    date.append(str(NextDay_Date))
    cases.append(int(predicted_confirmed_cases[df.shape[0]][i]))


# In[44]:


print(date)
print(cases)


# In[45]:


result=pd.DataFrame(list(zip(date,cases)),columns=['date','Predicted_Cases'])
result


# In[47]:


from bokeh.layouts import widgetbox, column,row
from bokeh.models import  LabelSet,ColumnDataSource,Text
import pandas as pd
from bokeh.io import show
from bokeh.models import ColumnDataSource,LabelSet
from bokeh.palettes import Category10
from bokeh.plotting import figure,output_notebook,output_file
from bokeh.models.tools import HoverTool
from bokeh.palettes import Turbo256,linear_palette,Inferno256
from bokeh.io import curdoc
import math
from bokeh.io import curdoc
from bokeh.plotting import figure, show, ColumnDataSource,output_notebook
from bokeh.models import RangeSlider,HoverTool
from bokeh.layouts import widgetbox, column, row
from bokeh.palettes import Greys256,Inferno256,Magma256,Plasma256,Viridis256,Cividis256,Turbo256,linear_palette,Set2,Spectral6
from bokeh.models.widgets import Div

df=df_j
df1=result
c1 = list(df['Date'].values)
d1=df['ConfirmedCases'].tolist()
#e=df1['Death_per_million_population'].tolist()
l1=len(c1)
color2=linear_palette(Turbo256,l1)
p12 = figure(x_range=c1, plot_height=350, title="Daily New Cases in India",
           toolbar_location='right', tools="zoom_in,zoom_out,reset,save,pan,box_select,box_zoom",plot_width=650)
a=p12.line(c1, d1, line_width=3,color='cornflowerblue')
p12.annulus(x=c1, y=d1, inner_radius=0.1, outer_radius=0.25,color=color2, alpha=0.8)
p12.segment(c1,-1000, c1,d1, line_width=3, line_color=color2, alpha=0.8 )
p12.y_range.start = -200
p12.xgrid.visible = False
p12.ygrid.visible = False
p12.axis.minor_tick_line_color = None
p12.title.align = "center"
p12.title.text_font_size = "20px"
p12.xaxis.major_label_orientation = math.pi/2
p12.xaxis.major_label_text_font_size="8px"
hover=HoverTool(tooltips=([('Date_Month_Year','@x'),('New_Cases','@y')]),renderers=[a])
p12.add_tools(hover)

"""c2 = list(df['Date'].values)
d2=df['Total_Cases'].tolist()
l=len(c2)
color3=linear_palette(Viridis256,l)
p13 = figure(x_range=c2, plot_height=350, title="Total Cases in India - Till date on daily basis",
           toolbar_location='right', tools="zoom_in,zoom_out,reset,save,pan,box_select,box_zoom",plot_width=650,y_axis_label='No.of.cases')
p13.segment(c2,-5000, c2,d2, line_width=3, line_color=color3, alpha=0.8 )
a=p13.circle(x=c2,y= d2, size=8, fill_color=color3, line_color="black", line_width=2,alpha=0.65 )
p13.xgrid.visible = False
p13.ygrid.visible = False
p13.axis.minor_tick_line_color = None

hover=HoverTool(tooltips=([('Date','@x'),('Cases','@y')]),renderers=[a])
p13.add_tools(hover)
p13.y_range.start = -5000
p13.title.align = "center"
p13.title.text_font_size = "20px"
p13.xaxis.major_label_orientation = math.pi/2
p13.xaxis.major_label_text_font_size="8px"
p13.yaxis.visible=False
#p13.y_axis_label='No.of.cases'
"""
from bokeh.transform import factor_cmap
a=df1["date"].tolist()
b=df1["Predicted_Cases"].tolist()
df1['color'] = Spectral6
source=ColumnDataSource(df1)
p1=figure(y_range=a,x_range=(0,16000),background_fill_color="white",title="Covid-19 Prediction",x_axis_label="Predicted Count",
         y_axis_label="Date", tools="pan,box_select,xzoom_out,save,reset,box_zoom",plot_height=350,plot_width=650)
renderers=p1.hbar(y='date',right='Predicted_Cases',left=0,height=0.4,fill_color='color',fill_alpha=0.9,source=source,legend_field='date')
hover=HoverTool()
hover.tooltips="""
<div>
<h3><strong>Date : </strong>@date</h3>
<div><strong>Cases : </strong>@Predicted_Cases</div>
</div>
"""
p1.x_range.start = 0
p1.title.align = "center"
p1.title.text_font_size = "20px"
p1.add_tools(hover)
p1.ygrid.grid_line_color = None
p1.xgrid.grid_line_color = None
from bokeh.core.properties import value
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource,HoverTool
from bokeh.plotting import figure
from bokeh.transform import dodge
from math import pi
dfc=df_sis

Date = list(dfc['Date'])
Cases =['Predicted','confirmed']
confirmed  =list(dfc['ConfirmedCases'])
Predicted=list(dfc['Prediction(1day)'])


data = {'Date' : Date,
        'confirmed'   : list(dfc['ConfirmedCases']),
        'Predicted'   : list(dfc['Prediction(1day)'])}

palette = ["deeppink","dodgerblue"]
source = ColumnDataSource(data=data)

p = figure(x_range=Date, y_range=(0, 12000), plot_height=350,plot_width=650,title="Confirmed Vs Predicted Cases",x_axis_label="Date",
           y_axis_label="Number_of_cases",toolbar_location='right',tools="box_select,box_zoom")

x1=p.vbar(x=dodge('Date', -0.25, range=p.x_range), top='confirmed', width=0.2, source=source,
       color="deeppink", legend=value("confirmed"))

x2=p.vbar(x=dodge('Date',  0.0,  range=p.x_range), top='Predicted', width=0.2, source=source,
       color="dodgerblue", legend=value("Predicted"))


#p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
p.xaxis.major_label_orientation = pi/2.5
p.grid.visible=False
hover=HoverTool(tooltips=([('Date','@Date'),('confirmed','@confirmed'),('predicted','@Predicted')]))
p.add_tools(hover)
p.title.align = "center"
p.title.text_font_size = "20px"
#output_notebook()
#show(p)
pre=Div(text=""" <div><h3><strong><center> Forecasting of covid19 in India</center></strong><h3></div>""",
        align='center',style={'color':'darkred','font-size':'30px','font-family':'Helvetica'})
layout1=column(pre,row(p1,p),row(p12))
show(layout1)
output_file("covid19prediction.html",title="covid19")
#show(p12)
#forecasting of covid19 in india
#curdoc().add_root(layout1)
#curdoc().title="covid19"


# In[ ]:





# In[ ]:




