#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pigeon-jupyter --user


# In[ ]:


#!pip install pigeon


# In[2]:


import pandas as pd
import numpy as np
import sys
import os
from IPython.display import display,Image
from pigeon import annotate


# # 1. Read Data 

# In[3]:


#Change the directory based on where you put the csv file
df=pd.read_csv('/home/jianhaozhang/Capstone/old/second_round_labeling.csv')


# In[5]:


df=df.sort_values(by=['obj_id','frame'],ascending=True).reset_index(drop=True)


# # 2. Read already labeled data (Do not run if this is your first time labelling!!!) 

# In[8]:


#Change the directory based on where you put the csv file
second_round_labeled=pd.read_csv("/home/jianhaozhang/Capstone/Label/second_round_labeled.csv")


# In[9]:


len(second_round_labeled)


# # 3. Creat image path

# In[10]:


path="/project2/msca/projects/AvianSolar/ImageDataset/raw_dataset"
Image_path=[]
Image_num=len(df)
for i in range(Image_num):
    Image_path.append(os.path.join(path,df['day_dir'][i],df['camera_dir'][i],df['video_dir'][i],str(df['track_dir'][i]),df['image_file'][i]))


# # 4. Start Labeling

# In[11]:


ann=annotate(Image_path[len(second_round_labeled):],
             options=['Flying over sky','Flying over other backgrounds','Flying over reflection','Flying just above ground',
                      'Flying with solar panel','Flying in the shadow of solar panel','Flying with shadow on the ground',
                      'About to Perch','Sit on panel','Walking on the panel','Sit on the ground',
                      'Walking on the ground','Sit in background','Collision','Unknown'],
            display_fn=lambda filename:display(Image(filename)))


# ### Check for more info if you unsure about the pic

# In[160]:


df.iloc[len(second_round_labeled)+len(ann),[10,11,25,26,27,28,29,30,31,32]]


# # 5. Store results

# In[173]:


ann_pd=pd.DataFrame(ann,columns=['Directory','Label'])


# In[174]:


union=pd.concat([second_round_labeled, ann_pd], ignore_index=True)


# In[175]:


#change your directory
union.to_csv('/home/jianhaozhang/Capstone/second_round_labeled.csv',index=False)

