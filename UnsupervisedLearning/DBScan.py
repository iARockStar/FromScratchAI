#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def DBSCAN(dataset, e, m_s):
    # create an array full of -1 values with a length of our dataset
    labels = np.full(len(dataset), -1)  

    # iterate over each datapoint of the dataset
    for i in range(len(dataset)):
        # check whether this datapoint is noisy/ in a cluster or not
        if labels[i] == -1:
            # if not, find its neighbors in the radius of e 
            dists = np.linalg.norm(dataset - dataset.iloc[i], axis=1)
            neighbors = [i for i, dist in enumerate(dists) if dist <= e]
            number_of_neighbors = len(neighbors)
            # if the count of neighbors is less than m_s, then it is a noisy point. (but for now, it may become a border point).
            if number_of_neighbors < m_s:
                labels[i] = 0  
            else:
                # a new core point is founed and we have a new label
                neighbor_indx = 0
                labels[i] = i
                # iterate over the neighbors if this core point 
                while neighbor_indx < len(neighbors):
                    # get the index of the jth neighbor
                    neighbor = neighbors[neighbor_indx]
                    # if the neighbor is noisy, then here it becomes a border point since its directly-density-reachable from 
                    # a core point.
                    if labels[neighbor] == 0:
                        labels[neighbor] = i  
                    # if the neighbor is not labeled yet, then find its neighbors again
                    elif labels[neighbor] == -1:
                        labels[neighbor] = i
                        dists = np.linalg.norm(dataset - dataset.iloc[neighbor], axis=1)
                        neighbors_neighbors = [i for i, dist in enumerate(dists) if dist <= e]    
                        # if the neighbor is also a core point, then the neighbor itself and its neighbors are also in our
                        # new cluster since they are density reachable from the core_point
                        if len(neighbors_neighbors) >= m_s:
                            neighbors.extend(neighbors_neighbors)
                    neighbor_indx += 1                
    return labels




# In[2]:


df_1 = pd.read_csv("d1.csv")
df_2 = pd.read_csv("d2.csv")

print(df_1.isna().sum())
print(df_2.isna().sum())


# In[3]:


df_1.head(10)


# In[4]:


df_2.head(10)


# In[5]:


labels_df1 = DBSCAN(df_1, 0.2, 10)

labels_df2 = DBSCAN(df_2, 0.2, 10)

unique_labels1 = np.unique(labels_df1)
unique_labels2 = np.unique(labels_df2)
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].scatter(df_1['x'], df_1['y'], color='gray')
axs[0, 0].set_xlabel('feature_1')
axs[0, 0].set_ylabel('feature_2')
axs[0, 0].set_title('scatter plot of the dataset 1')

for label in unique_labels1:
    axs[0, 1].scatter(df_1.loc[labels_df1 == label, 'x'], df_1.loc[labels_df1 == label, 'y'], label="cluster {}".format(label))
axs[0, 1].set_xlabel('feature_1')
axs[0, 1].set_ylabel('feature_2')
axs[0, 1].set_title('result for df_1')
axs[0, 1].legend()

axs[1, 0].scatter(df_2['x'], df_2['y'], color='gray')
axs[1, 0].set_xlabel('feature_1')
axs[1, 0].set_ylabel('feature_2')
axs[1, 0].set_title('scatter plot of the dataset 1')

for label in unique_labels2:
    axs[1, 1].scatter(df_2.loc[labels_df2 == label, 'x'], df_2.loc[labels_df2 == label, 'y'], label="cluster {}".format(label))

axs[1, 1].set_xlabel('feature_1')
axs[1, 1].set_ylabel('feature_2')
axs[1, 1].set_title('result for df_2')
axs[1, 1].legend()
plt.tight_layout()
plt.show()


# In[ ]:




