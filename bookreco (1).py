import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import isbnlib
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
from pylab import plot, show
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from progressbar import ProgressBar
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")
print('Libraries imported!')

df = pd.read_csv('output.csv')


df.index = df['bookID']

#print("# of NaN in each columns:", df.isnull().sum(), sep='\n')


format(len(df['language_code'].unique()))


#Merging 'en-US','en-GB','en-CA' to 'eng'
df.replace(to_replace='en-US', value = 'eng', inplace=True)
df.replace(to_replace='en-GB', value = 'eng', inplace=True)
df.replace(to_replace='en-CA', value = 'eng', inplace=True)

# Drop the 'bookID' column if it exists
df = df.drop(columns=['bookID'], errors='ignore')

# Reset the index
df_reset_index = df.reset_index()


# Now, you can proceed with creating the plot as before
fig = plt.figure(figsize=(15, 10))
plt.title("Language Distribution")
ax = sns.countplot(x='language_code', data=df_reset_index, palette='inferno')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')


trial = df[['average_rating', 'ratings_count']]
data = np.asarray([np.asarray(trial['average_rating']), np.asarray(trial['ratings_count'])]).T

X = data
distortions = []
for k in range(2,30):
    k_means = KMeans(n_clusters = k)
    k_means.fit(X)
    distortions.append(k_means.inertia_)

fig = plt.figure(figsize=(15,10))
#plt.plot(range(2,30), distortions, 'bx-')
#plt.title("Elbow Curve")

#Computing K means with K = 5, thus, taking it as 5 clusters
centroids, _ = kmeans(data, 5)

#assigning each sample to a cluster
#Vector Quantisation:

idx, _ = vq(data, centroids)

# some plotting using numpy's logical indexing
sns.set_context('paper')
plt.figure(figsize=(15,10))
plt.plot(data[idx==0,0],data[idx==0,1],'or',#red circles
     data[idx==1,0],data[idx==1,1],'ob',#blue circles
     data[idx==2,0],data[idx==2,1],'oy', #yellow circles
     data[idx==3,0],data[idx==3,1],'om', #magenta circles
     data[idx==4,0],data[idx==4,1],'ok',#black circles
)
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8, )


circle1 = Line2D(range(1), range(1), color = 'red', linewidth = 0, marker= 'o', markerfacecolor='red')
circle2 = Line2D(range(1), range(1), color = 'blue', linewidth = 0,marker= 'o', markerfacecolor='blue')
circle3 = Line2D(range(1), range(1), color = 'yellow',linewidth=0,  marker= 'o', markerfacecolor='yellow')
circle4 = Line2D(range(1), range(1), color = 'magenta', linewidth=0,marker= 'o', markerfacecolor='magenta')
circle5 = Line2D(range(1), range(1), color = 'black', linewidth = 0,marker= 'o', markerfacecolor='black')

plt.legend((circle1, circle2, circle3, circle4, circle5)
           , ('Cluster 1','Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'), numpoints = 1, loc = 0, )

df.to_csv('df.csv')



