import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv("/Users/yaarakaplan/Classes/ML/capstone/data/data (1).csv")

# check for good ratio of liked to non-liked songs
print(data[data["target"]==0].shape[0]) # = 997
print(data[data["target"]==1].shape[0]) # = 1020

# plot danceability
fig = plt.figure()
plt.title("Danceability Target distribution")
liked = data[data["target"]==1]["danceability"]
dis = data[data["target"]==0]["danceability"]
liked.hist(alpha=0.6, bins=50, label="liked")
dis.hist(alpha=0.6, bins=50, label="disliked")
plt.legend()
plt.show()

# plot energy
fig = plt.figure()
plt.title("Energy Target distribution")
liked = data[data["target"]==1]["energy"]
dis = data[data["target"]==0]["energy"]
liked.hist(alpha=0.6, bins=50, label="liked")
dis.hist(alpha=0.6, bins=50, label="disliked")
plt.legend()
plt.show()


# plot liveness, not very informative
fig = plt.figure()
plt.title("Liveness Target distribution")
liked = data[data["target"]==1]["liveness"]
dis = data[data["target"]==0]["liveness"]
liked.hist(alpha=0.6, bins=50, label="liked")
dis.hist(alpha=0.6, bins=50, label="disliked")
plt.legend()
plt.show()

# plot acousticness -- not particularly helpful
fig = plt.figure()
plt.title("Acousticness Target distribution")
liked = data[data["target"]==1]["acousticness"]
dis = data[data["target"]==0]["acousticness"]
liked.hist(alpha=0.6, bins=1000, label="liked")
dis.hist(alpha=0.6, bins=1000, label="disliked")
plt.xlim(0, 0.03)
plt.legend()
plt.show()

# plot key
fig = plt.figure()
plt.title("Key Target distribution")
liked = data[data["target"]==1]["key"]
dis = data[data["target"]==0]["key"]
liked.hist(alpha=0.6, bins=10, label="liked")
dis.hist(alpha=0.6, bins=10, label="disliked")
plt.legend()
plt.show()

# plot valence
fig = plt.figure()
plt.title("Valence Target distribution")
liked = data[data["target"]==1]["valence"]
dis = data[data["target"]==0]["valence"]
liked.hist(alpha=0.6, bins=50, label="liked")
dis.hist(alpha=0.6, bins=50, label="disliked")
plt.legend()
plt.show()

# plot instrumentalness
fig = plt.figure()
plt.title("Instrumentalness Target distribution")
liked = data[data["target"]==1]["instrumentalness"]
dis = data[data["target"]==0]["instrumentalness"]
liked.hist(alpha=0.6, bins=50, label="liked")
dis.hist(alpha=0.6, bins=50, label="disliked")
plt.legend()
plt.show()



# overall heat map
atts = ["acousticness","danceability","duration_ms","energy","instrumentalness","key",
        "liveness","loudness","mode","speechiness","tempo","time_signature","valence","target"]
heatmap_df = data[atts]
corr = heatmap_df.corr()
plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Correlation heatmap")
plt.show()