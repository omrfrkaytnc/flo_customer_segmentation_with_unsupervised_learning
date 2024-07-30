
###############################################################
# Customer Segmentation with Unsupervised Learning / Gözetimsiz Öğrenme ile Müşteri Segmentasyonu
###############################################################

###############################################################
# Business Problem / İş Problemi
###############################################################
# Customers will be segmented into clusters using unsupervised learning methods (K-means and hierarchical clustering) to observe their behaviors.
# Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering ) müşteriler kümelere ayrılarak davranışları gözlemlenmek istenmektedir.

###############################################################
# Data Set Story / Veri Seti Hikayesi
###############################################################

# The data set consists of information obtained from the past shopping behaviour of customers who made their last purchases as OmniChannel (both online and offline) in 2020 - 2021.
# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline) olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# 20.000 gözlem,
# 12 değişken

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı
#                 (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the last purchase was made / En son alışverişin yapıldığı kanal
# first_order_date : Date of the customer's first purchase / Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Date of the customer's last purchase / Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : The date of the last purchase made by the customer on the online platform / Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Date of the last purchase made by the customer on the offline platform / Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Total number of purchases made by the customer on the online platform / Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Total number of offline purchases made by the customer / Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Total price paid by the customer for offline shopping / Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Total price paid by the customer for online shopping / Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : List of categories in which the customer has shopped in the last 12 months / Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi





###############################################################
# Reading the data set and selecting the variables you will use when segmenting customers / Veri setini okutma ve müşterileri segmentlerken kullanıcağınız değişkenleri seçme
###############################################################
# Import necessary libraries / Gerekli kütüphanelerin importu
import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter(action="ignore")
import matplotlib
matplotlib.use("Qt5Agg")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

# first look at the data / veriye ilk bakış
df.head()
df.shape
# (19945, 12)

df.isnull().sum()
# no missing value / eksik değer yok

df.info()

df.describe().T


# converting to date variable / tarih değişkenine çevirme
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

df["last_order_date"].max()
#2021-05-30

analysis_date = dt.datetime(2021,6,1)


#recency: time since last purchase / son alışverişten bu yana geçen süre
df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')


#tenure: age of the customer / müşterinin yaşı
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')

df.head()

# let's select numerical variables for model df / model df için sayısal değişkenleri seçelim
# (We could also use the approach in feature eng. / feature eng. kısmındaki yaklaşımı da kullanabilirdik)

model_df = df[["order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]

model_df.head()
model_df.info()


###############################################################
# Customer Segmentation with K-Means / K-Means ile Müşteri Segmentasyonu
###############################################################

# 1. Standardising variables / Değişkenleri standartlaştırma.
# SKEWNESS
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)


# Log transformation to ensure normal distribution / Normal dağılımın sağlanması için Log transformation uygulanması

model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency']=np.log1p(model_df['recency'])
model_df['tenure']=np.log1p(model_df['tenure'])
model_df.head()


# Scaling / Ölçeklendirme
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()


# 2. Determining the optimum number of clusters / Optimum küme sayısını belirleme
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show(block=True)


# Model building and segmenting customers / Model oluşturma ve müşterileri segmentleme
k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments = k_means.labels_
segments

final_df = df[["master_id",
               "order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]
final_df["segment"] = segments
final_df.head()

final_df.describe().T

# segments 0-6, if we want to do 1-7 / segmentler 0-6, biz 1-7 arası yapmak istersek

final_df["segment"] = final_df["segment"] + 1

# Statistical analysis of each segment / Herbir segmenti istatistiksel olarak inceleme

final_df.groupby("segment").agg(["mean","min","max", "count"])


# final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
#                                   "order_num_total_ever_offline":["mean","min","max"],
#                                   "customer_value_total_ever_offline":["mean","min","max"],
#                                   "customer_value_total_ever_online":["mean","min","max"],
#                                   "recency":["mean","min","max"],
#                                   "tenure":["mean","min","max","count"]})



###############################################################
# Customer Segmentation with Hierarchical Clustering / Hierarchical Clustering ile Müşteri Segmentasyonu
###############################################################

# Determine the optimum number of clusters using our standardised dataframe / Standarlaştırdığımız dataframe'i kullanarak optimum küme sayısını belirleme

model_df.info()
hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show(block=True)


# Model building and segmenting customers / Model oluşturma ve müşterileri segmentleme
hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df_h = df[["master_id","order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]
final_df_h["segment"] = segments
final_df_h.head()

# Statistical analysis of each segment / Herbir segmenti istatistiksel olarak inceleme

final_df_h.groupby("segment").agg(["mean","min","max", "count"])
