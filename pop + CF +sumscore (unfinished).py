import sys
import random
import math
import os
from operator import itemgetter

import pandas as pd
import numpy as np
import time
import sqlite3
import datetime
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

triplet_dataset = pd.read_csv(filepath_or_buffer=r'C:\Users\ykang\Desktop\Big data\train_triplets\train_triplets.txt',nrows=10000,sep='\t',header=None,names=['user','song','play_count']) # 数据本身没有表头
triplet_dataset
triplet_dataset.head()

#################
# Plots hiddend #
#################

# How many songs a uer played
output_dict_user = {} #dictionary
with open(r'C:\Users\Jstarry\Desktop\Big data\train_triplets\train_triplets.txt') as f:
    for line_number, line in enumerate(f):
        user = line.split('\t')[0]
        play_count = int(line.split('\t')[2])
        if user in output_dict_user:
            play_count +=output_dict_user[user]
            output_dict_user.update({user:play_count})
        output_dict_user.update({user:play_count})

output_user_list = [{'user':k,'play_count':v} for k,v in output_dict_user.items()]
play_count_df = pd.DataFrame(output_user_list)
play_count_df
play_count_df = play_count_df.sort_values(by = 'play_count', ascending = False)

play_count_df.head()
play_count_df.to_csv(path_or_buf=r'C:\Users\ykang\Desktop\Big data\user_playcount_df.csv', index = False)

# How many times a song played
output_dict_song = {}
with open(r'C:\Users\ykang\Desktop\Big data\train_triplets\train_triplets.txt') as w:
    for line_number, line in enumerate(w):
        song = line.split('\t')[1]
        play_count = int(line.split('\t')[2])
        if song in output_dict_song:
            play_count +=output_dict_song[song]
            output_dict_song.update({song:play_count})
        output_dict_song.update({song:play_count})
output_song_list = [{'song':k,'play_count':v} for k,v in output_dict_song.items()]
song_count_df = pd.DataFrame(output_song_list)
song_count_df = song_count_df.sort_values(by = 'play_count', ascending = False)
song_count_df.head()
song_count_df.to_csv(path_or_buf=r'C:\Users\ykang\Desktop\Big data\song_playcount_df.csv', index = False)

total_play_count = sum(song_count_df.play_count) #歌曲总播放量

print ((float(play_count_df.head(n=100000).play_count.sum())/total_play_count)*100)

# eliminate data: since 80 includes almost all information we need from visualization
print (float(song_count_df.head(n=30000).play_count.sum())/total_play_count*100)
play_count_subset = play_count_df.head(n=100000)
song_count_subset = song_count_df.head(n=30000)
play_count_subset

# target user; target songs
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)

triplet_dataset = pd.read_csv(filepath_or_buffer=r'C:\Users\ykang\Desktop\Big data\train_triplets\train_triplets.txt',sep='\t',header=None,names=['user','song','play_count'])

# 
triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset) ]
del(triplet_dataset)

# 
triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]
del(triplet_dataset_sub)

triplet_dataset_sub_song.info()


# Metadata imported
conn = sqlite3.connect(r'C:\Users\Jstarry\Desktop\Big data\track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()

track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df_sub = track_metadata_df[track_metadata_df.song_id.isin(song_subset)]
track_metadata_df_sub.head()

# merge
del(track_metadata_df_sub['track_id'])
del(track_metadata_df_sub['artist_mbid'])
track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')
triplet_dataset_sub_song_merged.rename(columns={'play_count':'listen_count'},inplace=True)

# fetures we will not include
del(triplet_dataset_sub_song_merged['song_id'])
del(triplet_dataset_sub_song_merged['artist_id'])
del(triplet_dataset_sub_song_merged['duration'])
del(triplet_dataset_sub_song_merged['artist_familiarity'])
del(triplet_dataset_sub_song_merged['artist_hotttnesss'])
del(triplet_dataset_sub_song_merged['track_7digitalid'])
del(triplet_dataset_sub_song_merged['shs_perf'])
del(triplet_dataset_sub_song_merged['shs_work'])
del(triplet_dataset_sub_song_merged['title'])
del(triplet_dataset_sub_song_merged['release'])
del(triplet_dataset_sub_song_merged['artist_name'])
triplet_dataset_sub_song_merged.head()

triplet_dataset_sub_song_merged.to_csv(r'C:\Users\ykang\Desktop\Big data\triplet_dataset_sub_song_merged.csv',encoding='utf-8', index=False)

# popularity
popular_songs = triplet_dataset_sub_song_merged[['title','listen_count']].groupby('title').sum().reset_index()
# popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20)
 
# objects = (list(popular_songs_top_20['title']))
# y_pos = np.arange(len(objects))
# performance = list(popular_songs_top_20['listen_count'])

# popular_songs = triplet_dataset_sub_song_merged[['title','listen_count']].groupby('title').sum().reset_index()
# popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20)
 
# objects = (list(popular_songs_top_20['title']))
# y_pos = np.arange(len(objects))
# performance = list(popular_songs_top_20['listen_count'])
# plt.figure(figsize=(16,8)) 
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical',fontsize=12)
# plt.ylabel(u'播放次数')
# plt.title(u'最流行歌曲')
 
# plt.show()

# 
popular_release = triplet_dataset_sub_song_merged[['release','listen_count']].groupby('release').sum().reset_index()
# popular_release_top_20 = popular_release.sort_values('listen_count', ascending=False).head(n=20)

# objects = (list(popular_release_top_20['release']))
# y_pos = np.arange(len(objects))
# performance = list(popular_release_top_20['listen_count'])
 
# plt.figure(figsize=(16,8)) 
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical',fontsize=12)
# plt.ylabel(u'播放次数')
# plt.title(u'最流行专辑')
 
# plt.show()

# popular artist
popular_artist = triplet_dataset_sub_song_merged[['artist_name','listen_count']].groupby('artist_name').sum().reset_index()
# popular_artist_top_20 = popular_artist.sort_values('listen_count', ascending=False).head(n=20)

# objects = (list(popular_artist_top_20['artist_name']))
# y_pos = np.arange(len(objects))
# performance = list(popular_artist_top_20['listen_count'])
 
# plt.figure(figsize=(16,8)) 
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical',fontsize=12)
# plt.ylabel(u'播放次数')
# plt.title(u'最流行歌手')
 
# plt.show()

# 
user_song_count_distribution = triplet_dataset_sub_song_merged[['user','title']].groupby('user').count().reset_index().sort_values(by='title',ascending = False)
user_song_count_distribution.title.describe()


###############################################################################
#  Item Based Collaborative Filtering 
#  version 1.0 -- need to optimize after similarity score calculated
###############################################################################
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given user
    def get_user_items(self, user): #这个用户所有没听过的歌
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Get unique users for a given item (song)
    def get_item_users(self, item): #没有听过这首歌的所有用户
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #Get unique items (songs) in the training data
    def get_all_items_train_data(self): #删除train data中的重复歌曲
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_songs, all_songs):
            
        ####################################
        #Get users for all songs in user_songs. 每个用户听的所有的歌
        ####################################
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
            
        ###############################################
        #Initialize the item cooccurence matrix of size 
        #len(user_songs) X len(songs) 创造矩阵
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        #############################################################
        #Calculate similarity between user songs and all unique songs
        #in the training data
        #############################################################
        for i in range(0,len(all_songs)):
            #Calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                    
                #Get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]
                    
                #Calculate intersection of listeners of songs i and j #两首歌都听过的人
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index #并交比
                if len(users_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union)) #交集/并集
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])#matrix.shape:第一维0“行”第二维1“列”
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 50:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, user):
        
        ########################################
        #A. Get all unique songs for this user
        ########################################
        user_songs = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        user_songs = item_list
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations

#
from sklearn.model_selection import train_test_split
song_count_subset = song_count_df.head(n=500) # 选择最流行的100首歌
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song.isin(song_subset)]

train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size = 0.30, random_state=0)
is_model = item_similarity_recommender_py()
is_model.create(train_data, 'user', 'song')

user_id = list(train_data.user)[7]
user_id
user_items = is_model.get_user_items(user_id)
user_items
most_like = user_items [0]

is_model.recommend(user_id)


def write_csv(top20):
    file_name = 'top20.csv'
    save = pd.DataFrame(top20)
    try:
        save.to_csv(file_name)
    except UnicodeEncodeError:
        print("No")

write_csv(a)

Top10RecForUser = pd.read_csv(filepath_or_buffer='top20.csv')
print(Top10RecForUser)

song20 = Top10RecForUser["song"]
song20=list(song20)
print(song20)


filter=[".h5"]

song_to_track = pd.read_csv(r'C:\Users\Jstarry\Desktop\Big data\unique_tracks.txt',engine='python',sep="<SEP>")
song_to_track.columns = ["TrackID","SongID","S1","S2"]
print(song_to_track)
y = song_to_track.loc[song_to_track["SongID"].isin(song20)]
print(y["TrackID"])

most_like_trackID = song_to_track.loc[song_to_track["SongID"].isin([most_like])]
most_like_trackID = most_like_trackID ["TrackID"]

def all_path(dirname):
    
    result = [] #所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):
        '''
        print("1:",maindir) #当前主目录
        print("2:",subdir) #当前主目录下的所有目录
        print("3:",file_name_list)  #当前主目录下的所有文件
        '''

        for filename in file_name_list:
            filename = filename.replace(".h5","")

            if filename in y["TrackID"]:
                    #if filename in song20:

                apath = os.path.join(maindir, filename)#合并成一个完整路径
                ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容
                
                result.append(filename)
                '''
                if ext in filter:
                    result.append(apath)
                '''
    return result

file = r"C:\Users\ykang\Desktop\Big data\millionsongsubset_full\MillionSongSubset\data"

print(all_path(file))

type(triplet_dataset_sub_song_merged)
r = track_metadata_df.loc[track_metadata_df["song_id"].isin(song20)]
print (r)
duration = r["duration"]
duration = list(duration)
m = max(duration)
m
n = min (duration)
n
familiarity =r ["artist_familiarity"]
familiarity
hottness = r ["artist_hotttnesss"]
hottness

def MaxMinNormalization(x,M,N):
    for d in x:
        d = (d - N) / (M - N)
    return d
MaxMinNormalization (duration,m,n)
duration_normalized = MaxMinNormalization (duration,m,n)
sum_score = (duration_normalized+familiarity+hottness)/3
type(sum_score) 
sum_score = list(sum_score)


'''
def calculate_sum_score (songRows):

    duration = songRows ["duration"]
    duration
    Max = max (duration)
    #Max
    Min = min (duration)
    #Min
    familiarity =songRows ["artist_familiarity"]
    familiarity
    hottness = songRows ["artist_hotttnesss"]
    hottness

    def MaxMinNormalization(x,Max,Min):
        x = (x - Min) / (Max - Min)
        return x
    MaxMinNormalization (duration,max,min)
    duration_normalized = MaxMinNormalization (duration,max,min)

    sum_score = (duration_normalized+familiarity+hottness)/3
    return sum_score

calculate_sum_score (r)
'''

#计算用户平时最喜欢听的歌的sum score
k = track_metadata_df.loc[track_metadata_df["song_id"].isin([most_like])]
k
k.duration = k ["duration"]
k.duration = 0.5
type(k.duration)

k.familiarity = k ["artist_familiarity"]
k.hottness = k ["artist_hotttnesss"]
k.sum_score = (k.duration+k.familiarity+k.hottness)/3
k.sum_score

# 按照欧式距离 重新拍排序

distance = []
for i in range (50):
    dis = k.sum_score - sum_score[i]
    print (dis)
    distance.append(dis)
print (distance)
len(distance)
list(distance[1])

BL = []
for b in distance:
    b = list(b)
    BL.append(b)

print(BL)
len(BL)
Top10RecForUser ["distance"] = BL
len(Top10RecForUser ["distance"])

Top10RecForUser

######################################################################################
# CF - 2 : 

class ItemBasedCF():
    # 初始化参数
    def __init__(self):
        # 找到相似的20部电影，为目标用户推荐10部电影
        self.n_sim_movie = 20
        self.n_rec_movie = 10
 
        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
 
        # 用户相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0
 
        print('Similar movie number = %d' % self.n_sim_movie)
        print('Recommneded movie number = %d' % self.n_rec_movie)


    # 读文件得到“用户-电影”数据
    def get_dataset(self, filename, pivot=0.875):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if(random.random() < pivot):
                self.trainSet.setdefault(user, {})  #相当于trainSet.get(user)，若该键不存在，则设trainSet[user] = {}，典中典
 
                #键中键：形如{'1': {'1287': '2.0', '1953': '4.0', '2105': '4.0'}, '2': {'10': '4.0', '62': '3.0'}}
                #用户1看了id为1287的电影，打分2.0
                self.trainSet[user][movie] = rating
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)


    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)
 
 
    # 计算电影之间的相似度
    def calc_movie_sim(self):
        for user, movies in self.trainSet.items():  #循环取出一个用户和他看过的电影
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1  #统计每部电影共被看过的次数
 
        self.movie_count = len(self.movie_popular)  #得到电影总数
        print("Total movie number = %d" % self.movie_count)
 
        for user, movies in self.trainSet.items():  #得到矩阵C，C[i][j]表示同时喜欢电影i和j的用户数
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    # self.movie_sim_matrix[m1][m2] += 1  #同时喜欢电影m1和m2的用户+1    21.75  10.5   16.67
                    self.movie_sim_matrix[m1][m2] += 1 /math.log(1 + len(movies)) #ItemCF-IUF改进，惩罚了活跃用户 22.00 10.65 14.98
        print("Build co-rated users matrix success!")
 
        # 计算电影之间的相似性
        print("Calculating movie similarity matrix ...")
        for m1, related_movies in self.movie_sim_matrix.items():    #电影m1，及m1这行对应的电影们
            for m2, count in related_movies.items():    #电影m2 及 同时看了m1和m2的用户数
                # 注意0向量的处理，即某电影的用户数为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    #计算出电影m1和m2的相似度
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('Calculate movie similarity matrix success!')
 
        #添加归一化    precisioin=0.2177	recall=0.1055	coverage=0.1497
        maxDict = {}
        max = 0
        for m1, related_movies in self.movie_sim_matrix.items():
            for m2, _ in related_movies.items():
                if self.movie_sim_matrix[m1][m2] > max:
                    max = self.movie_sim_matrix[m1][m2]
 
        for m1, related_movies in self.movie_sim_matrix.items():    #归一化
            for m2, _ in related_movies.items():
                # self.movie_sim_matrix[m1][m2] = self.movie_sim_matrix[m1][m2] / maxDict[m2]
                self.movie_sim_matrix[m1][m2] = self.movie_sim_matrix[m1][m2] / max
 
 
 
    # 针对目标用户U，找到K部相似的电影，并推荐其N部电影
    def recommend(self, user):
        K = self.n_sim_movie    #找到相似的20部电影
        N = self.n_rec_movie    #为用户推荐10部
        rank = {}
        watched_movies = self.trainSet[user]    #该用户看过的电影
 
        for movie, rating in watched_movies.items():    #遍历用户看过的电影及对其评价
            #找到与movie最相似的K部电影,遍历电影及与movie相似度
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies: #如果用户已经看过了，不推荐了
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * float(rating)    #计算用户对该电影的兴趣
        #返回用户最感兴趣的N部电影
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
 
 
    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluating start ...')
        N = self.n_rec_movie    #要推荐的电影数
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()
 
        for i, user in enumerate(self.trainSet):
            test_moives = self.testSet.get(user, {})    #测试集中用户喜欢的电影
            rec_movies = self.recommend(user)   #得到推荐的电影及计算出的用户对它们的兴趣
 
            for movie, w in rec_movies: #遍历给user推荐的电影
                if movie in test_moives:    #测试集中有该电影
                    hit += 1                #推荐命中+1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_moives)
 
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
 
 
if __name__ == '__main__':
    rating_file = 'C:/Users/Jstarry/Desktop/Big data/triplet_dataset_sub_song_merged.csv'
    itemCF = ItemBasedCF()
    itemCF.get_dataset(rating_file)
    itemCF.calc_movie_sim()
    itemCF.evaluate()# coding = utf-8