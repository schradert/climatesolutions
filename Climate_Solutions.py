""" CDP Unlocking Climate Solutions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

####################
### DATA LOADING ###
####################

cities_dir  =  'Cities/Cities Responses/'
cc_dir  =  'Corporations/Corporations Responses/Climate Change/'
ws_dir  =  'Corporations/Corporations Responses/Water Security/'

cities_dir_disc  =  'Cities/Cities Disclosing/'
cc_dir_disc  =  'Corporations/Corporations Disclosing/Climate Change/'
ws_dir_disc  =  'Corporations/Corporations Disclosing/Water Security/'

"""                          CITIES                  """
city_disclosing  =  pd.read_csv(cities_dir_disc + "2020_Cities_Disclosing_to_CDP.csv")
city_disclosing_DIC  =  pd.read_csv(cities_dir_disc + "Cities_Disclosing_to_CDP_Data_Dictionary.csv")

city_response  =  pd.read_csv(cities_dir + "2020_Full_Cities_Dataset.csv")
city_response_DIC  =  pd.read_csv(cities_dir + "Full_Cities_Response_Data_Dictionary.csv")

"""                         CORPORATE                """
corporate_disclosing_climate  =  pd.read_csv(cc_dir_disc + "2020_Corporates_Disclosing_to_CDP_Climate_Change.csv")
corporate_disclosing_climate_DIC  =  pd.read_csv(cc_dir_disc + "Corporations_Disclosing_to_CDP_Data_Dictionary.csv")
corporate_disclosing_water  =  pd.read_csv(ws_dir_disc + "2020_Corporates_Disclosing_to_CDP_Water_Security.csv")
corporate_disclosing_water_DIC  =  pd.read_csv(ws_dir_disc + "Corporations_Disclosing_to_CDP_Data_Dictionary.csv")

corporate_response_climate  =  pd.read_csv(cc_dir + "2020_Full_Climate_Change_Dataset.csv")
corporate_response_water  =  pd.read_csv(ws_dir + "2020_Full_Water_Security_Dataset.csv")




NUM_Q = 'Question Number'
NUM_C = 'Column Number'
NUM_R = 'Row Number'

NAME_Q = 'Question Name'
NAME_C = 'Column Name'

ANS = 'Response Answer'

# merging unique Q num and question (to improve)
dic_questions = \
    city_response\
        .groupby([NAME_Q, NUM_Q])\
        .size().reset_index(name='Freq')
dic_corporate_water = \
    corporate_response_water\
        .groupby(['data_point_name'])\
        .size().reset_index(name='Freq')

def response(df, Q_num: str):
    """ Return df with all the answer to that question+question name """
    df_q = df[df[NUM_Q] == Q_num]
    Q_name = dic_questions[dic_questions[NUM_Q] == Q_num][NAME_Q].to_list()

    answer = pd.DataFrame(df_q[ANS], columns=[f'Q.nb : {Q_num}  {Q_name}'])
    return pd.concat([df_q[['Country', 'Organization']], answer], axis=1)

def clean_df(df):
    """ Remove rows with null/-ish answer values """
    NULLISH = 'Question not applicable'
    return df[(df[ANS].notna()) & (df[ANS] != NULLISH)]


# Number of unique responses to each question as `occurences`
df = clean_df(city_response)
uniques = dic_questions[NUM_Q].map(lambda num: len(df[df[NUM_Q] == str(num)][ANS].unique()))
occurences = pd.concat([
    dic_questions[[NUM_Q, NAME_Q]], 
    pd.DataFrame(uniques, columns=['# of unique answers'])
], axis=1)



#######################################
### EXPLORATORY DATA ANALYSIS (EDA) ###
#######################################

clean_city = clean_df(city_response)

#---------------------------------------------

""" Natural Hazards (Q2.1) """
nh = clean_city[clean_city[NUM_Q] == '2.1']
nh_q = list(nh[NAME_C].unique())

""" Q(2.1.3) : `Social impact of hazard overall` """
nh_q3 = nh[nh[NAME_C] == nh_q[5]]
uniques = nh_q3[ANS].value_counts()
# Aggregate `Other` categories
others_idx = [i for i in range(len(uniques)) if "Other" in uniques.index[i]]
uniques.drop(uniques.index[others_idx], axis=0, inplace=True)
uniques['Others'] = [uniques.iloc[others_idx,0].sum()]
# PIE CHART
fig1, ax1 = plt.subplots()
ax1.pie(uniques.iloc[:,0], labels=uniques.index, 
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(f'Repartition of answers to Q : {nh_q[5]}')
plt.show()

""" Natural Hazards by zone (in proportion)"""
nh_q1 = nh[nh[NAME_C] == nh_q[0]]
cross = pd.crosstab(nh_q1[ANS], nh_q1['CDP Region'])
ttest = cross.sum(axis = 1)

df = cross
df.index = [idx.split('>')[0] for idx in df.index] 
df = df.groupby(df.index).sum()
df = df / df.sum(axis=0) # normalize

fig, ax = plt.subplots(figsize=(10,5))
df.T.plot(
    kind='bar', ax=ax, colormap=plt.get_cmap('coolwarm_r'), 
    grid=False, title=r"% of each hazards, by continent")
#-------------------------------------------

""" Collaborations """
df = clean_city
collab = df[df['Section'] == "Collaboration"]
collab_description_answers = collab[collab[NAME_C] == "Description"][ANS]

"""Food--- nothing there"""
df = clean_city
df = df[df[NUM_Q] == '12.4']
food = df[df[NUM_R] == 2]
food_uniques = food[ANS].value_counts()

"""water related issus"""
df = clean_city
water = df[df[NUM_Q] == '14.1']
#---------------------------------------------


"""           NOTES
-- Ideas  For City:
    -Check reports published by GIEC/Data provider --> ideas of plot/analyses
    -Go along the questionnary and cluster questions that look good together,
    that could make a nice KPI

    

--natural hazard:
    -Q 2.1 many dependencies (partition is not good enough yet)
    - 10 columns (9/10 are directive)

    - nh_q[6] = expecting evolution of the hazard : 80% increase, nice introduction plot
    
--  KPI Q.3.3



OTHERS : 
    Merge Disclosing - Response?
    
    Select Response by questions which is Directed "please select"/
    
    "Question not Applicable": 
    -%?

    Industry--> Sector--> Activity . Different from Primary In./Primary S./Primary Activity?
Question name = actual question/ then Column NAME/then Row name for particularities
Use of Tableau Free trial for Data Vizualization? hacked version ??

Correlation/plot ideas
stack bar plot "future changes in frequencies/hazard type etc, by continent"



09/11/2020 - 

Focus on the most answered questions??

Use questionnaire for each years, time evolution ??
Food

"""


##################
### WORD CLOUD ###
##################
from wordcloud import WordCloud

questions = pd.DataFrame(clean_city[NAME_Q].unique(), columns=["Question"])
text = ' '.join(questions.tolist())
wordcloud = WordCloud(
    max_font_size=None, 
    background_color='white', 
    collocations=False
).generate(text)
fig = plt.imshow(wordcloud)


####################
### GEO ANALYSIS ###
####################
#import geopandas as gpd ==> establish more secure import!

clean_city = clean_df(city_response)
city_loc = city_disclosing[['City','City Location']]
city_loc.iloc[0,:]


###################################
### NATURAL LANGUAGE PROCESSING ###
###################################
from nltk.corpus import stopwords

Q = 'Question'
stop = stopwords.words('english')

# Lowercase => Remove stopwords => Remove punctuation
questions[Q] =  \
    questions[Q]\
        .map(str.lower)\
        .map(lambda q: " ".join(word for word in q.split() if word not in stop))\
        .str.replace(r'[^\w\s]', '')

# word frequencies
freqs = pd.Series(questions[Q]).value_counts()
commons = freqs[:10]
rares = freqs[-15:]

# ???
df = clean_city[(clean_city[NUM_Q] == '3.3') & (clean_city[NUM_C] == 7)]
answer = df[ANS]

