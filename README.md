# 19_bigcontest
Bigcontest Innovation 
*team : 미세베이션*

----------------------------------------
**1. 환경기상데이터 전처리**
- 패키지 설치
```
import glob
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import copy
```
- 노원,종로구 데이터 합치기
```
## 노원, 종로구 데이터 합치기
# 아래 파일 경로 안에 대회에서 제공한 환경기상데이터 csv 파일들이 다 들어있어야 함.
path =r'C:\Users\korea\practice\환경기상데이터' 
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
finedust= []
for file_ in allFiles:
    data_frame=pd.read_csv(file_,index_col=None)
    finedust.append(data_frame)
    #list_.append(finedust)
finedust_concat=pd.concat(finedust,axis=0,ignore_index=True)


## 노원, 종로구에 구, 동 변수 추가. dong= 각 파일마다 변수
# 우선, 측정기번호, 측정기위치, 행정구, 행정동 변수가 있는 파일을 가져오자.
# 환경기상데이터 csv파일과 다른 경로에 있어야 함. 
envi_gu_dong=pd.read_csv(r'C:\Users\korea\practice\environment_gu_dong.csv',encoding='utf-8')

#그리고 serial을 기준으로 merge하자.
finedust=pd.merge(finedust_concat,envi_gu_dong,on='serial')

# 이상한 변수 Unnamed: 10 삭제
finedust = finedust.drop('Unnamed: 10',axis=1)
```

- 결측치, 극단치 제어 및 각종 전처리
```
# 실내 측정기 있는지 확인. 아예 없으면, co2와 vocs는 삭제 가능
# flag, co2, vocs 삭제 근거 찾기
pd.value_counts(finedust['flag'].values, sort=False)
# 실내측정기(flag)는 없는 것 같다. 없거나 -999. 

# 실내에서만 측정되는 co2와 vocs의 빈도도 확인해보자.
pd.value_counts(finedust['co2'].values, sort=False)
# 역시 삭제하자.

pd.value_counts(finedust['vocs'].values, sort=False)
# 이 변수도 삭제하자.

# 위에서 언급한 3개 변수(flag,co2,vocs) 삭제
finedust=finedust.drop(['flag','co2','vocs'],axis=1)

## 본격적으로 결측치와 극단치 제어.
# 먼저, 도렴동(V01o1610468)의 관측치를 제거하자. 도렴동 csv 파일을 열어보니 데이터가 모두 결측.
# register_date 필요 없음. 2018년 4월 이전에 다 등록. 해당 변수 삭제. 
finedust = finedust[finedust.serial!='V01o1610468']
finedust = finedust.drop('register_date',axis=1)

# 음의 극단치들 모두 결측으로 처리하기
finedust1 = copy.deepcopy(finedust)

finedust1.pm10.loc[finedust1.pm10<0] = None
finedust1.noise.loc[finedust1.noise<0] = None
finedust1.temp.loc[finedust1.temp<-40] = None
finedust1.humi.loc[finedust1.humi<0] = None
finedust1.pm25.loc[finedust1.pm25<0] = None

# 양의 극단치 pm25 없애주기. IQR*3 + Q3 기준. 
# 참고로 온도, 습도, 바람은 양의 극단치 발견 X
#np.percentile(x, 75)  # 3사분위 수.
# boxplot에서 최대값은 3분위수 + 1.5 * IQR
finedust2=finedust1.reset_index(inplace=False)

# pm25의 3분위수를 계산하자.
finedust3_pm25 = finedust2['pm25'].dropna() # 결측이 있으면 percentile 못 쓴다. 
# 계산비용으로 인해 IQR * 3 + Q3 적용. 
iqr_pm25 = (np.percentile(finedust3_pm25,75)-np.percentile(finedust3_pm25,25))*3 + np.percentile(finedust3_pm25,75) 
[iqr_pm25,sum(finedust3_pm25>=iqr_pm25)] 
# 83 초과는 내 검사기에 들어갈 것이고 대략 120만개.. 

pm25_outlier=finedust3_pm25[finedust3_pm25>=iqr_pm25] # iqr max 넘은 것들만 거르기
pm25_outlier_index= pm25_outlier.index.tolist() # 인덱스들을 리스트로 만들고.. 
[iqr_pm25,sum(finedust3_pm25>=iqr_pm25)] 

# 관측치의 앞 5개 관측치보다 너무 다르면, 결측으로 처리하는 코드
for i in pm25_outlier_index:
    bound=range(i-5,i)
    before_pm25=finedust2.iloc[list(bound),6]
    if before_pm25.isnull().all()==True: # i번째 전 5개 관측이 모두 결측이면 넘어가기
        continue
    if max(filter(lambda v: v is not None, before_pm25))*10 <= finedust2.pm25[i]:
        finedust2.pm25[i] = None # 6번째가 pm25. 앞 5개의 최대값 * 10이 지금 값보다 작다면 그건 양의 결측치

###양의 극단치 pm10 없애주기. IQR*3 + Q3

finedust3_pm10 = finedust2['pm10'].dropna() # 결측이 있으면 percentile 못 쓴다. 
iqr_pm10 = (np.percentile(finedust3_pm10,75)-np.percentile(finedust3_pm10,25))*4 + np.percentile(finedust3_pm10,75) 
# 계산비용으로 인해 IQR * 3 + Q3 적용. 
[iqr_pm10,sum(finedust3_pm10>=iqr_pm10)] 
# 83 초과는 내 검사기에 들어갈 것이고 대략 120만개.. 

pm10_outlier=finedust3_pm10[finedust3_pm10>=iqr_pm10] # iqr max 넘은 것들만 거르기
pm10_outlier_index= pm10_outlier.index.tolist() # 인덱스들을 리스트로 만들고.. 
[iqr_pm10,sum(finedust3_pm10>=iqr_pm10)] 

for i in pm10_outlier_index:
    bound=range(i-5,i)
    before_pm10=finedust2.iloc[list(bound),6]
    if before_pm10.isnull().all()==True: # i번째 전 5개 관측이 모두 결측이면 넘어가기
        continue
    if max(filter(lambda v: v is not None, before_pm10))*10 <= finedust2.pm10[i]:
        finedust2.pm10[i] = None # 6번째가 pm10. 앞 5개의 최대값 * 10이 지금 값보다 작다면 그건 양의 결측치

# 영천동 277(천연동) -> 교남동 , 태평로1가 68-2(명동) -> 종로 1,2,3,4가동
finedust2.dong[finedust2.spot=='영천동 277']='교남동'
finedust2.dong[finedust2.spot=='태평로1가 68-2']='종로 1,2,3,4가동'

finedust2.isnull().sum()
# isnull().all()로 했을 때. 결측이 약 1만개, 2만개 증가.

# 행정동 띄어쓰기 없애기
finedust2.set_index('index',inplace=True)
finedust2.dong.loc[finedust2.dong=='종로 1,2,3,4가동'] = '종로1,2,3,4가동' # 행정동에 공백 없애기
finedust2.head()

## 이제, 주요변수인 pm10과 pm25부터 분포를 살펴보자.
# 2가지 주요변수는 숫자형임을 주목하자. 
numerical_feature=['pm10','pm25']

for col in numerical_feature:
    sns.distplot(finedust2.loc[finedust2[col].notnull(), col])
    plt.title(col)
    plt.show()
# 두 변수 모두 극단치가 많은 분포. 오른쪽으로 꼬리가 길다.

# 세종로 100 관측소가 여러 개여서 평균을 통해서 하나로 합치기.
finedust_temp = finedust2.loc[finedust2.spot=='세종로 100']
grouped1=finedust_temp['pm25'].groupby(finedust_temp['tm'])
grouped2=finedust_temp['pm10'].groupby(finedust_temp['tm'])
grouped3=finedust_temp['noise'].groupby(finedust_temp['tm'])
grouped4=finedust_temp['temp'].groupby(finedust_temp['tm'])
grouped5=finedust_temp['humi'].groupby(finedust_temp['tm'])

# 그룹화한 것들을 평균 때리자.
grouped1.mean()
grouped2.mean()
grouped3.mean()
grouped4.mean()
grouped5.mean()

# 바로 위의 시리즈들을 합쳐서 데이터프레임으로 만들자
grouped_temp = pd.concat([grouped1.mean(),grouped2.mean(),grouped3.mean(),grouped4.mean(),grouped5.mean()],axis=1)

# tm 변수 살리고, spot 변수 추가 
grouped_temp=grouped_temp.reset_index()
grouped_temp['spot'] = '세종로 100'
grouped_temp['dong'] = '종로1,2,3,4가동'
grouped_temp['gu'] = 0
grouped_temp['serial'] = 'sejong'

## 이제 finedust2에서 세종로 100인 것을 삭제하고 grouped_temp를 가져오자.
finedust3 = finedust2.loc[finedust2.spot != '세종로 100',:]

finedust3 = pd.concat([finedust3,grouped_temp],ignore_index=True, sort=False)

# 관측소들을 행정동 기준으로 합치기 
############## 관측소들을 행정동 기준으로 합치기#################
#################################################################
finedust4 = finedust3.iloc[0:2,:] # 한 행 데이터프레임에 변수명 가져오는 효율적인 방법. 나중엔 1~2행 지워줘야! 
finedust4 = finedust4.drop(['serial','spot'],axis=1) #

# 동 리스트 
dong_list = finedust3.dong.value_counts().index.tolist()

# 동별로 합치기. 
for i in dong_list:
    finedust_temp = finedust3.loc[finedust3.dong == i]
    grouped1=finedust_temp['pm25'].groupby(finedust_temp['tm'])
    grouped2=finedust_temp['pm10'].groupby(finedust_temp['tm'])
    grouped3=finedust_temp['noise'].groupby(finedust_temp['tm'])
    grouped4=finedust_temp['temp'].groupby(finedust_temp['tm'])
    grouped5=finedust_temp['humi'].groupby(finedust_temp['tm'])
    grouped6=finedust_temp['gu'].groupby(finedust_temp['tm'])
   
    # 그룹화한 것들을 평균 때리자.
    grouped1.mean()
    grouped2.mean()
    grouped3.mean()
    grouped4.mean()
    grouped5.mean()
    grouped6.mean()
    
    # 바로 위의 시리즈들을 합쳐서 데이터프레임으로 만들자
    grouped_temp = pd.concat([grouped1.mean(),grouped2.mean(),grouped3.mean(),grouped4.mean(),grouped5.mean(),grouped6.mean()],axis=1)
    
    # tm 변수 살리고, dong 변수 추가 
    grouped_temp=grouped_temp.reset_index()
    grouped_temp['dong'] = i
    finedust4 = pd.concat([finedust4,grouped_temp],axis=0,ignore_index=True)

# for 루프 다 돌리고 1,2행 삭제
finedust4 = finedust4.iloc[2:finedust4.shape[0],:]

# 없는 행정동 보정하기: 주변 측정소의 평균 or 대치
# 우선 없는 행정동과 spot에 대한 딕셔너리를 만들자
update_dong = {'숭인1동':['창신동 17-9','숭인동 227-2'],
              '창신2동':['창신동 17-9','창신동 170-3'],
              '상계8동':['상계동 1102-17','상계동 692-2'],
              '상계9동':['상계동 397-12','상계동 456-16','상계2동 407-39'],
              '월계2동':['하계동 280','월계동 411-53'],
              '중계1동':['중계동 364-17','상계동 1281'],
              '중계4동':['중계동 364-17','상계동 1281','상계5동 156-203'],
              '하계2동':['하계동 280','공릉동 385-5'],
               '삼청동':['세종로 1-57','가회동 177-18','재동 95-2'],
               '무악동': ['영천동 277']
              }

# 동별로 합치기.
finedust6 = copy.deepcopy(finedust3)

for key,val in update_dong.items():
    finedust_temp = finedust6.loc[finedust6['spot'].apply(lambda x: x in val)]
    grouped1=finedust_temp['pm25'].groupby(finedust_temp['tm'])
    grouped2=finedust_temp['pm10'].groupby(finedust_temp['tm'])
    grouped3=finedust_temp['noise'].groupby(finedust_temp['tm'])
    grouped4=finedust_temp['temp'].groupby(finedust_temp['tm'])
    grouped5=finedust_temp['humi'].groupby(finedust_temp['tm'])
    grouped6=finedust_temp['gu'].groupby(finedust_temp['tm'])
   
    # 그룹화한 것들을 평균 때리자.
    grouped1.mean()
    grouped2.mean()
    grouped3.mean()
    grouped4.mean()
    grouped5.mean()
    grouped6.mean()
    
    # 바로 위의 시리즈들을 합쳐서 데이터프레임으로 만들자
    grouped_temp = pd.concat([grouped1.mean(),grouped2.mean(),grouped3.mean(),grouped4.mean(),grouped5.mean(),grouped6.mean()],axis=1)
    
    # tm 변수 살리고, dong 변수 추가 
    grouped_temp=grouped_temp.reset_index()
    grouped_temp['dong'] = key
    finedust6 = pd.concat([finedust6,grouped_temp],axis=0,ignore_index=True)

finedust6 = finedust6.drop(['serial','spot'],axis=1) # serial, spot 변수 삭제 
finedust4 = pd.concat([finedust4,finedust6],axis=0) # 기존 행정동 + 없던 행정동 

finedust4.head()
finedust4.to_csv('finedust4.csv')

# 날짜화 시킬 때 메모리에러 뜨기에 저장하고 껐다 켜서 시작하기
finedust4 = pd.read_csv('finedust4.csv')
finedust4 = finedust4.drop('Unnamed: 0',axis=1)
finedust4.head()

# tm 변수를 날짜화 시키고 인덱스로 만들어서 월별, 일별, 시간별 데이터 만들기
finedust5= copy.deepcopy(finedust4)
finedust5['tm']= finedust5['tm'].apply(str)
finedust5['tm'] = pd.to_datetime(finedust5['tm'])
finedust5.set_index('tm',inplace=True)

# finedust3 가져오기. 
finedust3 = pd.read_csv('finedust3.csv',encoding='euc-kr')
finedust3 = finedust3.drop('Unnamed: 0',axis=1)

finedust3.head(1)
finedust_month.head(1)

```

- 월별, 일별, 시간별 미세먼지 데이터 생성
```

############## 월별로 합치기 #################
#################################################################

finedust_month = finedust3.iloc[0:2,:] # 한 행 데이터프레임에 변수명 가져오는 효율적인 방법. 나중엔 1~2행 지워줘야! 
finedust_month = finedust_month.drop(['serial','spot'],axis=1) #

# 동 리스트 
dong_list = finedust4.dong.value_counts().index.tolist()
dong_list

# 우선, 월별로 합쳐보자.
for i in dong_list:
    finedust_temp = copy.deepcopy(finedust5)
    finedust_temp = finedust_temp.loc[finedust_temp.dong == i]
    finedust_temp = finedust_temp.resample(rule='M').mean()
    
    # 동 변수 넣어주기
    finedust_temp['dong'] = i
    finedust_month = pd.concat([finedust_month,finedust_temp],axis=0,ignore_index=False)

# for 루프 다 돌리고 1,2행 삭제
finedust_month = finedust_month.iloc[2:finedust_month.shape[0],:]
finedust_month = finedust_month.drop(['tm'],axis=1)


############## 일별로 합치기#################
#################################################################
finedust_day = finedust3.iloc[0:2,:] # 한 행 데이터프레임에 변수명 가져오는 효율적인 방법. 나중엔 1~2행 지워줘야! 
finedust_day = finedust_day.drop(['serial','spot'],axis=1) #

# 동별로 합치기. 
for i in dong_list:
    finedust_temp = copy.deepcopy(finedust5)
    finedust_temp = finedust_temp.loc[finedust_temp.dong == i]
    finedust_temp = finedust_temp.resample(rule='D').mean()
    
    # 동 변수 넣어주기
    finedust_temp['dong'] = i
    finedust_day = pd.concat([finedust_day,finedust_temp],axis=0,ignore_index=False)

# for 루프 다 돌리고 1,2행 삭제
finedust_day = finedust_day.iloc[2:finedust_day.shape[0],:]
finedust_day = finedust_day.drop(['tm'],axis=1)


############## 시간별로 합치기#################
#################################################################
finedust_hour = finedust3.iloc[0:2,:] # 한 행 데이터프레임에 변수명 가져오는 효율적인 방법. 나중엔 1~2행 지워줘야! 
finedust_hour = finedust_hour.drop(['serial','spot'],axis=1) #

# 동별로 합치기. 
for i in dong_list:
    finedust_temp = copy.deepcopy(finedust5)
    finedust_temp = finedust_temp.loc[finedust_temp.dong == i]
    finedust_temp = finedust_temp.resample(rule='H').mean()
    
    # 동 변수 넣어주기
    finedust_temp['dong'] = i
    finedust_hour = pd.concat([finedust_hour,finedust_temp],axis=0,ignore_index=False)

# for 루프 다 돌리고 1,2행 삭제
finedust_hour = finedust_hour.iloc[2:finedust_hour.shape[0],:]
finedust_hour = finedust_hour.drop(['tm'],axis=1)

# index로 들어간 tm을 살리고
finedust_month=finedust_month.reset_index()
finedust_day=finedust_day.reset_index()
finedust_hour=finedust_hour.reset_index()

# index라는 이름을 tm으로 바꾸고. 
finedust_month.rename(columns = {'index':'tm'},inplace=True)
finedust_day.rename(columns = {'index':'tm'},inplace=True)
finedust_hour.rename(columns = {'index':'tm'},inplace=True)

finedust_month.head(1)

### 요일 변수 추가 . 0: 월요일~ 6: 일요일
finedust_day['yoil'] = finedust_day.tm.apply(lambda x : x.weekday())
finedust_hour['yoil'] = finedust_hour.tm.apply(lambda x : x.weekday())

# finedust2.dong.loc[finedust2.dong=='종로 1,2,3,4가동'] = '종로1,2,3,4가동'
# finedust_temp = finedust6.loc[finedust6['spot'].apply(lambda x: x in val)]
## 공휴일 변수 추가. 0이면 평일, 1이면 공휴일(공휴일,토,일). 
# finedust_day['weekend'] = 0
holiday = ['2018-05-01','2018-05-07','2018-05-22','2018-06-06','2018-06-13','2018-08-15',
      '2018-09-24','2018-09-25','2018-09-26','2018-10-03','2018-10-09','2018-12-25',
      '2019-01-01','2019-02-04','2019-02-05','2019-02-06','2019-03-01']
holiday = pd.to_datetime(holiday)

finedust_day['weekend'] = finedust_day['tm'].apply(lambda x: 1 if (x.weekday() >= 5 ) or (x in holiday) else 0)
finedust_hour['weekend'] = finedust_hour['tm'].apply(lambda x: 1 if (x.weekday() >= 5) or (x.date() in holiday) else 0)

####### 미세먼지 등급별로 만들기. 0,1,2,3으로 좋음 보통 나쁨 매우나쁨 나눔. 
# 월별인데 미세먼지 등급기준을 한국 기준으로 그냥 띡 하는게 맞나..? 
finedust_month['pm25_class'] = finedust_month['pm25'].apply(lambda x : 3 if x >= 75
                                                           else 2 if x >=35
                                                           else 1 if x >=15
                                                           else 0 if x >=0
                                                           else None)
finedust_month['pm10_class'] = finedust_month['pm10'].apply(lambda x : 3 if x >= 150
                                                           else 2 if x >=80
                                                           else 1 if x >=30
                                                           else 0 if x >=0
                                                           else None)
# 일별 데이터
finedust_day['pm25_class'] = finedust_day['pm25'].apply(lambda x : 3 if x >= 75
                                                           else 2 if x >=35
                                                           else 1 if x >=15
                                                           else 0 if x >=0
                                                           else None)
finedust_day['pm10_class'] = finedust_day['pm10'].apply(lambda x : 3 if x >= 150
                                                           else 2 if x >=80
                                                           else 1 if x >=30
                                                           else 0 if x >=0
                                                           else None)
# 시간별 데이터 
finedust_hour['pm25_class'] = finedust_hour['pm25'].apply(lambda x : 3 if x >= 75
                                                           else 2 if x >=35
                                                           else 1 if x >=15
                                                           else 0 if x >=0
                                                           else None)
finedust_hour['pm10_class'] = finedust_hour['pm10'].apply(lambda x : 3 if x >= 150
                                                           else 2 if x >=80
                                                           else 1 if x >=30
                                                           else 0 if x >=0
                                                           else None)

#### 동 공백 제거, 없는 동 넣기, 요일/주말 변수, 미세먼지 등급 기준
# 저장하기
finedust_month.to_csv('finedust_month1.csv',encoding='euc-kr')
finedust_day.to_csv('finedust_day1.csv',encoding='euc-kr')
finedust_hour.to_csv('finedust_hour1.csv',encoding='euc-kr')
```

----------------------------------------
**2. SNS 분석**
**2.  SNS 분석**
- 미세먼지 관련 글수 세기
```
# import 와 파일 로드 (1-8 바꿔가면서 로드)
import pandas as pd
df1 = pd.read_excel("../data/SNS_8.xlsx")
df1.head()

# 결측값을 제거하고 인덱스를 다시 매긴다음 data, title, content column 만
df2 = df1.dropna()
df2 = df2.reset_index()
df2 = df2[["DATE","TITLE","CONTENT"]]

# 새로운 dataframe 생성 
# 일자별 미세먼지언급량, 총글수 담을 용도
daily = pd.DataFrame(columns=[['date','cnt_fine','cnt']])

index=0

# cnt1 : 총글수 cnt2 : 미세먼지관련 글수 basket: 총글수 담는 용도
cnt1=0
basket=[]
cnt2=0
# date 별로 뽑기위한 이중포문
# 날짜는 계속 변경해가면서 data 수집
for j in range(1,31):
    if j<10:
        date = '2018040' + str(j)
    else: 
        date = '201804'+ str(j)
#date = '20180403'
#cnt=0
    for i in range (len(df2)):
        if (str(df2.iloc[i][0]).count(date))==1:
            cnt1+=1
            string= df2.iloc[i][1]
            string2 = df2.loc[i][2]
            cnt2 += string.count("미세먼지")
            cnt2 += string2.count("미세먼지")
    basket.append(cnt1)
    daily.loc[index] = [date, cnt2,cnt1]
    index += 1
    cnt2=0
    cnt1=0

df3 = pd.read_csv("../data/201804.csv")
df3.head()

# 미세먼지 언급량 list에 넣기 (df 추가 위해서)
add_list = []
for k in range(len(daily)):
    adding = int(daily["cnt_fine"].iloc[k])
    add_list.append(adding)

df3["cnt8_fine"] = add_list
df3["cnt8"] = basket

# 미세먼지언급량/총글수 변수 생성
new=[]
for i in range(len(add_list)):
    if basket[i]==0:
        new.append(0)
    else:
        new.append(add_list[i]/basket[i])

df3["new8"] = new
df3.head()
df3.to_csv("../data/201804.csv",index=False)
```

-  sns 자연어처리
```
import codecs
from konlpy.tag import Twitter
from gensim.models import word2vec
# konlpy twitter 사용
import pandas as pd

df1 = pd.read_csv("../data/antiDust.csv",encoding="euc-kr")
df1.head()

#df1 = df1[["TITLE","CONTENT"]]
df1 = df1["content"]
df1.head()

df1.isnull().sum()
len(df1)

df2 = df1.dropna()
df2 = df2.reset_index()
#df2 = df2[["TITLE","CONTENT"]]
df2.head()

# 한글만 추출
import re
for i in range(len(df2)):
    #s = df2["TITLE"].iloc[i]
    s2 = df2["content"].iloc[i]
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
   # result = hangul.sub('',str(s))
    result2 = hangul.sub('',str(s2))
   # df2["TITLE"].iloc[i] = result
    df2["content"].iloc[i] = result2

df2.head()
df2.to_csv("../data/antiDust_clean.csv",index=False,encoding="euc-kr")
```

- sns 1-8 data 너무 방대하니, 랜덤으로 추출
```
from random import *
import pandas as pd

#skiprows를 활용해, 각 데이터 마다 랜덤으로 추출
p = 0.01
df1 = pd.read_csv("../data/SNS_1_clean.csv", skiprows=lambda i: i > 0 and random() > p)
len(df1)

# dataframe 만들어서, 랜덤으로 뽑힌 데이터 저장
new = pd.DataFrame(columns=[['title','content']])
tmp = len(new)
tmp
for i in range(len(df1)):
    t = df1["TITLE"].iloc[i]
    c = df1["CONTENT"].iloc[i]
    new.loc[tmp+i] = [t,c]
print(len(new))
new.head()
new.to_csv("../data/SNS_random_1.csv",encoding="euc-kr",index=False)
```

- 워드클라우드 
```
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# sns 별 단어 빈도수를 기반으로 새로운 txt 작성
f = open("../data/wordCloud.txt","w")

# 단어 빈도수를 기반으로 twitter noun 추출하는 시간을 아끼기위해
# 빈도수를 비율로 따져서 명사만 있는 파일 작성
for i in range(2000):
    if (i<400):
        a = "미세먼지 피부 사용 기능 제품 아이 시간 관리 생각 청소 효과 우리 정말 오늘 제거 추천 사진 가격 사람 얼굴 하나 마스크 요즘 공기 케어 성분 필터 차단 환경 건강 서울 공기청정기 판매 환기 학교 창업 농도\n"
        f.write(a)
    elif (i<600):
        a = "미세먼지 피부 사용 기능 제품 아이 시간 관리 생각 청소 효과 우리 정말 오늘 제거 추천 사진 가격 사람 얼굴 마스크  공기 케어 성분 필터 차단 환경 건강 서울 공기청정기 판매 환기 학교 \n"
        f.write(a)
    elif (i<800):
        a="미세먼지 피부 사용 기능 제품 아이 관리 생각 청소 효과 우리 제거 추천 가격 사람 얼굴 마스크  공기 케어 성분  차단 환경 건강 서울 공기청정기 판매 환기 \n"
        f.write(a)
    elif (i<1000):
        a= "미세먼지 피부 사용 제품 아이 관리 생각 효과 제거 추천 가격 사람 얼굴 마스크  공기 케어 성분  차단 환경 건강 공기청정기 판매 환기\n"
        f.write(a)
    elif (i<1500):
        a = "미세먼지 피부 제품 아이 제거  마스크  공기 케어 차단 환경 건강 공기청정기\n"
        f.write(a)
    else:
        a = "미세먼지 피부 \n"
        f.write(a)
f.close()
text = open("../data/wordCloud.txt",'rt', encoding='UTF8').read()

from konlpy.tag import Twitter
engin = Twitter()
nouns = engin.nouns(text)
nouns = [n for n in nouns if len(n)>1]
count = Counter(nouns)
tags = count.most_common(50)

import numpy as np
from PIL import Image

coloring = np.array(Image.open("data/cloud.png"))
from wordcloud import ImageColorGenerator
image_colors = ImageColorGenerator(coloring)

plt.figure(figsize=(11,11))
plt.imshow(coloring, interpolation='bilinear',cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# 이미지에 맞춰 워드클라우드 제작
wordcloud1 = WordCloud(font_path="data/NanumSquare_acB.ttf",background_color="white",mask=coloring,relative_scaling=0.1).generate_from_frequencies(dict(tags))

plt.figure(figsize=(12,12))
plt.imshow(wordcloud1.recolor(color_func=image_colors), interpolation='bilinear')
plt.axis("off")
plt.show()
```

- gensim 을 활용한 연관도 분석
```
import codecs
from konlpy.tag import Twitter
from gensim.models import word2vec
import pandas as pd

df1 = pd.read_csv("../data/SNS_random.csv", encoding="euc-kr")
df1.head()
len(df1)
df1.isnull().sum()
df2 = df1.dropna()
df2 = df2.reset_index()
#df2 = df2[["title","content"]]
df2 = df2["content"]
df2.head()
len(df2)

# twitter를 사용하고, 품사태깅을 통해 자연어 처리
twitter = Twitter()
results = []
for i in range (len(df2)):
    line = df2.iloc[i]
    malist = twitter.pos(line, norm=True, stem= True)
    r = []
    for word in malist:
        if not word[1] in ["Josa", "Eomi","PreEomi","Exclamation","Adverb","Verb","Conjunction","Determiner", "Punctuation","KoreanParticle"] and len(word[0])>1 and word[0]!="로부터" and word[0]!="스트" and word[0]!="있다" and word[0]!="같다":        
            r.append(word[0])
    rl = (" ".join(r)).strip()
    # strip : 좌우공백 없애기
    results.append(rl)
    #print(rl)

wakati_file = 'data/mise_random_1.wakati'
with open(wakati_file, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(results))

# 앞뒤로 70개를 보며, 2000 번 반복학습하고 쿼드코어 사용
data = word2vec.LineSentence(wakati_file)
model = word2vec.Word2Vec(data,
                          size = 100, window=70, hs=1, min_count=40, sg=1, workers=4, iter=2000)

model.save("data/mise_random_1.model")
print(model)

from gensim.models import word2vec
model = word2vec.Word2Vec.load("data/mise_random_1.model")

#연관있는 단어 10개 추출
model.most_similar(positive=["미세먼지"],topn=10)
```

- 단어별 빈도수 추출
```
import pandas as pd
df1 = pd.read_excel("../data/SNS_1.xlsx")
df1.head()
df1 = df1[["TITLE","CONTENT"]]
df1.head()
# null 확인 (konlpy 쓸때 error 발생)
df1.isnull().sum()
df1_title = df1["TITLE"]
df1_content = df1["CONTENT"]
#len(df1_title) #1561112
df1_title.dropna()
df1_title.reset_index()
df1_content.dropna()
df1_content.reset_index()
#df1_content.head()

from konlpy.tag import Twitter
from collections import Counter
sns2 = Twitter()

def get_dict(text):
    spliter=Twitter()
    nounss = spliter.nouns(text)
    count= Counter(nounss)
    return_list = []
    for n,c in count.most_common():
        tmp = {'tag':n,'count':c}
        return_list.append(tmp)
    return return_list

text_file_name="../data/sns1_contet_final.txt"
noun_count= 0
output_file_name = "../data/sns1_content.txt"
open_text_file = open(text_file_name,'r')
text = open_text_file.read()
tags = get_dict(text)
open_text_file.close()

open_output_file = open(output_file_name,'w')
for tag in tags:
    noun = tag['tag']
    count = tag['count']
    if len(noun)>1:
        open_output_file.write('{} {}\n'.format(noun,count))
        noun_count = noun_count+1
        if noun_count == 50:
            break
open_output_file.close()
```

---------------------
**3. 카드매출**
- 전처리
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

card_df = pd.read_csv(r'C:\Users\ejr93\Desktop\빅콘\가공데이터\card_df.txt', sep = '\t')
card_df.head()

card_df['STD_DD'] = card_df['STD_DD'].astype(str)
card_df['STD_DD'] = card_df['STD_DD'].apply(lambda x:x[0:4]+'-'+x[4:6]+'-'+x[6:8] )
card_df['STD_DD'] = card_df['STD_DD'].astype('datetime64[ns]')
card_df['GU_CD'] = card_df['GU_CD'].astype(str)
card_df['DONG_CD'] = card_df['DONG_CD'].astype(str)
card_df['GU_DONG'] = card_df['GU_CD'] + card_df['DONG_CD']
card_df['GU_DONG'] = card_df['GU_DONG'].astype(int)

def function1(x):
    if x == 'M':
        b = '0'
    elif x == 'F':
        b = '1'
    return b

card_df['SEX_CD'] = card_df['SEX_CD'].apply(lambda x: function1(x))

def function2(x):
    if x == 110515:
        a = '청운효자동'
    elif x == 110530:
        a = '사직동'
    elif x == 110540:
        a = '삼청동'
    elif x == 110550:
        a = '부암동'
    elif x == 110560:
        a = '평창동'
    elif x == 110570:
        a = '무악동'
    elif x == 110580:
        a = '교남동'
    elif x == 110600:
        a = '가회동'
    elif x == 110615:
        a = '종로1.2.3.4가동'
    elif x == 110630:
        a = '종로5.6가동'
    elif x == 110640:
        a = '이화동'
    elif x == 110650:
        a = '혜화동'
    elif x == 110670:
        a = '창신1동'
    elif x == 110680:
        a = '창신2동'
    elif x == 110690:
        a = '창신3동'
    elif x == 110700:
        a = '숭인1동'
    elif x == 110710:
        a = '숭인2동'
    elif x == 350560:
        a = '월계1동'
    elif x == 350570:
        a = '월계2동'
    elif x == 350580:
        a = '월계3동'
    elif x == 350595:
        a = '공릉1동'
    elif x == 350600:
        a = '공릉2동'
    elif x == 350611:
        a = '하계1동'
    elif x == 350612:
        a = '하계2동'
    elif x == 350619:
        a = '중계본동'
    elif x == 350621:
        a = '중계1동'
    elif x == 350624:
        a = '중계4동'
    elif x == 350625:
        a = '중계2.3동'
    elif x == 350630:
        a = '상계1동'
    elif x == 350640:
        a = '상계2동'
    elif x == 350665:
        a = '상계3.4동'
    elif x == 350670:
        a = '상계5동'
    elif x == 350695:
        a = '상계6.7동'
    elif x == 350700:
        a = '상계8동'
    elif x == 350710:
        a = '상계9동'
    elif x == 350720:
        a = '상계10동'
    return a

card_df['DONG_CD'] = card_df['GU_DONG'].apply(lambda x: function2(x))

def function3(x):
    if x == '110':
        c = '종로구'
    elif x == '350':
        c = '노원구'
    return c
card_df['GU_CD'] = card_df['GU_CD'].apply(lambda x: function3(x))
card_df.drop('GU_DONG', axis = 1, inplace = True)
card_df.head()
card_df.to_csv('C:\\Users\\ejr93\\Desktop\\빅콘\\가공데이터\\card_ppc_df.csv', sep = ',')
```

- 미세먼지 - 매출액 (전체행정동)
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
finedust_day1_df = pd.read_csv(r'C:\\Users\\ejr93\\Desktop\\빅콘\\가공데이터\\finedust_day1.csv', encoding = 'CP949')
finedust_day1_df.head(365)

finedust_day1_df.rename(columns = {'tm' : 'time'}, inplace = True)

def function1(x):
    if x == '종로1,2,3,4가동':
        a = '종로1.2.3.4가동'
    elif x == '종로5,6가동':
        a = '종로5.6가동'
    elif x == '중계2,3동':
        a = '중계2.3동'
    elif x == '상계3,4동':
        a = '상계3.4동'
    elif x == '상계6,7동':
        a = '상계6.7동'
    else:
        a = x
    return a

finedust_day1_df['dong'] = finedust_day1_df['dong'].apply(lambda x: function1(x))
finedust_day1_df.head()
card_df.drop('Unnamed: 0', axis = 1, inplace = True)
card_df.rename(columns = {'STD_DD' : 'time', 'GU_CD' : 'gu', 'DONG_CD' : 'dong'}, inplace = True)
card_df['time'] = card_df['time'].astype(str)
card_dust_df = pd.merge(card_df, finedust_day1_df, on = ['time', 'dong'])
card_dust_df.info()

card_dust_df.drop(['gu_x','dong', 'SEX_CD', 'AGE_CD','Unnamed: 0', 'gu_y', 'humi', 'noise', 'pm25', 'temp', 'yoil', 'weekend','pm25_class', 'pm10_class'], axis = 1, inplace = True)
card_dust_group_df = card_dust_df.groupby(by = ['time', 'MCT_CAT_CD']).agg({'USE_CNT' : 'sum', 'USE_AMT' : 'sum', 'pm10' : 'mean'})
card_dust_group_df.reset_index(inplace=True)
card_dust_group_df.head()

card_dust_group_10_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 10]
card_dust_group_20_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 20]
card_dust_group_21_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 21]
card_dust_group_22_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 22]
card_dust_group_30_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 30]
card_dust_group_31_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 31]
card_dust_group_32_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 32]
card_dust_group_33_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 33]
card_dust_group_34_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 34]
card_dust_group_35_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 35]
card_dust_group_40_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 40]
card_dust_group_42_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 42]
card_dust_group_43_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 43]
card_dust_group_44_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 44]
card_dust_group_50_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 50]
card_dust_group_52_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 52]
card_dust_group_60_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 60]
card_dust_group_62_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 62]
card_dust_group_70_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 70]
card_dust_group_71_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 71]
card_dust_group_80_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 80]
card_dust_group_81_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 81]
card_dust_group_92_df = card_dust_group_df[card_dust_group_df['MCT_CAT_CD'] == 92]

card_dust_group_10_df.reset_index(inplace=True)
card_dust_group_20_df.reset_index(inplace=True)
card_dust_group_21_df.reset_index(inplace=True)
card_dust_group_22_df.reset_index(inplace=True)
card_dust_group_30_df.reset_index(inplace=True)
card_dust_group_31_df.reset_index(inplace=True)
card_dust_group_32_df.reset_index(inplace=True)
card_dust_group_33_df.reset_index(inplace=True)
card_dust_group_34_df.reset_index(inplace=True)
card_dust_group_35_df.reset_index(inplace=True)
card_dust_group_40_df.reset_index(inplace=True)
card_dust_group_42_df.reset_index(inplace=True)
card_dust_group_43_df.reset_index(inplace=True)
card_dust_group_44_df.reset_index(inplace=True)
card_dust_group_50_df.reset_index(inplace=True)
card_dust_group_52_df.reset_index(inplace=True)
card_dust_group_60_df.reset_index(inplace=True)
card_dust_group_62_df.reset_index(inplace=True)
card_dust_group_70_df.reset_index(inplace=True)
card_dust_group_71_df.reset_index(inplace=True)
card_dust_group_80_df.reset_index(inplace=True)
card_dust_group_81_df.reset_index(inplace=True)
card_dust_group_92_df.reset_index(inplace=True)

plt.figure(figsize=(8,6), dpi=80)
plt.plot(card_dust_group_21_df['pm10'], color = 'blue')

from statsmodels.tsa.stattools import adfuller

# 10 
#모든 변수 분산 안정화 변환
pm10_10_log = np.log1p(card_dust_group_10_df['pm10'])
USE_AMT_10_log = np.log1p(card_dust_group_10_df['USE_AMT'])
adfuller(pm10_10_log) # 10 log_pm10 : 비정상 시계열
pm10_10_log_diff1 =np.diff(pm10_10_log) 
adfuller(pm10_10_log_diff1) #log_pm10 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(pm10_10_log_diff1, color = 'blue') #log_pm10 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_dust_group_10_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_10_log) # 10 매출액 : 정상시계열
USE_AMT_10_log_diff1 = np.diff(USE_AMT_10_log)
adfuller(USE_AMT_10_log_diff1) # 10 매출액 1차 차분: 정상시계열

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_10_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_10_log_diff1, lags=50, ax=ax[1])
plt.show()
model_10 = sm.tsa.SARIMAX(USE_AMT_10_log,
                          order=(1,1,1),
                          seasonal_order=(3,1,0,7),
                         exog =pm10_10_log)
results_10 = model_10.fit()
print (results_10.summary())
res_10 = results_10.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_10, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_10, lags=50, ax=ax[1])
plt.show()

# 20
#모든 변수 분산 안정화 변환
pm10_20_log = np.log1p(card_dust_group_20_df['pm10'])
USE_AMT_20_log = np.log1p(card_dust_group_20_df['USE_AMT'])
adfuller(pm10_20_log) # 20 log_pm10 : 비정상 시계열
pm10_20_log_diff1 =np.diff(pm10_20_log) 
adfuller(pm10_20_log_diff1) #log_pm10 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(pm10_20_log_diff1, color = 'blue') #log_pm10 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_dust_group_20_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_20_log) # 20 매출액 : 정상시계열(유의수준 5%)
USE_AMT_20_log_diff1 = np.diff(USE_AMT_20_log)
adfuller(USE_AMT_20_log_diff1) #20 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_20_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_20_log_diff1, lags=50, ax=ax[1])
plt.show()
model_20 = sm.tsa.SARIMAX(USE_AMT_20_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                          exog = pm10_20_log)
results_20 = model_20.fit()
print (results_20.summary())
res_20 = results_20.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_20, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_20, lags=50, ax=ax[1])
plt.show()

# 22
#모든 변수 분산 안정화 변환
pm10_22_log = np.log1p(card_dust_group_22_df['pm10'])
USE_AMT_22_log = np.log1p(card_dust_group_22_df['USE_AMT'])
adfuller(pm10_22_log) # 22 log_pm10 : 비정상 시계열
pm10_22_log_diff1 =np.diff(pm10_22_log) 
adfuller(pm10_22_log_diff1) #log_pm10 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(pm10_22_log_diff1, color = 'blue') #log_pm10 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_dust_group_22_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_22_log) # 22 매출액 : 정상시계열(유의수준 5%)
USE_AMT_22_log_diff1 = np.diff(USE_AMT_22_log)
adfuller(USE_AMT_22_log_diff1) #42 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_22_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_22_log_diff1, lags=50, ax=ax[1])
plt.show()
model_22 = sm.tsa.SARIMAX(USE_AMT_22_log,
                          order=(0,1,2),
                          seasonal_order=(0,1,1,7),
                         exog = pm10_22_log)
results_22 = model_22.fit()
print (results_22.summary())
res_22 = results_22.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_22, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_22, lags=50, ax=ax[1])
plt.show()

# 62
#모든 변수 분산 안정화 변환
pm10_62_log = np.log1p(card_dust_group_62_df['pm10'])
USE_AMT_62_log = np.log1p(card_dust_group_62_df['USE_AMT'])
adfuller(pm10_62_log) # 62 log_pm10 : 비정상 시계열
pm10_62_log_diff1 =np.diff(pm10_62_log) 
adfuller(pm10_62_log_diff1) #log_pm10 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(pm10_62_log_diff1, color = 'blue') #log_pm10 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_dust_group_62_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_62_log) # 62 매출액 : 정상시계열
USE_AMT_62_log_diff1 = np.diff(USE_AMT_62_log)
adfuller(USE_AMT_62_log_diff1) #62 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_62_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_62_log_diff1, lags=50, ax=ax[1])
plt.show()
model_62 = sm.tsa.SARIMAX(USE_AMT_62_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =pm10_62_log)
results_62 = model_62.fit()
print (results_62.summary())
res_62 = results_62.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_62, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_62, lags=50, ax=ax[1])
plt.show()

# 70
#모든 변수 분산 안정화 변환
pm10_70_log = np.log1p(card_dust_group_70_df['pm10'])
USE_AMT_70_log = np.log1p(card_dust_group_70_df['USE_AMT'])
adfuller(pm10_70_log) # 70 log_pm10 : 비정상 시계열
pm10_70_log_diff1 =np.diff(pm10_70_log) 
adfuller(pm10_70_log_diff1) #log_pm10 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(pm10_70_log_diff1, color = 'blue') #log_pm10 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_dust_group_70_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_70_log) # 70 매출액 : 정상시계열
USE_AMT_70_log_diff1 = np.diff(USE_AMT_70_log)
adfuller(USE_AMT_70_log_diff1) #70 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_70_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_70_log_diff1, lags=50, ax=ax[1])
plt.show()
model_70 = sm.tsa.SARIMAX(USE_AMT_70_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =pm10_70_log)
results_70 = model_70.fit()
print (results_70.summary())
res_70 = results_70.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_70, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_70, lags=50, ax=ax[1])
plt.show()
```

- 미세먼지언급량 - 매출액 (전체행정동)
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm

card_df = pd.read_csv(r'C:\\Users\\ejr93\\Desktop\\빅콘\\가공데이터\\card_ppc_df.csv')
card_df.head()

card_df.drop('Unnamed: 0', axis = 1, inplace = True)
card_df.rename(columns = {'STD_DD' : 'time', 'GU_CD' : 'gu', 'DONG_CD' : 'dong'}, inplace = True)
card_df['time'] = card_df['time'].astype(str)
dust_count_df = pd.read_csv(r'C:\\Users\\ejr93\\Desktop\\빅콘\\가공데이터\\dust_count.csv', encoding = 'CP949')
dust_count_df.head()
card_sns_df = pd.merge(card_df, dust_count_df, on = ['time'])
card_sns_df.head()

card_sns_df.drop(['gu','dong','USE_CNT' ,'SEX_CD', 'AGE_CD', 'fine_total', 'new_total'], axis = 1, inplace = True)
card_sns_df.head()

card_sns_group_df = card_sns_df.groupby(by = ['time', 'MCT_CAT_CD']).agg({'USE_AMT' : 'sum', 'cnt_total' : 'mean'})
card_sns_group_df.reset_index(inplace=True)
card_sns_group_df.head()
card_sns_group_df.describe()

card_sns_group_10_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 10]
card_sns_group_20_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 20]
card_sns_group_21_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 21]
card_sns_group_22_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 22]
card_sns_group_30_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 30]
card_sns_group_31_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 31]
card_sns_group_32_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 32]
card_sns_group_33_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 33]
card_sns_group_34_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 34]
card_sns_group_35_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 35]
card_sns_group_40_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 40]
card_sns_group_42_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 42]
card_sns_group_43_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 43]
card_sns_group_44_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 44]
card_sns_group_50_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 50]
card_sns_group_52_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 52]
card_sns_group_60_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 60]
card_sns_group_62_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 62]
card_sns_group_70_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 70]
card_sns_group_71_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 71]
card_sns_group_80_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 80]
card_sns_group_81_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 81]
card_sns_group_92_df = card_sns_group_df[card_sns_group_df['MCT_CAT_CD'] == 92]

card_sns_group_10_df.reset_index(inplace=True)
card_sns_group_20_df.reset_index(inplace=True)
card_sns_group_21_df.reset_index(inplace=True)
card_sns_group_22_df.reset_index(inplace=True)
card_sns_group_30_df.reset_index(inplace=True)
card_sns_group_31_df.reset_index(inplace=True)
card_sns_group_32_df.reset_index(inplace=True)
card_sns_group_33_df.reset_index(inplace=True)
card_sns_group_34_df.reset_index(inplace=True)
card_sns_group_35_df.reset_index(inplace=True)
card_sns_group_40_df.reset_index(inplace=True)
card_sns_group_42_df.reset_index(inplace=True)
card_sns_group_43_df.reset_index(inplace=True)
card_sns_group_44_df.reset_index(inplace=True)
card_sns_group_50_df.reset_index(inplace=True)
card_sns_group_52_df.reset_index(inplace=True)
card_sns_group_60_df.reset_index(inplace=True)
card_sns_group_62_df.reset_index(inplace=True)
card_sns_group_70_df.reset_index(inplace=True)
card_sns_group_71_df.reset_index(inplace=True)
card_sns_group_80_df.reset_index(inplace=True)
card_sns_group_81_df.reset_index(inplace=True)
card_sns_group_92_df.reset_index(inplace=True)
card_sns_group_10_df.head()

plt.figure(figsize=(8,5), dpi=80)
plt.plot(card_sns_group_21_df['cnt_total'], color = 'blue')

from statsmodels.tsa.stattools import adfuller

# 10분석
#모든 변수 분산 안정화 변환
cnt_total_10_log = np.log1p(card_sns_group_10_df['cnt_total'])
USE_AMT_10_log = np.log1p(card_sns_group_10_df['USE_AMT'])
adfuller(cnt_total_10_log) # 10 log_미세먼지 언급량 : 비정상 시계열
cnt_total_10_log_diff1 =np.diff(cnt_total_10_log) 
adfuller(cnt_total_10_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_10_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_10_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_10_log) # 10 매출액 : 정상시계열
USE_AMT_10_log_diff1 = np.diff(USE_AMT_10_log)
adfuller(USE_AMT_10_log_diff1) # 10 매출액 1차 차분: 정상시계열

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_10_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_10_log_diff1, lags=50, ax=ax[1])
plt.show()

model_10 = sm.tsa.SARIMAX(USE_AMT_10_log,
                          order=(1,1,1),
                          seasonal_order=(3,1,0,7),
                         exog =cnt_total_10_log)
results_10 = model_10.fit()
print (results_10.summary())
res_10 = results_10.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_10, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_10, lags=50, ax=ax[1])
plt.show()

# 20 분석
#모든 변수 분산 안정화 변환
cnt_total_20_log = np.log1p(card_sns_group_20_df['cnt_total'])
USE_AMT_20_log = np.log1p(card_sns_group_20_df['USE_AMT'])
adfuller(cnt_total_20_log) # 20 log_미세먼지 언급량 : 비정상 시계열
import statsmodels
cnt_total_20_log_diff1 =np.diff(cnt_total_20_log) 
adfuller(cnt_total_20_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_20_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_20_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_20_log) # 20 매출액 : 정상시계열(유의수준 5%)
USE_AMT_20_log_diff1 = np.diff(USE_AMT_20_log)
adfuller(USE_AMT_20_log_diff1) #20 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_20_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_20_log_diff1, lags=50, ax=ax[1])
plt.show()
model_20 = sm.tsa.SARIMAX(USE_AMT_20_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =cnt_total_20_log)
results_20 = model_20.fit()
print (results_20.summary())
res_20 = results_20.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_20, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_20, lags=50, ax=ax[1])
plt.show()

# 21분석
#모든 변수 분산 안정화 변환
cnt_total_21_log = np.log1p(card_sns_group_21_df['cnt_total'])
USE_AMT_21_log = np.log1p(card_sns_group_21_df['USE_AMT'])
adfuller(cnt_total_21_log) # 21 log_미세먼지 언급량 : 비정상 시계열
cnt_total_21_log_diff1 =np.diff(cnt_total_21_log) 
adfuller(cnt_total_21_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_21_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_21_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_21_log) # 21 매출액 : 정상시계열
USE_AMT_21_log_diff1 = np.diff(USE_AMT_21_log)
adfuller(USE_AMT_21_log_diff1) #42 매출액 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(USE_AMT_21_log_diff1, color = 'blue')
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_21_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_21_log_diff1, lags=50, ax=ax[1])
plt.show()
model_21 = sm.tsa.SARIMAX(USE_AMT_21_log,
                          order=(1,1,1),
                          seasonal_order=(1,1,2,7),
                         exog =cnt_total_10_log)
results_21 = model_21.fit()
print (results_21.summary())
res_21 = results_21.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_21, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_21, lags=50, ax=ax[1])
plt.show()

# 22분석
#모든 변수 분산 안정화 변환
cnt_total_22_log = np.log1p(card_sns_group_22_df['cnt_total'])
USE_AMT_22_log = np.log1p(card_sns_group_22_df['USE_AMT'])
adfuller(cnt_total_22_log) # 22 log_미세먼지 언급량 : 비정상 시계열
cnt_total_22_log_diff1 =np.diff(cnt_total_22_log) 
adfuller(cnt_total_22_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_22_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_22_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_22_log) # 22 매출액 : 정상시계열(유의수준 5%)
USE_AMT_22_log_diff1 = np.diff(USE_AMT_22_log)
adfuller(USE_AMT_22_log_diff1) #42 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_22_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_22_log_diff1, lags=50, ax=ax[1])
plt.show()
model_22 = sm.tsa.SARIMAX(USE_AMT_22_log,
                          order=(1,1,2),
                          seasonal_order=(1,1,0,7),
                         exog =cnt_total_22_log)
results_22 = model_22.fit()
print (results_22.summary())
res_22 = results_22.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_22, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_22, lags=50, ax=ax[1])
plt.show()

# 33분석
#모든 변수 분산 안정화 변환
cnt_total_33_log = np.log1p(card_sns_group_33_df['cnt_total'])
USE_AMT_33_log = np.log1p(card_sns_group_33_df['USE_AMT'])
adfuller(cnt_total_33_log) # 33 log_미세먼지 언급량 : 비정상 시계열
cnt_total_33_log_diff1 =np.diff(cnt_total_33_log) 
adfuller(cnt_total_33_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_33_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(USE_AMT_33_log, color = 'blue')
adfuller(USE_AMT_33_log) # 33 매출액 : 비정상시계열
statsmodels.tsa.stattools.coint(USE_AMT_33_log, cnt_total_33_log, trend = 'nc') #H0 : "공적분 존재 안함" 기각 못함
USE_AMT_33_log_diff1 = np.diff(USE_AMT_33_log)
adfuller(USE_AMT_33_log_diff1) #33 매출액 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(USE_AMT_33_log_diff1, color = 'blue')
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_33_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_33_log_diff1, lags=50, ax=ax[1])
plt.show()
model_33 = sm.tsa.SARIMAX(USE_AMT_33_log,
                          order=(1,1,1),
                          seasonal_order=(1,1,1,7),
                         exog =cnt_total_33_log)
results_33 = model_33.fit()
print (results_33.summary())
res_33 = results_33.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_33, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_33, lags=50, ax=ax[1])
plt.show()

# 40분석
#모든 변수 분산 안정화 변환
cnt_total_40_log = np.log1p(card_sns_group_40_df['cnt_total'])
USE_AMT_40_log = np.log1p(card_sns_group_40_df['USE_AMT'])
adfuller(cnt_total_40_log) # 40 log_미세먼지 언급량 : 비정상 시계열
cnt_total_40_log_diff1 =np.diff(cnt_total_40_log) 
adfuller(cnt_total_40_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_40_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_40_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_40_log) # 40 매출액 : 정상시계열(유의수준 5%)
USE_AMT_40_log_diff1 = np.diff(USE_AMT_40_log)
adfuller(USE_AMT_40_log_diff1) #40 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_40_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_40_log_diff1, lags=50, ax=ax[1])
plt.show()
model_40 = sm.tsa.SARIMAX(USE_AMT_40_log,
                          order=(1,1,1),
                          seasonal_order=(1,1,1,7),
                         exog =cnt_total_40_log)
results_40 = model_40.fit()
print (results_40.summary())
res_40 = results_40.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_40, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_40, lags=50, ax=ax[1])
plt.show()

# 42
#모든 변수 분산 안정화 변환
cnt_total_42_log = np.log1p(card_sns_group_42_df['cnt_total'])
USE_AMT_42_log = np.log1p(card_sns_group_42_df['USE_AMT'])
adfuller(cnt_total_42_log) # 42 log_미세먼지 언급량 : 비정상 시계열
cnt_total_42_log_diff1 =np.diff(cnt_total_42_log) 
adfuller(cnt_total_42_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_42_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_42_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_42_log) # 42 매출액 : 비정상 시계열
statsmodels.tsa.stattools.coint(USE_AMT_42_log, cnt_total_42_log, trend = 'nc') #H0 : "공적분 존재 안함" 기각 못함
USE_AMT_42_log_diff1 = np.diff(USE_AMT_42_log)
adfuller(USE_AMT_42_log_diff1) #42 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_42_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_42_log_diff1, lags=50, ax=ax[1])
plt.show()
model_42 = sm.tsa.SARIMAX(USE_AMT_42_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =cnt_total_42_log)
results_42 = model_42.fit()
print (results_42.summary())
res_42 = results_42.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_42, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_42, lags=50, ax=ax[1])
plt.show()

# 43분석 
#모든 변수 분산 안정화 변환
cnt_total_43_log = np.log1p(card_sns_group_43_df['cnt_total'])
USE_AMT_43_log = np.log1p(card_sns_group_43_df['USE_AMT'])
adfuller(cnt_total_43_log) # 43 log_미세먼지 언급량 : 비정상 시계열
cnt_total_43_log_diff1 =np.diff(cnt_total_43_log) 
adfuller(cnt_total_43_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_43_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_43_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_43_log) # 43 매출액 : 정상시계열
USE_AMT_43_log_diff1 = np.diff(USE_AMT_43_log)
adfuller(USE_AMT_43_log_diff1) #43 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_43_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_43_log_diff1, lags=50, ax=ax[1])
plt.show()
model_43 = sm.tsa.SARIMAX(USE_AMT_43_log,
                          order=(1,1,1),
                          seasonal_order=(1,1,1,7),
                         exog =cnt_total_43_log)
results_43 = model_43.fit()
print (results_43.summary())
res_43 = results_43.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_43, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_43, lags=50, ax=ax[1])
plt.show()

# 44분석
#모든 변수 분산 안정화 변환
cnt_total_44_log = np.log1p(card_sns_group_44_df['cnt_total'])
USE_AMT_44_log = np.log1p(card_sns_group_44_df['USE_AMT'])
adfuller(cnt_total_44_log) # 44 log_미세먼지 언급량 : 비정상 시계열
cnt_total_44_log_diff1 =np.diff(cnt_total_44_log) 
adfuller(cnt_total_44_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_44_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_44_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_44_log) # 44 매출액 : 정상시계열
USE_AMT_44_log_diff1 = np.diff(USE_AMT_44_log)
adfuller(USE_AMT_44_log_diff1) #43 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_44_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_44_log_diff1, lags=50, ax=ax[1])
plt.show()
model_44 = sm.tsa.SARIMAX(USE_AMT_44_log,
                          order=(1,1,1),
                          seasonal_order=(1,1,1,7),
                         exog =cnt_total_44_log)
results_44 = model_44.fit()
print (results_44.summary())
res_44 = results_44.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_44, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_44, lags=50, ax=ax[1])
plt.show()

# 50분석
#모든 변수 분산 안정화 변환
cnt_total_50_log = np.log1p(card_sns_group_50_df['cnt_total'])
USE_AMT_50_log = np.log1p(card_sns_group_50_df['USE_AMT'])
adfuller(cnt_total_50_log) # 50 log_미세먼지 언급량 : 비정상 시계열
cnt_total_50_log_diff1 =np.diff(cnt_total_50_log) 
adfuller(cnt_total_50_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_50_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_50_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_50_log) # 50 매출액 : 비정상시계열
statsmodels.tsa.stattools.coint(USE_AMT_50_log, cnt_total_50_log, trend = 'nc') #H0 : "공적분 존재 안함" 기각 못함
USE_AMT_50_log_diff1 = np.diff(USE_AMT_50_log)
adfuller(USE_AMT_50_log_diff1) #50 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_50_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_50_log_diff1, lags=50, ax=ax[1])
plt.show()
model_50 = sm.tsa.SARIMAX(USE_AMT_50_log,
                          order=(1,1,1),
                          seasonal_order=(1,1,1,7),
                         exog =cnt_total_50_log)
results_50 = model_50.fit()
print (results_50.summary())
res_50 = results_50.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_50, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_50, lags=50, ax=ax[1])
plt.show()

# 62분석
#모든 변수 분산 안정화 변환
cnt_total_62_log = np.log1p(card_sns_group_62_df['cnt_total'])
USE_AMT_62_log = np.log1p(card_sns_group_62_df['USE_AMT'])
adfuller(cnt_total_62_log) # 62 log_미세먼지 언급량 : 비정상 시계열
cnt_total_62_log_diff1 =np.diff(cnt_total_62_log) 
adfuller(cnt_total_62_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_62_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_62_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_62_log) # 62 매출액 : 정상시계열
USE_AMT_62_log_diff1 = np.diff(USE_AMT_62_log)
adfuller(USE_AMT_62_log_diff1) #62 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_62_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_62_log_diff1, lags=50, ax=ax[1])
plt.show()
card_sns_group_62_df.head()
model_62 = sm.tsa.SARIMAX(USE_AMT_62_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =cnt_total_62_log)
results_62 = model_62.fit()
print (results_62.summary())
res_62 = results_62.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_62, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_62, lags=50, ax=ax[1])
plt.show()

# 70분석
#모든 변수 분산 안정화 변환
cnt_total_70_log = np.log1p(card_sns_group_70_df['cnt_total'])
USE_AMT_70_log = np.log1p(card_sns_group_70_df['USE_AMT'])
adfuller(cnt_total_70_log) # 70 log_미세먼지 언급량 : 비정상 시계열
cnt_total_70_log_diff1 =np.diff(cnt_total_70_log) 
adfuller(cnt_total_70_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_70_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_70_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_70_log) # 70 매출액 : 정상시계열
USE_AMT_70_log_diff1 = np.diff(USE_AMT_70_log)
adfuller(USE_AMT_70_log_diff1) #70 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_70_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_70_log_diff1, lags=50, ax=ax[1])
plt.show()
model_70 = sm.tsa.SARIMAX(USE_AMT_70_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =cnt_total_70_log)
results_70 = model_70.fit()
print (results_70.summary())
res_70 = results_70.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_70, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_70, lags=50, ax=ax[1])
plt.show()

# 71분석
#모든 변수 분산 안정화 변환
cnt_total_71_log = np.log1p(card_sns_group_71_df['cnt_total'])
USE_AMT_71_log = np.log1p(card_sns_group_71_df['USE_AMT'])
adfuller(cnt_total_71_log) # 71 log_미세먼지 언급량 : 비정상 시계열
cnt_total_71_log_diff1 =np.diff(cnt_total_71_log) 
adfuller(cnt_total_71_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_71_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_71_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_71_log) # 71 매출액 : 정상시계열
USE_AMT_71_log_diff1 = np.diff(USE_AMT_71_log)
adfuller(USE_AMT_71_log_diff1) #71 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_71_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_71_log_diff1, lags=50, ax=ax[1])
plt.show()
model_71 = sm.tsa.SARIMAX(USE_AMT_71_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =cnt_total_71_log)
results_71 = model_71.fit()
print (results_71.summary())
res_71 = results_71.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_71, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_71, lags=50, ax=ax[1])
plt.show()

# 80분석
#모든 변수 분산 안정화 변환
cnt_total_80_log = np.log1p(card_sns_group_80_df['cnt_total'])
USE_AMT_80_log = np.log1p(card_sns_group_80_df['USE_AMT'])
adfuller(cnt_total_80_log) # 80 log_미세먼지 언급량 : 비정상 시계열
cnt_total_80_log_diff1 =np.diff(cnt_total_80_log) 
adfuller(cnt_total_80_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_80_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_80_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_80_log) # 80 매출액 : 정상시계열
USE_AMT_80_log_diff1 = np.diff(USE_AMT_80_log)
adfuller(USE_AMT_80_log_diff1) #80 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_80_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_80_log_diff1, lags=50, ax=ax[1])
plt.show()
model_80 = sm.tsa.SARIMAX(USE_AMT_80_log,
                          order=(1,1,1),
                          seasonal_order=(1,1,1,7),
                         exog =cnt_total_80_log)
results_80 = model_80.fit()
print (results_80.summary())
res_80 = results_80.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_80, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_80, lags=50, ax=ax[1])
plt.show()

# 81분석
#모든 변수 분산 안정화 변환
cnt_total_81_log = np.log1p(card_sns_group_81_df['cnt_total'])
USE_AMT_81_log = np.log1p(card_sns_group_81_df['USE_AMT'])
adfuller(cnt_total_81_log) # 81 log_미세먼지 언급량 : 비정상 시계열
cnt_total_81_log_diff1 =np.diff(cnt_total_81_log) 
adfuller(cnt_total_81_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_81_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_81_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_81_log) # 81 매출액 : 정상시계열
USE_AMT_81_log_diff1 = np.diff(USE_AMT_81_log)
adfuller(USE_AMT_81_log_diff1) #81 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_81_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_81_log_diff1, lags=50, ax=ax[1])
plt.show()
model_81 = sm.tsa.SARIMAX(USE_AMT_81_log,
                          order=(0,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =cnt_total_81_log)
results_81 = model_81.fit()
print (results_81.summary())
res_81 = results_81.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_81, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_81, lags=50, ax=ax[1])
plt.show()

# 92분석
#모든 변수 분산 안정화 변환
cnt_total_92_log = np.log1p(card_sns_group_92_df['cnt_total'])
USE_AMT_92_log = np.log1p(card_sns_group_92_df['USE_AMT'])
adfuller(cnt_total_92_log) # 92 log_미세먼지 언급량 : 비정상 시계열
cnt_total_92_log_diff1 =np.diff(cnt_total_92_log) 
adfuller(cnt_total_92_log_diff1) #log_미세먼지 언급량 1차 차분 : 정상시계열
plt.figure(figsize=(10,5), dpi=80)
plt.plot(cnt_total_92_log_diff1, color = 'blue') #log_미세먼지 언급량 1차 차분 plot
plt.figure(figsize=(10,5), dpi=80)
plt.plot(card_sns_group_92_df['USE_AMT'], color = 'blue')
adfuller(USE_AMT_92_log) # 92 매출액 : 비정상시계열
USE_AMT_92_log_diff1 = np.diff(USE_AMT_92_log)
adfuller(USE_AMT_92_log_diff1) #92 매출액 1차 차분 : 정상시계열
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(USE_AMT_92_log_diff1, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(USE_AMT_92_log_diff1, lags=50, ax=ax[1])
plt.show()
model_92 = sm.tsa.SARIMAX(USE_AMT_92_log,
                          order=(1,1,1),
                          seasonal_order=(0,1,1,7),
                         exog =cnt_total_92_log)
results_92 = model_92.fit()
print (results_92.summary())
res_92 = results_92.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res_92, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res_92, lags=50, ax=ax[1])
plt.show()
```

- 민감업종별 주요 행정동
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
card_df = pd.read_csv(r'C:\\Users\\ejr93\\Desktop\\빅콘\\가공데이터\\card_ppc_df.csv')
card_df.head()
card_df.drop('Unnamed: 0', axis = 1, inplace = True)
card_df.rename(columns = {'STD_DD' : 'time', 'GU_CD' : 'gu', 'DONG_CD' : 'dong'}, inplace = True)
card_df['time'] = card_df['time'].astype(str)
card_df.head()
card_92_df = card_df[card_df['MCT_CAT_CD'] == 92]
card_70_df = card_df[card_df['MCT_CAT_CD'] == 70]
card_62_df = card_df[card_df['MCT_CAT_CD'] == 62]
card_10_df = card_df[card_df['MCT_CAT_CD'] == 10]
card_20_df = card_df[card_df['MCT_CAT_CD'] == 20]
card_22_df = card_df[card_df['MCT_CAT_CD'] == 22]
card_71_df = card_df[card_df['MCT_CAT_CD'] == 71]
card_81_df = card_df[card_df['MCT_CAT_CD'] == 81]
card_92_group_df = card_92_df.groupby(by = 'dong').sum()
card_92_group_df = card_92_group_df.sort_values(['USE_AMT'], ascending=[False])
card_92_group_df.head()
card_70_group_df = card_70_df.groupby(by = 'dong').sum()
card_70_group_df = card_70_group_df.sort_values(['USE_AMT'], ascending=[False])
card_70_group_df.head()
card_62_group_df = card_62_df.groupby(by = 'dong').sum()
card_62_group_df = card_62_group_df.sort_values(['USE_AMT'], ascending=[False])
card_62_group_df.head()
card_10_group_df = card_10_df.groupby(by = 'dong').sum()
card_10_group_df = card_10_group_df.sort_values(['USE_AMT'], ascending=[False])
card_10_group_df.head()
card_20_group_df = card_20_df.groupby(by = 'dong').sum()
card_20_group_df = card_20_group_df.sort_values(['USE_AMT'], ascending=[False])
card_20_group_df.head()
card_22_group_df = card_22_df.groupby(by = 'dong').sum()
card_22_group_df = card_22_group_df.sort_values(['USE_AMT'], ascending=[False])
card_22_group_df.head()
card_71_group_df = card_71_df.groupby(by = 'dong').sum()
card_71_group_df = card_71_group_df.sort_values(['USE_AMT'], ascending=[False])
card_71_group_df.head()
card_81_group_df = card_81_df.groupby(by = 'dong').sum()
card_81_group_df = card_81_group_df.sort_values(['USE_AMT'], ascending=[False])
card_81_group_df.head()
```

----------------
4. 유동인구 + 의사결정나무

**5. 유동인구 데이터 분석**
```
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as seaborn
from collections import Counter
import math
import glob
```

> 유동인구 데이터 분석
- 공공데이터에서 갖고온 산업체 대분류별 종사자 수 데이터와 세대수 데이터를 갖고와서 행정동별 이해를 높이는 것이 목적
-  종사자 수 데이터 생성  후 유동인구 데이터와 결합 
-  종사자 수 데이터의 경우 업종별 총 종사자 수와 여성 종사자 수만 나와 있음 따라서 두 칼럼의 차인 남성 종사자수 칼럼을 새로 생성
-  이후 이 데이터를 기존의 유동인구 데이터와 결합하여 의사결정나무 분석을 하고 자 함
-  또한 그 전에 기본적인 탐색적 자료 분석을 하는것이 선행되어야 함. 따라서 종로구, 노원구만 가지고 빈도분석 하는 것이 필요해보임


```
os.chdir('C:/Users/Rangkku/Desktop/bigcon/people/')
labor = pd.read_csv("labor.txt",sep = "\t") ## 산업체 대분류별 종사자 수
lb = labor
lb.columns
lb.head()
lb = lb.loc[(lb.자치구 == '종로구')|(lb.자치구 == '노원구')]##종로구 노원구만 인덱싱
lb =  lb.reset_index(drop = True)
lb.head()
lb= lb.rename(columns={'합계': '사업체 수', '합계.1': '총 종사자 수','합계.2':'여성 종사자 수'})
lb_t = lb.copy()
for i in range(len(lb_t.columns)): ### 이름 변경 .1 에서 여성종사자 수로 
    if '1' in lb_t.columns[i]:
        lb =  lb.rename(columns= {lb_t.columns[i]:lb_t.columns[i-1]+' 여성 종사자 수'})
#     df_sex_t.iloc[i,1] = df_sex_t.iloc[i,1].replace(".",",")
# lb_t.head(2)
col_ht = [lb_t.columns[x] for x in range(4,43,2)]
col_ht
## 데이터중  - 로 되어 있는 데이터 0으로 바꾸기
for i in range(len(lb_t.columns)):
     lb_t[lb_t.columns[i]] = lb_t[lb_t.columns[i]].apply(lambda x : 0 if x == '-' else x)
for  i in col_ht:
    if '여성' in i : 
        continue
    else : 
        print(i+' 남성 종사자 수')
lb_t.columns
type(lb_t['광업'][1])
# apply(lambda x : 0 if x == '-' else x)
for i in range(len(lb_t)):
    lb_t.iloc[i,1] = lb_t.iloc[i,1].replace(",","")
x = '10,490'
print(type(x))
x.replace(",","")

lb_cht = lb_t[lb_t['동']=='소계']
lb_cht = lb_cht[['자치구','총 종사자 수',
 '농업 임업 및 어업',
 '광업',
 '제조업',
 '전기 가스 증기 및 공기조절 공급업',
 '수도 하수 및 폐기물 처리 원료 재생업',
 '건설업',
 '도매 및 소매업',
 '운수 및 창고업',
 '숙박 및 음식점업',
 '정보통신업',
 '금융 및 보험업',
 '부동산업',
 '전문 과학 및 기술 서비스업',
 '사업시설 관리 사업 지원 및 임대 서비스업',
 '공공행정 국방 및 사회보장 행정',
 '교육 서비스업',
 '보건업 및 사회복지 서비스업',
 '예술 스포츠 및 여가관련 서비스업',
 '협회 및 단체 수리 및 기타 개인 서비스업']]
group_sizes = list(lb_cht.iloc[1][2:])
group_names = list(lb_cht.columns[2:])
group_explodes = ()
lb_cht.columns[1:]
len(group_names )
len(group_sizes)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
mpl.rcParams['axes.unicode_minus'] = False
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams['figure.figsize'] = [12, 8]
lb_cnt = labor[labor['동']=='소계']
lb_cnt = lb_cnt[['자치구','총 종사자 수',
 '농업 임업 및 어업',
 '광업',
 '제조업',
 '전기 가스 증기 및 공기조절 공급업',
 '수도 하수 및 폐기물 처리 원료 재생업',
 '건설업',
 '도매 및 소매업',
 '운수 및 창고업',
 '숙박 및 음식점업',
 '정보통신업',
 '금융 및 보험업',
 '부동산업',
 '전문 과학 및 기술 서비스업',
 '사업시설 관리 사업 지원 및 임대 서비스업',
 '공공행정 국방 및 사회보장 행정',
 '교육 서비스업',
 '보건업 및 사회복지 서비스업',
 '예술 스포츠 및 여가관련 서비스업',
 '협회 및 단체 수리 및 기타 개인 서비스업']]
lb_cnt = lb_cnt.T
lb_cnt
lb_cnt.columns
No= lb_cnt.iloc[1:,1]
No.sort_values(ascending=False)
jong= lb_cnt.iloc[1:,0]
jong.sort_values(ascending=False)[:6]
jong.sort_values(ascending=False)[:6].index
nowon = list(lb_cht.iloc[1][:]);nowon
```

> 노원/종로 산업체 종사자
- 노원 top5 산업체 종사자 

교육 서비스업                     21222 >
보건업 및 사회복지 서비스업       18404 >
도매 및 소매업                    15534 >
숙박 및 음식점업                   14211 >
운수 및 창고업                    12677

- 종로 top5 산업체 종사자 

도매 및 소매업                    44263 >
건설업                         30754 >
숙박 및 음식점업                   26577 >
금융 및 보험업                    21634 >
전문 과학 및 기술 서비스업             20315

```
def make_pie(group_size):
    import matplotlib.pyplot as plt
    group_sizes = list(group_size)
    plt.pie(group_sizes, labels=group_size.index) # text font size
    plt.savefig("fig.png")
labor=pd.read_csv("labor_all.csv",encoding="euc-kr")
lbr= labor[['동',
 '농업 임업 및 어업',
 '광업',
 '제조업',
 '전기 가스 증기 및 공기조절 공급업',
 '수도 하수 및 폐기물 처리 원료 재생업',
 '건설업',
 '도매 및 소매업',
 '운수 및 창고업',
 '숙박 및 음식점업',
 '정보통신업',
 '금융 및 보험업',
 '부동산업',
 '전문 과학 및 기술 서비스업',
 '사업시설 관리 사업 지원 및 임대 서비스업',
 '공공행정 국방 및 사회보장 행정',
 '교육 서비스업',
 '보건업 및 사회복지 서비스업',
 '예술 스포츠 및 여가관련 서비스업',
 '협회 및 단체 수리 및 기타 개인 서비스업']]
lbr_cnt = lbr[lbr['동']!='소계']
lbr_cnt =  lbr_cnt.T
lbr_cnt.head()
lbr_cnt.iloc[0]
lbr_cnt.columns = lbr_cnt.iloc[0]
lbr_cnt = lbr_cnt.drop("동",axis = 0 )
dnum = len(lbr_cnt.columns)## 동개수
tmp= lbr_cnt.iloc[1:,0]
tmp.sort_values(ascending=False)[:5]
for i in range(dnum):
    tmp= lbr_cnt.iloc[1:,i]
    print(tmp.sort_values(ascending=False)[:5])

tmp= lbr_cnt.iloc[1:,0]
jong.sort_values(ascending=False)[:6]
len(lbr_cnt)
lbr_cnt.head()

group_sizes = list(lb_cht.iloc[0][2:])## 종로구
plt.pie(group_sizes,labels=group_names) 
plt.title("종로구 업종별 종사자 수")
#10,490과 같이 문자로 되어 있는 경우 이를 숫자로 변경하기 위한 전처리를 진행
for i in range(len(lb_t.columns)):
    if i > 2: 
        lb_t[lb_t.columns[i]] = lb_t[lb_t.columns[i]].apply(lambda x : x.replace(",","") if type(x) == str else x )##,를 제거
        lb_t[lb_t.columns[i]] = lb_t[lb_t.columns[i]].apply(lambda x : int(float(x)) )## int형으로 만듬
for  i in col_ht :
    if '여성' not in i : 
        lb_t[i+' 남성 종사자 수']  = lb_t[i]-lb_t[i+' 여성 종사자 수']

# 남성종사자수 = 총 - 여성 종사자수
lb_t.columns
lb_t.to_csv("labor_all.csv",header=True,index = False, encoding="euc-kr")
labor=pd.read_csv("labor_all.csv",encoding="euc-kr")
labor.head()

# 세대수 데이터 EDA
household = pd.read_csv("house.txt",sep = "\t") ## 세대 수 19
hd = household
household.head()
hd = hd.loc[(hd.자치구 == '종로구')|(hd.자치구 == '노원구')]##종로구 노원구만 인덱싱
hd = hd.reset_index(drop = True) ##인덱스  초기화
hd = hd.rename(columns={'기간': 'STD_YM', '자치구': 'SGNG_NM','행정동':'HDONG_NM'})# 칼럼 이름 변경(merge 하기위한 준비)
hd.head()
hd.iloc[2,0]
for i in range(len(hd)):
    hd.iloc[i,0] = str(hd.iloc[i,0]).replace(".","")
#     hd.iloc[i,0] = int(hd.iloc[i,0])
hd.STD_YM = hd.STD_YM.apply(lambda x : int(x))
print(type(hd.iloc[2,0]))
hd.head()
##- 를 제거
for i in range(len(hd.columns)):
    if i> 2: 
        hd[hd.columns[i]]= hd[hd.columns[i]].apply(lambda x : 0 if x == '-' else x)
#10,490과 같이 문자로 되어 있는 경우 이를 숫자로 변경하기 위한 전처리를 진행
for i in range(len(hd.columns)):
    if i> 2: 
        hd[hd.columns[i]]= hd[hd.columns[i]].apply(lambda x : x.replace(",","") if type(x) == str else x )##,를 제거
        hd[hd.columns[i]]= hd[hd.columns[i]].apply(lambda x : int(float(x)))## int형으로 만듬

hd['2-3인세대'] = hd['2인세대']+hd['3인세대']
hd['4인세대 이상']  = hd['4인세대'] +hd['5인세대'] +hd['6인세대'] +hd['7인세대'] +hd['8인세대'] +hd['9인세대']+ hd['10인세대 이상']
hd['5인세대 이상']  = hd['5인세대'] +hd['6인세대'] +hd['7인세대'] +hd['8인세대'] +hd['9인세대']+ hd['10인세대 이상']
hd['3-4인세대'] = hd['3인세대']+hd['4인세대']
hdt = hd[['STD_YM', 'SGNG_NM', 'HDONG_NM', '전체세대수', '1인세대','2인세대','3-4인세대',
       '5인세대 이상']].copy()
hdt.head()

sth_j = hdt[(hdt['SGNG_NM']=='종로구')&(hdt['HDONG_NM']!='소계')].groupby('HDONG_NM').mean()
sth_n = hdt[(hdt['SGNG_NM']!='종로구')&(hdt['HDONG_NM']!='소계')].groupby('HDONG_NM').mean()
sth_j[['1인세대','2인세대','3-4인세대',
       '5인세대 이상']].plot(kind = 'bar')
sth_n[['1인세대','2인세대','3-4인세대',
       '5인세대 이상']].plot(kind = 'bar')
sth [['1인세대','2인세대','3-4인세대','5인세대 이상']]

from collections import Counter
import math
sth.iloc[0,1:]
sth.iloc[1,1:]
sth.columns[1:]
plt.figure(figsize=(4,3))
plt.bar(sth.columns[2:],sth.iloc[0,2:],width = 0.5)
# plt.xlabel('code', fontsize = 14)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.figure(figsize=(4,3))
plt.bar(sth.columns[2:],sth.iloc[1,2:],width = 0.5)
# plt.xlabel('code', fontsize = 14)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
sth.iloc[1,1:]
len(hdt)
hdt.describe()
plt.boxplot(hdt[['1인세대','2-3인세대','4인세대 이상']])

fig = plt.figure(figsize=(20,14))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.set_title('종로구 1인 세대 수')
ax1.bar(hdt[hdt['SGNG_NM']=='종로구']['HDONG_NM'],hdt[hdt['SGNG_NM']=='종로구']['1인세대'])
ax2.set_title('노원구 1인 세대 수')
ax2.bar(hdt[hdt['SGNG_NM']!='종로구']['HDONG_NM'],hdt[hdt['SGNG_NM']!='종로구']['1인세대'])
hdt[hdt['SGNG_NM']=='종로구']['HDONG_NM']
hdg = hdt.groupby('HDONG_NM')
np.round(hdg.sum(),2)

# 시간대별 성연령대별 
#시간대별 데이터 불러오기
#성연령대별 + 미세먼지 데이터 불러오기 
os.chdir('C:/Users/Rangkku/Desktop/bigcon/people/')
poptm = pd.read_csv("poptime.csv",encoding="euc-kr")
popsd = pd.read_csv("dustpop.csv",encoding="euc-kr")
poptm.columns
popsd.columns
poptm.head()
hdpop=  pd.merge(poptm, hdt, on = ['STD_YM','HDONG_NM','SGNG_NM'],how = 'inner')
hdpop['HDONG_NM'] = hdpop['HDONG_NM'].apply(lambda x : x.replace(".",","))
## 세대수 데이터와 기존의 시간대별 유동인구 데이터와 결합
pop_nm= popsd.drop_duplicates('HDONG_NM',keep ='first')
len(pop_nm['HDONG_NM'])
hd_nm = hdpop.drop_duplicates('HDONG_NM',keep ='first')
hd_nm['HDONG_NM'] = hd_nm['HDONG_NM'].apply(lambda x : x.replace(".",","))
len(hd_nm['HDONG_NM'])
print(len(hdpop))
hdpop.head()
type(poptm.iloc[1,0])
print(len(popsd))
popsd.head()
pop_i = pd.merge(hdpop,popsd,on = ['STD_YM','HDONG_NM','SGNG_NM','STD_YMD','HDONG_CD'],how = 'inner')
pop_i.columns
pop_i.columns

# 의사결정나무
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn.cross_validation import train_test_split
import pydot
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn.cross_validation import train_test_split
import pydot
poptm1 = poptm[['STD_YM', 'STD_YMD', 'HDONG_NM','SGNG_NM', 'ya', 'jeon',
       'sim', 'hu', 'jeo']]
popsd1 =  popsd[['SGNG_NM', 'HDONG_NM', 'STD_YM', 'STD_YMD','MAN_FLOW',
       'WMAN_FLOW', 'ALL_FLOW', 'humi', 'noise', 'pm10', 'pm25', 'temp',
       'yoil', 'weekend', 'pm25_class', 'pm10_class']]

import sys; print(sys.executable); import graphviz; print(graphviz.__path__) 
sd_dong = popsd1.drop_duplicates('HDONG_NM',keep='first')
sd_dong.HDONG_NM
for i in range(len(poptm1)):
    lb_t[lb_t.columns[i]] = lb_t[lb_t.columns[i]].apply(lambda x : x.replace(",","") 
poptm1['HDONG_NM'] = poptm1['HDONG_NM'].apply(lambda x : x.replace(".",",") )
tm_dong = poptm1.drop_duplicates(['HDONG_NM'],keep='first')
tm_dong.HDONG_NM
data = pd.merge(popsd1,poptm1,on=['STD_YM', 'STD_YMD', 'HDONG_NM','SGNG_NM'],how= 'inner')
data.head()
!pip install pydot
data = pop_i
data.columns
data['전체세대수'] = data['전체세대수'].apply(lambda x : x.replace(",","") if type(x) == str else x )##,를 제거
data = data.dropna()
y = data.ALL_FLOW
x = data[[ 'STD_YM', 'STD_YMD','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil', 'weekend','전체세대수','1인세대', '2-3인세대', '4인세대 이상']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
os.getcwd()

X_train.isnull().any()

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV

estimator = DecisionTreeRegressor()
param_grid = {'criterion':['mse'], 'max_depth':[None,2,3,4,5,6]}
#param_grid = {'criterion':['mse','friedman_mse','mae'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], max_features:[None,'sqrt','log2',3,4,5]}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='r2') #디폴트로 cv=3, 회귀에서 디폴트로 scoring='r2'

grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_max_depth'))
#print(df.sort_values(by='param_max_depth', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
```
> 결과
- 의사결정나무의 하이퍼 파라미터를 찾아본 결과 max depth가 아예 없는 경우가 가장 mean test score 가 높게 나왔다.

```
rm = DecisionTreeRegressor(max_depth = None)
model = rm.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)

plt.figure(figsize = (8,6))
sns.barplot(x=ftr_sort, y = ftr_sort.index)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)

plt.figure(figsize = (8,6))
sns.barplot(x=ftr_sort, y = ftr_sort.index)
plt.show()
```

> 유동인구 카드 데이터 파일 + 의사결정나무 분석
```
pc=pd.read_csv("pop_card_df.csv")## 유동인구와 카드 결합된 데이터 
pc.head()

pc['time'] = pc['time'].apply(lambda x : x.replace("-",""))## 전처리를 위해 날짜의 '-'를 제거
pc['dong'] = pc['dong'].apply(lambda x : x.replace(".",","))## 다음에 pi 데이터와 합치기 위해 .->,로 바꿈
pc['time'] = pc['time'].astype(int)
data =pc[['time', 'SEX_CD', 'AGE_CD', 'USE_AMT','flow']]
x = data[['time','SEX_CD', 'AGE_CD', 'USE_CNT','flow']]
y = data.USE_AMT
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))
ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)

plt.figure(figsize = (8,6))
sns.barplot(x=ftr_sort, y = ftr_sort.index)
plt.show()
```

> 유동+세대수+ 카드 의사결정나무
```
pc = pc[['time', 'gu', 'dong', 'MCT_CAT_CD', 'SEX_CD', 'AGE_CD',
       'USE_CNT', 'USE_AMT', 'flow']]
pop_i =pd.read_csv("real_pop.csv",encoding="euc-kr")
pop_i.head()
pi = pop_i[[ 'STD_YM','HDONG_NM', 'STD_YMD','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil', 'weekend']]
pi = pi.rename(columns = {'STD_YMD':'time','HDONG_NM':'dong'})
pi['time'] = pi['time'].astype(int)
pc['time'] = pc['time'].astype(int)
data = pd.merge(pi,pc,on = ['time','dong'])
```

> 의사결정나무 분석 미세먼지 - 매출액 예측
```
data.columns
data  = data.dropna()
y = data.USE_AMT
x = data[[  'humi', 'noise', 'pm10', 'pm25', 'temp','AGE_CD']]
x.isnull().any()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))
import io
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz

def draw_decision_tree(model):
    dot_buf = io.StringIO()
    export_graphviz(model, out_file=dot_buf, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
#     node [fontname = font_name, fontsize="11"]
#     image.savefig('tree.png')
    return Image(image)
feature_names = list(x.columns)
draw_decision_tree(model)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)

plt.figure(figsize = (8,6))
sns.barplot(x=ftr_sort, y = ftr_sort.index)
plt.show()

y = data.USE_AMT
x = data[[ 'time', 'humi', 'noise', 'pm10', 'pm25', 'temp',
       'yoil', 'weekend', '전체세대수', '1인세대', '2-3인세대', '4인세대 이상',
       'MCT_CAT_CD', 'SEX_CD', 'AGE_CD', 'USE_CNT',  'flow']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

feature_names = list(x.columns)
draw_decision_tree(model)

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train, y_train)
knr_prediction = model.predict(X_test)
result = np.sqrt(mean_squared_error(knr_prediction, y_test))
rmse.append(result)
```

> 의사결정나무를 위한 그래프 생성함수
```
!pip install graphviz
import io
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz

def draw_decision_tree(model):
    dot_buf = io.StringIO()
    export_graphviz(model, out_file=dot_buf, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
#     node [fontname = font_name, fontsize="11"]
#     image.savefig('tree.png')
    return Image(image)
import io
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz

def draw_decision_tree(model):
    dot_buf = io.StringIO()
    export_graphviz(model, out_file=dot_buf, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
#     node [fontname = font_name, fontsize="11"]
#     image.savefig('tree.png')
    return Image(image)
dot_buf = io.StringIO()
dot_buf
feature_names = list(x.columns)
# dot_buf = 'pop.dot'
draw_decision_tree(model)
data.columns
```

> 업종별 종사자수 EDA
```
labor=pd.read_csv("labor_all.csv",encoding="euc-kr")
labor.head()
lb= lb.rename(columns={'합계': '사업체 수', '합계.1': '총 종사자 수','합계.2':'여성 종사자 수'})
labor.rename(columns={'동':'HDONG_NM'})
dong_NM  =  data.drop_duplicates(['HDONG_NM'],keep='first')
pop_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
pop_tree.fit(X_train, y_train)
df_dust = pd.read_csv("finedust_day.csv")
df_dust = df_dust.rename(columns = {'Unnamed: 0': 'STD_YMD'})
df_dust.head()
df_dust.isnull().any()## 어떻게 처리할지 고민해보기
from sklearn import tree
pop_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
pop_tree.fit(X_train, y_train)
dong = pd.get_dummies(data['HDONG_NM'])
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
!pip install pydotplus
```

> 편의점 데이터 동별로 확인
```
gs_df = pd.read_csv("gs.csv")
gs_df.columns
gs1 = gs_df.copy()
gs1 = gs1.rename(columns={'10_rate':'food_rate', '20_rate':'snack_rate',
       '30_rate':'drink_rate', '40_rate':'homeliving_rate', '50_rate':'health_rate', '60_rate':'hobby_rate', '70_rate':'social_rate', '80_rate':'baby_rate'})
gs1 = gs1.rename(columns={'10_index':'food_index', '20_index':'snack_index','30_index':'drink_index', '40_index':'homeliving_index', '50_index':'health_index', '60_index':'hobby_index', '70_index':'social_index', '80_index':'baby_index'})
gs1.columns
gs1 = gs1[['time', 'gu', 'dong', 'sales_index', 'food_rate',
       'snack_rate', 'drink_rate', 'homeliving_rate', 'health_rate',
       'hobby_rate', 'social_rate', 'baby_rate', 'food_index', 'snack_index',
       'drink_index', 'homeliving_index', 'health_index', 'hobby_index',
       'social_index', 'baby_index']]
gs1 = gs1.rename(columns={"time":'tm'})
gs1['dong'] = gs1['dong'].apply(lambda x : x.replace(".",","))
fine_dust = pd.read_csv("finedust_day1.csv",encoding = 'euc-kr')
dust = fine_dust.iloc[:,1:]
dust = dust[['tm', 'dong', 'humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class']]
dust.isnull().any()
gsdust = pd.merge(gs1,dust,on = ['tm','dong'],how = 'left')
gsdust['dong'].drop_duplicates(keep='first')
gsdust.to_csv("gs_dust.csv",encoding='euc-kr')
```

> 민감한 행정동 + 편의점
-하계2동 상계 1동 교남동 가회동 종로5,6가동 중계2,3동 월계 3동 공릉 1동 공릉2동 하계1동 상계8동

```
gsdust = gsdust[(gsdust['dong']=='하계2동')|(gsdust['dong']=='상계1동')|(gsdust['dong']=='교남동')|(gsdust['dong']=='가회동')|(gsdust['dong']=='종로5,6가동')|(gsdust['dong']=='중계2,3동')|(gsdust['dong']=='월계3동')|(gsdust['dong']=='공릉1동')|(gsdust['dong']=='공릉2동')|(gsdust['dong']=='하계1동')|(gsdust['dong']=='상계8동')]

# 미세먼지 민감도 지수 생성
tsg = pd.read_csv('df_tsg.csv',encoding='euc-kr')
tsg.columns
tsg = tsg[['HDONG_NM', 'SGNG_NM',
       'AREA', 'LENGTH', 'X_COORD', 'Y_COORD', 'STD_YMD',
       'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]
tsg['all_flow']=tsg['TMST_00']+tsg['TMST_01']+tsg['TMST_02']+tsg['TMST_03']+tsg['TMST_04']+tsg['TMST_05']+tsg['TMST_06']+tsg['TMST_07']+tsg['TMST_08']+tsg['TMST_09']+tsg['TMST_10']+tsg['TMST_11']+tsg['TMST_12']+tsg['TMST_13']+tsg['TMST_14']+tsg['TMST_15']+tsg['TMST_16']+tsg['TMST_17']+tsg['TMST_18']+tsg['TMST_19']+tsg['TMST_20']+tsg['TMST_21']+tsg['TMST_23']+tsg['TMST_22']
tsg = tsg.rename(columns={'HDONG_NM' :'dong','SGNG_NM':'gu','STD_YMD':'tm'})
tsg['tm']=tsg['tm'].astype(int)
tsg['dong'] = tsg['dong'].apply(lambda x: x.replace(".",","))
gsdust = pd.read_csv("gs_dust.csv",encoding='euc-kr')
gsdust['tm'] = gsdust['tm'].apply(lambda x: x.replace("-",""))

gsdust['tm'] = gsdust['tm'].astype(int)
gsppds = pd.merge(gsdust,tsg,on= ['dong','gu','tm'],how='inner')## gs people dust join

data = gsppds[['pm10', 'pm25','pm25_class', 'pm10_class','dong','all_flow']]
data['pm25_c'] = data['pm25'].apply(lambda x : 1 if x >= 35 else 0)
data['pm10_c'] = data['pm10'].apply(lambda x : 1 if x >= 80 else 0)                                                       
mom = 0  
chd = 0
for i in range(len(data)):
    if data['pm25_c'][i]==1: 
        mom = mom+ data['all_flow'][i]
    else : 
        chd += data['all_flow'][i]
        
pm25_jisu = (chd/mom)
mom = 0  
chd = 0
mom = 0  
chd = 0
pm10_jisu 
pm25_jisu
cr25 = pm25_jisu
cr10 = pm10_jisu 
cr25 = pm25_jisu
cr10 = pm10_jisu 
df1['all_flow']
def dust_sen(df1,df2,cr25,cr10):
    ####미세먼지 민감도 25
        mom1 = 0; chd1 = 0;
        for i in range(len(df1)):
            if df1['pm25_c'][i]==1: 
                mom1 = mom1+ df1['all_flow'][i]
            else : 
                chd1 += df1['all_flow'][i]
        rst25 = (chd1/mom1)
        print("미세먼지 pm25 민감도 :",rst25)
        if rst25 > cr25 :print("pm25 미세먼지 민감") 
        else :print("pm25 미세먼지 덜 민감") 
        ####미세먼지 민감도 10
        mom2 = 0; chd2 = 0;
        for i in range(len(df2)):
            if df2['pm10_c'][i]==1: 
                mom2 = mom2+ df2['all_flow'][i]
            else : 
                chd2 += df2['all_flow'][i]
        rst10 = (chd2/mom2)
        print("미세먼지 pm10 민감도 :",rst10)
        if rst10 > cr10 :print("pm10 미세먼지 민감") 
        else :print("pm10 미세먼지 덜 민감") 
dong_list = data['dong'].drop_duplicates(keep='first')
print(cr10,cr25)
for i in range(len(dong_list)):
            df1 = data[data['dong']==dong_list[i]][['pm25_c','all_flow']]
            df1 = df1.reset_index()
            df2 = data[data['dong']==dong_list[i]][['pm10_c','all_flow']]
            df2 = df2.reset_index()
            print("행정동명 :",dong_list[i])
            dust_sen(df1,df2,cr25,cr10)
dt = gsppds[['dong','humi', 'noise', 'pm10', 'pm25', 'temp','pm25_class', 'pm10_class', 'AREA', 'LENGTH', 'X_COORD',
       'Y_COORD', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 
             'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23','all_flow']]

# 동별 유동인구 gis로
dt.to_csv("flow_gis.csv",encoding= 'utf-8')
dt1 = dt.groupby('dong').mean()
dt1.to_csv("flow_gis1.csv",encoding= 'utf-8')
data= gsppds[['sales_index', 'food_rate', 'snack_rate',
       'drink_rate', 'homeliving_rate', 'health_rate', 'hobby_rate',
       'social_rate', 'baby_rate', 'food_index', 'snack_index', 'drink_index',
       'homeliving_index', 'health_index', 'hobby_index', 'social_index',
       'baby_index', 'humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'AREA', 'LENGTH', 'X_COORD',
       'Y_COORD', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

import sys; print(sys.executable); import graphviz; print(graphviz.__path__) 
data = data.dropna(axis = 0)
y = data.sales_index
x = data[data.columns[data.columns!='sales_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

data = gsppds[['sales_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

data = data.dropna(axis = 0)
y = data.sales_index
x = data[data.columns[data.columns!='sales_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

###importance plot
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)

plt.figure(figsize = (10,10))
sns.barplot(x=ftr_sort, y = ftr_sort.index)
plt.show()

data.columns
data= gsppds[['sales_index', 'food_rate', 'snack_rate','dong',
       'drink_rate', 'homeliving_rate', 'health_rate', 'hobby_rate',
       'social_rate', 'baby_rate', 'food_index', 'snack_index', 'drink_index',
       'homeliving_index', 'health_index', 'hobby_index', 'social_index',
       'baby_index', 'humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'AREA', 'LENGTH', 'X_COORD',
       'Y_COORD', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

data = data[(data['dong']=='종로1,2,3,4가동')|(data['dong']=='중계1동')|(data['dong']=='상계1동')|(data['dong']=='이화동')|(data['dong']=='종로5,6가동')|(data['dong']=='상계6,7동')|(data['dong']=='중계3동')|(data['dong']=='공릉1동')|(data['dong']=='상계2동')|(data['dong']=='사직동')]

data1=  data.copy()
data1 = data1.dropna(axis = 0)
data1 = data1[['sales_index', 'food_rate', 'snack_rate',
       'drink_rate', 'homeliving_rate', 'health_rate', 'hobby_rate',
       'social_rate', 'baby_rate', 'food_index', 'snack_index', 'drink_index',
       'homeliving_index', 'health_index', 'hobby_index', 'social_index',
       'baby_index', 'humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'AREA', 'LENGTH', 'X_COORD',
       'Y_COORD', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]


y = data1.sales_index
x = data1[data1.columns[data1.columns!='sales_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)

###importance plot
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)

plt.figure(figsize = (10,10))
sns.barplot(x=ftr_sort, y = ftr_sort.index)
plt.show()

gsppds.columns

```

> 매출업종별로 유의미한 변수 의사결정나무로 확인
```
# food
datf = data1[['food_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

datf= datf.dropna(axis = 0)
y = datf.food_index
x = datf[datf.columns[datf.columns!='food_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("그래디언트훈련정확도 :",model.score(X_train, y_train)) #
print("그래디언트테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

# snack
dats = data1[['snack_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]
dats = dats.dropna(axis = 0)
y = dats.snack_index
x = dats[dats.columns[dats.columns!='snack_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("그래디언트훈련정확도 :",model.score(X_train, y_train)) #
print("그래디언트테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

# drink
datd = data1[['drink_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]
datd = datd.dropna(axis = 0)
y = datd.drink_index
x = datd[datd.columns[datd.columns!='drink_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("그래디언트훈련정확도 :",model.score(X_train, y_train)) #
print("그래디언트테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

dath = data1[['homeliving_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

dath = dath.dropna(axis = 0)
y = dath.homeliving_index
x = dath[dath.columns[dath.columns!='homeliving_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("그래디언트훈련정확도 :",model.score(X_train, y_train)) #
print("그래디언트테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

datb = data1[['health_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

datb = datb.dropna(axis = 0)
y = datb.health_index
x = datb[datb.columns[datb.columns!='health_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("그래디언트훈련정확도 :",model.score(X_train, y_train)) #
print("그래디언트테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

daty = data1[['hobby_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

daty = daty.dropna(axis = 0)
y = daty.hobby_index
x = daty[daty.columns[daty.columns!='hobby_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("그래디언트훈련정확도 :",model.score(X_train, y_train)) #
print("그래디언트테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

dats = data1[['social_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

dats = dats.dropna(axis = 0)
y = dats.social_index
x = dats[dats.columns[dats.columns!='social_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("그래디언트훈련정확도 :",model.score(X_train, y_train)) #
print("그래디언트테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

datb = data1[['baby_index','humi', 'noise', 'pm10', 'pm25', 'temp', 'yoil',
       'weekend', 'pm25_class', 'pm10_class', 'MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509',
       'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519',
       'MAN_FLOW_POP_CNT_2024', 'MAN_FLOW_POP_CNT_2529',
       'MAN_FLOW_POP_CNT_3034', 'MAN_FLOW_POP_CNT_3539',
       'MAN_FLOW_POP_CNT_4044', 'MAN_FLOW_POP_CNT_4549',
       'MAN_FLOW_POP_CNT_5054', 'MAN_FLOW_POP_CNT_5559',
       'MAN_FLOW_POP_CNT_6064', 'MAN_FLOW_POP_CNT_6569',
       'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004',
       'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014',
       'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024',
       'WMAN_FLOW_POP_CNT_2529', 'WMAN_FLOW_POP_CNT_3034',
       'WMAN_FLOW_POP_CNT_3539', 'WMAN_FLOW_POP_CNT_4044',
       'WMAN_FLOW_POP_CNT_4549', 'WMAN_FLOW_POP_CNT_5054',
       'WMAN_FLOW_POP_CNT_5559', 'WMAN_FLOW_POP_CNT_6064',
       'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U', 'TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']]

datb = datb.dropna(axis = 0)
y = datb.baby_index
x = datb[datb.columns[datb.columns!='baby_index']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
### 모델 생성
rm = DecisionTreeRegressor(max_depth = 5)
model = rm.fit(X_train, y_train)
###하이퍼 파라미터

###정확도
print("훈련 세트 정확도: {:.3f}".format(rm.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rm.score(X_test, y_test)))

ftr_importances_values = rm.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_sort = ftr_importances.sort_values(ascending=False)
imp_var = ftr_sort[ftr_sort>0]
print(imp_var)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("그래디언트훈련정확도 :",model.score(X_train, y_train)) #
print("그래디언트테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538
```

> 업종별 매출지수를 의사결정나무로 분석해본 결과
미세먼지가 유의미한 변수에 있는 경우는  baby, social, health&beauty,snackindex가 있었다.

```
from sklearn.model_selection import GridSearchCV

estimator = DecisionTreeRegressor()
param_grid = {'criterion':['mse'], 'max_depth':[None,2,3,4,5,6]}
#param_grid = {'criterion':['mse','friedman_mse','mae'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], max_features:[None,'sqrt','log2',3,4,5]}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='r2') #디폴트로 cv=3, 회귀에서 디폴트로 scoring='r2'

grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_max_depth'))
#print(df.sort_values(by='param_max_depth', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_

# GradientBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

##########모델 검증

print(model.score(X_train, y_train)) #
print(model.score(X_test, y_test)) #0.7421680021828538

# randomforestregressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

##########모델 검증

print("랜덤숲훈련정확도 :",model.score(X_train, y_train)) #
print("랜덤숲테스트정확도 :",model.score(X_test, y_test)) #0.7421680021828538

```

------------------

**5. 카드data EDA**

```

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import glob
df_card = pd.read_csv("CARD_SPENDING_190809.txt",sep='\t') 
#빅콘측에서 8월9일 카드데이터 업데이트함

df_card.head()

#df_card[(df_card['DONG_CD'] == 550) & (df_card['GU_CD'] != 110)]
#구,동 결합한 파생변수 생성
df_card.loc[:, 'GU_DONG_CD'] = 1000 * df_card.loc[:, 'GU_CD'] + df_card.loc[:, 'DONG_CD']
df_card.iloc[2:11, 0:11]

#GU_CD 변환
df_card.loc[df_card['GU_CD'] == 110,'GU_CD'] = ['종로구']
df_card.loc[df_card['GU_CD'] == 350,'GU_CD'] =['노원구'] 
df_card['GU_CD'].value_counts()
df_card.head()

#DONG_CD 변환
df_card['GUDONG_CD']=df_card['GU_DONG_CD'].astype("category")
df_card['GUDONG_CD'].cat.categories
df_card['GUDONG_CD'].cat.categories=['청운효자동','사직동','삼청동','부암동','평창동',
                                     '무악동','교남동','가회동','종로1,2,3,4가동','종로5,6가동',
                                     '이화동','혜화동', '창신1동', '창신2동', '창신3동','숭인1동',
                                     '숭인2동','월계1동','월계2동', '월계3동', '공릉1동', '공릉2동',
                                     '하계1동','하계2동','중계본동','중계1동','중계4동','중계2,3동',
                                     '상계1동', '상계2동', '상계3,4동', '상계5동', '상계6,7동', '상계8동',
                                     '상계9동', '상계10동']

df_card['TYPE']=df_card['MCT_CAT_CD'].astype("category")
df_card['TYPE'].cat.categories
df_card['TYPE'].cat.categories=["숙박","레저용품","레저업소","문화취미","가구","전기","주방용구","연료판매",
                         "광학제품","가전","유통업","의복","직물","신변잡화","서적문구","사무통신","자동차판매",
                         "자동차정비","의료기관","보건위생","요식업소","음료식품","수리서비스"]

# 필요한 변수만 남기기
df_card= df_card[['STD_DD','GU_CD','GUDONG_CD','SEX_CD','AGE_CD','TYPE','USE_CNT','USE_AMT']]
# 이름을 편하게
df_card = df_card.rename(columns = {'STD_DD': 'STD_YMD'})
df_card.head()

#일별 카드 데이터
amt_sum = df_card["USE_AMT"].groupby(df_card["STD_YMD"]).sum() # 일별 이용금액 총합
cnt_sum = df_card["USE_CNT"].groupby(df_card["STD_YMD"]).sum() # 일별 이용건수의 총합
amt_mean = df_card["USE_AMT"].groupby(df_card["STD_YMD"]).sum()/df_card["USE_CNT"].groupby(df_card["STD_YMD"]).sum()

card_day = pd.concat([amt_sum,cnt_sum,amt_mean],axis=1,
                    keys=['amt_sum','cnt_sum','amt_mean'])
card_day.head()

card_day.to_csv('card_day.csv')
card_day = pd.read_csv('card_day.csv',index_col='STD_YMD',parse_dates=True)

fig = plt.figure(figsize=(13,4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.set_title('USE_CNT')
ax1.boxplot(df_card['USE_CNT'],vert=False)
ax2.set_title('USE_AMT')
ax2.boxplot(df_card['USE_AMT'],vert=False)
#이용금액과 이용건은 둘다 left-skewed. 각각 median이 47만 9천원과 23건이지만, 상위 25%에서 차이가 가장큼. 

#가장 큰 소비금액 52억원은 2018년 6월 15일 50~59세 남성 샘플인구가 '종로구 종로1.2.3.4가동 (110615)'의 '의료기관(70)'에서 489건 소비함
df_card.ix[df_card['USE_AMT'].idxmax()]

df_card.iloc[418962:418982,0:10] #최대 소비금액을 (418972)을 기준으로 20개 케이스를 봄. 특별하게 더 많은 소비가 보이진 않음.

df_card.nlargest(50, ['USE_AMT']) 

df_card[df_card["USE_AMT"]>=1095046] 

가장 큰 소비가 일어나는  50곳들은 지역별 특정 소비 패턴이 보임
- 종로구 종로1.2.3.4가동 (110 615) 의료기관(70) =>"서울대학교 병원" 추정
- 노원구 상계6.7동 (350 695), 중계본동(350 619) 주유소, LPG가스와 같은 연료판매(33) => 상계 6,7동 "SK엔크린 양지진흥상계주유소", 중계본동 "S-Oil 주유소"
- 주로 4,50대 소비자들
- 의료기관에서는 건당 평균 1억원 정도, 연료판매에서는 건당 평균 7만원 정도.

from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams["font.size"] = 11
def bindo(df,wid):## char  데이터 빈도 분석, 그래프까지 반환
    dong = Counter(df)
    dong_cd = list(dong.keys())
    dong_va = list(dong.values())
    dong_df = pd.DataFrame()
    dong_df['cd'] = dong_cd
    dong_df['value'] = dong_va##장르 이름
    print("가장 큰 값 :\n",dong_df[dong_df['value']==dong_df['value'].max()])##가장 큰 값
    print("가장 작은 값 :\n",dong_df[dong_df['value']==dong_df['value'].min()])##가장 작은값
    plt.rcParams["font.size"] = 11
    plt.figure(figsize = (40,20))
    plt.bar(dong_cd, dong_va,width=wid)
    plt.xlabel('code', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    return 

#bindo(df_card['GU_CD'],3)
# 노원 종로 큰 차이가 없다.
bindo(df_card['GUDONG_CD'], .5) #빈도가 가장 많은 곳(종로1.2.3.4가동)과 가장 작은 곳(창신3동)의 차이는 약 4배 이상.

# 행정도별 카드 매출
df_card.head()
df_cd= df_card.copy()
dong_amt= df_cd['USE_AMT'].groupby([df_card["GUDONG_CD"]]).mean()
dong_amt.plot("box")
plt.figure(figsize = (6,1.5))
sns.boxplot(dong_amt)
plt.show()
dong_amt.values
temp_x = dong_amt.index
temp_y = dong_amt.values
print("최소값:",temp_y.min(),"최대값: ",temp_y.max())
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
mpl.rcParams['axes.unicode_minus'] = False
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams["font.size"] = 10 ## 폰트 크기 결정
plt.figure(figsize = (7,5))
dong_amt.plot("bar")

# 연령카드 매출
tmp_age = df_cd['USE_AMT'].groupby([df_card["AGE_CD"],df_card["SEX_CD"]]).mean()
tmp_sex = df_cd['USE_AMT'].groupby([df_card["SEX_CD"],df_card["AGE_CD"]]).mean()
tmp_sex = df_cd[['USE_AMT','AGE_CD']].groupby([df_card["SEX_CD"]]).mean()
tmp_sex 
 = pd.DataFrame(tmp_sex)
tmp_age
tmp_age.plot("bar")
tmp_age.plot("bar")

# 업종별 카드 매출
df_cdty = df_cd.rename(columns={'TYPE':'업종'})
tmp_ty = df_cdty['USE_AMT'].groupby([df_cdty["업종"]]).mean()

plt.figure(figsize=(9,5))
tmp_ty.plot("bar",width=0.6)
plt.xticks(fontsize = 12)
plt.xlabel("업종별 평균 카드 매출액",fontsize = 14)
print("최대매출 : ",tmp_ty.argmax(),tmp_ty.max())
print("최소매출 : ",tmp_ty.argmin(),tmp_ty.min())
tmp_ty = df_cdty['USE_CNT'].groupby([df_cdty["업종"]]).sum()
print("최대매출건수 : ",tmp_ty.argmax(),tmp_ty.max())
print("최소매출 건수: ",tmp_ty.argmin(),tmp_ty.min())
plt.figure(figsize=(9,5))
tmp_ty.plot("bar",width=0.6)
plt.xticks(fontsize = 12)
plt.xlabel("업종별 평균 카드 매출건수",fontsize = 14)

tmp_ty = df_cd['USE_AMT'].groupby([df_card["TYPE"]]).mean()
plt.figure(figsize=(9,5))
tmp_ty.plot("bar",width=0.5)
```

-------------------

