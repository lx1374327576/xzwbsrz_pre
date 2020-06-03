
暂时还没把后面的模型接上去

Todo:
- xgboost
- word2vec的两层连接层，因为还没确定用那个深度学习框架写
- 需要探究一些train集和test在ad.csv的上重合度，可能需要合起来做个map
- 现在特征工程做的太少了，



#### 一些临时笔记
three table: click_log.csv，user.csv，ad.csv

user.csv not in test
二分类 -> person2person
多分类 -> 
```
click_log.csv {
    time:
    user_id:
    creative_id:
    click_times:
}

user.csv {
    user_id:
    age: [1-10]
    gender: [1,2]
}

ad.csv {
    creative_id:
    ad_id:
    product_id
    product_category:
    advertiser_id:
    industry:
}
```

Todo:
统计一下信息
- 每个商品出现次数的图表
- 不同性别的数据量
- 不同年龄的数据量
- 不同商品的数据量
- 不同种类商品的数据量

在广告ad里面以下几列有脏数据：product_id，industry
在click_log中，click_times存在异常数据


对比一下train和test的ad文件

