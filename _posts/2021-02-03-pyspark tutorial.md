---
category: programmers
tags: [K-digital training, week9_day3, ml_basics, pyspark, SQL ]
use_math: true
---

PySpark을 로컬머신에 설치하고 노트북을 사용하기 보다는 머신러닝 관련 다양한 라이브러리가 이미 설치되었고 좋은 하드웨어를 제공해주는 Google Colab을 통해 실습을 진행한다.

이를 위해 pyspark과 Py4J 패키지를 설치한다. Py4J 패키지는 파이썬 프로그램이 자바가상머신상의 오브젝트들을 접근할 수 있게 해준다. Local Standalone Spark을 사용한다.


```
!pip install pyspark==3.0.1 py4j==0.10.9 
```

    Collecting pyspark==3.0.1
    [?25l  Downloading https://files.pythonhosted.org/packages/f0/26/198fc8c0b98580f617cb03cb298c6056587b8f0447e20fa40c5b634ced77/pyspark-3.0.1.tar.gz (204.2MB)
    [K     |████████████████████████████████| 204.2MB 64kB/s 
    [?25hCollecting py4j==0.10.9
    [?25l  Downloading https://files.pythonhosted.org/packages/9e/b6/6a4fb90cd235dc8e265a6a2067f2a2c99f0d91787f06aca4bcf7c23f3f80/py4j-0.10.9-py2.py3-none-any.whl (198kB)
    [K     |████████████████████████████████| 204kB 48.8MB/s 
    [?25hBuilding wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py) ... [?25l[?25hdone
      Created wheel for pyspark: filename=pyspark-3.0.1-py2.py3-none-any.whl size=204612242 sha256=5f26268e1b3f9f2c40852763dc32dd19c61a4ec58c0da7b8d485467177900370
      Stored in directory: /root/.cache/pip/wheels/5e/bd/07/031766ca628adec8435bb40f0bd83bb676ce65ff4007f8e73f
    Successfully built pyspark
    Installing collected packages: py4j, pyspark
    Successfully installed py4j-0.10.9 pyspark-3.0.1


지금부터 실습은 Redshift에 있는 데이터를 가지고 해볼 예정이고 그래서 Redshift 관련 JAR 파일을 설치해야함


```
!cd /usr/local/lib/python3.6/dist-packages/pyspark/jars && wget https://s3.amazonaws.com/redshift-downloads/drivers/jdbc/1.2.20.1043/RedshiftJDBC42-no-awssdk-1.2.20.1043.jar
```

    --2021-02-03 06:50:47--  https://s3.amazonaws.com/redshift-downloads/drivers/jdbc/1.2.20.1043/RedshiftJDBC42-no-awssdk-1.2.20.1043.jar
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.244.30
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.244.30|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2413910 (2.3M) [application/java-archive]
    Saving to: ‘RedshiftJDBC42-no-awssdk-1.2.20.1043.jar’
    
    RedshiftJDBC42-no-a 100%[===================>]   2.30M  3.85MB/s    in 0.6s    
    
    2021-02-03 06:50:48 (3.85 MB/s) - ‘RedshiftJDBC42-no-awssdk-1.2.20.1043.jar’ saved [2413910/2413910]
    


**Spark Session:** 이번 SparkSession은 spark.jars를 통해 앞서 다운로드받은 Redshift 연결을 위한 JDBC 드라이버를 사용함 (.config("spark.jars", ...)


```python
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.jars", "/usr/local/lib/python3.6/dist-packages/pyspark/jars/RedshiftJDBC42-no-awssdk-1.2.20.1043.jar") \
    .getOrCreate()
```


```python
spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://34588fae8310:4040">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v3.0.1</code></dd>
      <dt>Master</dt>
        <dd><code>local[*]</code></dd>
      <dt>AppName</dt>
        <dd><code>Python Spark SQL basic example</code></dd>
    </dl>
</div>

    </div>




# **SparkSQL 맛보기**

판다스로 일단 CSV 파일 하나 로드하기


```python
import pandas as pd

namegender_pd = pd.read_csv("https://s3-geospatial.s3-us-west-2.amazonaws.com/name_gender.csv")
```


```python
namegender_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adaleigh</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Amryn</td>
      <td>Unisex</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apurva</td>
      <td>Unisex</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aryion</td>
      <td>M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alixia</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>




```
namegender_pd.groupby(["gender"]).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <td>65</td>
    </tr>
    <tr>
      <th>M</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Unisex</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



판다스 데이터프레임을 Spark 데이터프레임으로 변환하기


```python
namegender_df = spark.createDataFrame(namegender_pd)
```


```python
namegender_df.printSchema()
```

    root
     |-- name: string (nullable = true)
     |-- gender: string (nullable = true)
    



```python
namegender_df.show()
```

    +----------+------+
    |      name|gender|
    +----------+------+
    |  Adaleigh|     F|
    |     Amryn|Unisex|
    |    Apurva|Unisex|
    |    Aryion|     M|
    |    Alixia|     F|
    |Alyssarose|     F|
    |    Arvell|     M|
    |     Aibel|     M|
    |   Atiyyah|     F|
    |     Adlie|     F|
    |    Anyely|     F|
    |    Aamoni|     F|
    |     Ahman|     M|
    |    Arlane|     F|
    |   Armoney|     F|
    |   Atzhiry|     F|
    | Antonette|     F|
    |   Akeelah|     F|
    | Abdikadir|     M|
    |    Arinze|     M|
    +----------+------+
    only showing top 20 rows
    



```python
# collect를 붙이면 로컬로 가져와서 작업을 수행
namegender_df.groupBy(["gender"]).count().collect()
```




    [Row(gender='F', count=65),
     Row(gender='M', count=28),
     Row(gender='Unisex', count=7)]




```python
# https://towardsdatascience.com/pyspark-and-sparksql-basics-6cb4bf967e53
```

데이터프레임을 테이블뷰로 만들어서 SparkSQL로 처리해보기


```python
namegender_df.createOrReplaceTempView("namegender")
```


```python
namegender_group_df = spark.sql("SELECT gender, count(1) FROM namegender GROUP BY 1")
```


```python
namegender_group_df.collect()
```




    [Row(gender='F', count(1)=65),
     Row(gender='M', count(1)=28),
     Row(gender='Unisex', count(1)=7)]



Redshift와 연결해서 테이블들을 데이터프레임으로 로딩하기


```python
df_user_session_channel = spark.read \
    .format("jdbc") \
    .option("driver", "com.amazon.redshift.jdbc42.Driver") \
    .option("url", "jdbc:redshift://learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod?user=guest&password=Guest1!*") \
    .option("dbtable", "raw_data.user_session_channel") \
    .load()
```


```python
df_session_timestamp = spark.read \
    .format("jdbc") \
    .option("driver", "com.amazon.redshift.jdbc42.Driver") \
    .option("url", "jdbc:redshift://learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod?user=guest&password=Guest1!*") \
    .option("dbtable", "raw_data.session_timestamp") \
    .load()
```


```python
df_user_session_channel.createOrReplaceTempView("user_session_channel")
```


```python
df_session_timestamp.createOrReplaceTempView("session_timestamp")
```


```python
channel_count_df = spark.sql("""
    SELECT channel, count(distinct userId) uniqueUsers
    FROM session_timestamp st
    JOIN user_session_channel usc ON st.sessionID = usc.sessionID
    GROUP BY 1
    ORDER BY 1
""")
```


```python
channel_count_df
```




    DataFrame[channel: string, uniqueUsers: bigint]




```python
channel_count_df.show()
```

    +---------+-----------+
    |  channel|uniqueUsers|
    +---------+-----------+
    | Facebook|        889|
    |   Google|        893|
    |Instagram|        895|
    |    Naver|        882|
    |  Organic|        895|
    |  Youtube|        889|
    +---------+-----------+
    



```python
channel_with_o_count_df = spark.sql("""
    SELECT COUNT(1)
    FROM user_session_channel
    WHERE channel like '%o%'
""")
```


```python
channel_with_o_count_df.collect()
```




    [Row(count(1)=50864)]


