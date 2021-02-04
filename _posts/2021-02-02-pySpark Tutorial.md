---
category: programmers
tags: [K-digital training, week9_day2, ml_basics, pyspark ]
use_math: true
---

## pySpark Tutorial with Colab
PySpark을 로컬머신에 설치하고 노트북을 사용하기 보다는 머신러닝 관련 다양한 라이브러리가 이미 설치되었고 좋은 하드웨어를 제공해주는 Google Colab을 통해 실습을 진행한다.

이를 위해 pyspark과 py4J 패키지를 설치한다. Py4J패키지는 파이썬 프로그램이 자바가상머신상의 오브젝트들을 접근할 수 있게 해준다. Local Standardalone Spark를 사용한다.


```
!pip install pyspark==3.0.1 py4j==0.10.9
```

    Collecting pyspark==3.0.1
    [?25l  Downloading https://files.pythonhosted.org/packages/f0/26/198fc8c0b98580f617cb03cb298c6056587b8f0447e20fa40c5b634ced77/pyspark-3.0.1.tar.gz (204.2MB)
    [K     |████████████████████████████████| 204.2MB 66kB/s 
    [?25hCollecting py4j==0.10.9
    [?25l  Downloading https://files.pythonhosted.org/packages/9e/b6/6a4fb90cd235dc8e265a6a2067f2a2c99f0d91787f06aca4bcf7c23f3f80/py4j-0.10.9-py2.py3-none-any.whl (198kB)
    [K     |████████████████████████████████| 204kB 47.7MB/s 
    [?25hBuilding wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py) ... [?25l[?25hdone
      Created wheel for pyspark: filename=pyspark-3.0.1-py2.py3-none-any.whl size=204612242 sha256=fca352d74764aefb8a09781c034a614b6d6d368403f6797e5fee21b4d312e888
      Stored in directory: /root/.cache/pip/wheels/5e/bd/07/031766ca628adec8435bb40f0bd83bb676ce65ff4007f8e73f
    Successfully built pyspark
    Installing collected packages: py4j, pyspark
    Successfully installed py4j-0.10.9 pyspark-3.0.1



```
!ls -tl
```

    total 4
    drwxr-xr-x 1 root root 4096 Jan 20 17:27 sample_data



```
!ls -tl sample_data
```

    total 55504
    -rw-r--r-- 1 root root 18289443 Jan 20 17:27 mnist_test.csv
    -rw-r--r-- 1 root root 36523880 Jan 20 17:27 mnist_train_small.csv
    -rw-r--r-- 1 root root   301141 Jan 20 17:27 california_housing_test.csv
    -rw-r--r-- 1 root root  1706430 Jan 20 17:27 california_housing_train.csv
    -rwxr-xr-x 1 root root     1697 Jan  1  2000 anscombe.json
    -rwxr-xr-x 1 root root      930 Jan  1  2000 README.md


**Spark Session**: Spark Session은 Spark 2.0부터 엔트리 포인트로 사용된다. 그 이전에는 SparkContext가 사용되었다. SparkSession을 이용해 RDd, 데이터 프레임 등을 만든다. SparkSession은 SparkSession.builder를 호출하여 생성하며 다양한 함수들을 통해 세부 설정이 가능하다. 


```python
from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .master("local[*]")\  #마스터의 인자로는 사용하고 싶은 spark cluster호스트 이름을 줌
        .appName('pySpark_Tutorial')\
        .getOrCreate()
```


```python
spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://216bf774aeac:4040">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v3.0.1</code></dd>
      <dt>Master</dt>
        <dd><code>local[*]</code></dd>
      <dt>AppName</dt>
        <dd><code>pySpark_Tutorial</code></dd>
    </dl>
</div>

    </div>




### Python 객체를 RDD로 변환해보기
#### 1> Python 리스트 생성


```python
name_list_json = ['{"name" : "keeyong"}', '{"name" : "benjamin"}', '{"name" : "claire"}']
```


```python
for n in name_list_json:
  print(n)
```

    {"name" : "keeyong"}
    {"name" : "benjamin"}
    {"name" : "claire"}



```python
import json
for n in name_list_json:
  jn = json.loads(n)
  print(jn["name"])
```

    keeyong
    benjamin
    claire


#### 2> 파이썬 리스트를 RDD로 변환, RDD로 변환되는 순간 Spark 클러스터의 서버들에 데이터가 나누어 저장됨(파티션). 또한 Lazy Execution이 된다는 점 기억


```python
rdd = spark.sparkContext.parallelize(name_list_json)
```


```python
rdd
```




    ParallelCollectionRDD[0] at readRDDFromFile at PythonRDD.scala:262




```python
rdd.count()
```




    3




```python
parsed_rdd = rdd.map(lambda el : json.loads(el))
```


```python
parsed_rdd
```




    PythonRDD[2] at RDD at PythonRDD.scala:53




```python
parsed_rdd.collect()
```




    [{'name': 'keeyong'}, {'name': 'benjamin'}, {'name': 'claire'}]




```python
parsed_name_rdd = rdd.map(lambda el : json.loads(el)["name"])
```


```python
parsed_name_rdd.collect()
```




    ['keeyong', 'benjamin', 'claire']



### 파이썬 리스트를 데이터 프레임으로 변환하기


```python
from pyspark.sql.types import StringType

df = spark.createDataFrame(name_list_json, StringType())
```


```python
df.count()
```




    3




```python
df.printSchema()
```

    root
     |-- value: string (nullable = true)
    



```python
df.select('*').collect()  # * : 모든 객체 불러오기
```




    [Row(value='{"name" : "keeyong"}'),
     Row(value='{"name" : "benjamin"}'),
     Row(value='{"name" : "claire"}')]




```python
df.select('value').collect()
```




    [Row(value='{"name" : "keeyong"}'),
     Row(value='{"name" : "benjamin"}'),
     Row(value='{"name" : "claire"}')]




```python
from pyspark.sql import Row

row = Row("name") # Or some other column name
df_name = parsed_name_rdd.map(row).toDF()
```


```python
df_name.printSchema()
```

    root
     |-- name: string (nullable = true)
    



```python
df_name.select('name').collect()
```




    [Row(name='keeyong'), Row(name='benjamin'), Row(name='claire')]

