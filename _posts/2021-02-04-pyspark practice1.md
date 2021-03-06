---
category: programmers
tags: [K-digital training, week9_day4, ml_basics, pyspark, SQL ]
use_math: true
---

<a href="https://colab.research.google.com/github/learn-programmers/programmers_kdt_II/blob/main/9%EC%A3%BC%EC%B0%A8_PySpark_%EA%B8%B0%EB%B3%B8_4%EC%9D%BC%EC%B0%A8_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

PySpark을 로컬머신에 설치하고 노트북을 사용하기 보다는 머신러닝 관련 다양한 라이브러리가 이미 설치되었고 좋은 하드웨어를 제공해주는 Google Colab을 통해 실습을 진행한다.

이를 위해 pyspark과 Py4J 패키지를 설치한다. Py4J 패키지는 파이썬 프로그램이 자바가상머신상의 오브젝트들을 접근할 수 있게 해준다. Local Standalone Spark을 사용한다.


```
!pip install pyspark==3.0.1 py4j==0.10.9 
```

    Collecting pyspark==3.0.1
    [?25l  Downloading https://files.pythonhosted.org/packages/f0/26/198fc8c0b98580f617cb03cb298c6056587b8f0447e20fa40c5b634ced77/pyspark-3.0.1.tar.gz (204.2MB)
    [K     |████████████████████████████████| 204.2MB 74kB/s 
    [?25hCollecting py4j==0.10.9
    [?25l  Downloading https://files.pythonhosted.org/packages/9e/b6/6a4fb90cd235dc8e265a6a2067f2a2c99f0d91787f06aca4bcf7c23f3f80/py4j-0.10.9-py2.py3-none-any.whl (198kB)
    [K     |████████████████████████████████| 204kB 51.5MB/s 
    [?25hBuilding wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py) ... [?25l[?25hdone
      Created wheel for pyspark: filename=pyspark-3.0.1-py2.py3-none-any.whl size=204612242 sha256=8c35336b8b45346848bd16c59bcc5e503bb02d29a269652d6ea473f79f651390
      Stored in directory: /root/.cache/pip/wheels/5e/bd/07/031766ca628adec8435bb40f0bd83bb676ce65ff4007f8e73f
    Successfully built pyspark
    Installing collected packages: py4j, pyspark
    Successfully installed py4j-0.10.9 pyspark-3.0.1



```python
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Boston Housing Linear Regression example") \
    .getOrCreate()
```

# 보스턴 주택 가격 예측 모델 만들기





```python
spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://ffc033bbb4bb:4040">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v3.0.1</code></dd>
      <dt>Master</dt>
        <dd><code>local[*]</code></dd>
      <dt>AppName</dt>
        <dd><code>Boston Housing Linear Regression example</code></dd>
    </dl>
</div>

    </div>





```
!wget https://s3-geospatial.s3-us-west-2.amazonaws.com/boston_housing.csv
```

    --2021-02-04 06:42:52--  https://s3-geospatial.s3-us-west-2.amazonaws.com/boston_housing.csv
    Resolving s3-geospatial.s3-us-west-2.amazonaws.com (s3-geospatial.s3-us-west-2.amazonaws.com)... 52.218.224.209
    Connecting to s3-geospatial.s3-us-west-2.amazonaws.com (s3-geospatial.s3-us-west-2.amazonaws.com)|52.218.224.209|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 36240 (35K) [text/csv]
    Saving to: ‘boston_housing.csv’
    
    boston_housing.csv  100%[===================>]  35.39K  --.-KB/s    in 0.08s   
    
    2021-02-04 06:42:53 (432 KB/s) - ‘boston_housing.csv’ saved [36240/36240]
    



```
!ls -tl
```

    total 40
    drwxr-xr-x 1 root root  4096 Feb  1 17:27 sample_data
    -rw-r--r-- 1 root root 36240 Jan 31 01:46 boston_housing.csv



```python
data = spark.read.csv('./boston_housing.csv', header=True, inferSchema=True)
```


```python
data.printSchema()
```

    root
     |-- crim: double (nullable = true)
     |-- zn: double (nullable = true)
     |-- indus: double (nullable = true)
     |-- chas: integer (nullable = true)
     |-- nox: double (nullable = true)
     |-- rm: double (nullable = true)
     |-- age: double (nullable = true)
     |-- dis: double (nullable = true)
     |-- rad: integer (nullable = true)
     |-- tax: integer (nullable = true)
     |-- ptratio: double (nullable = true)
     |-- b: double (nullable = true)
     |-- lstat: double (nullable = true)
     |-- medv: double (nullable = true)
    



```python
data.show()
```

    +-------+----+-----+----+-----+-----+-----+------+---+---+-------+------+-----+----+
    |   crim|  zn|indus|chas|  nox|   rm|  age|   dis|rad|tax|ptratio|     b|lstat|medv|
    +-------+----+-----+----+-----+-----+-----+------+---+---+-------+------+-----+----+
    |0.00632|18.0| 2.31|   0|0.538|6.575| 65.2|  4.09|  1|296|   15.3| 396.9| 4.98|24.0|
    |0.02731| 0.0| 7.07|   0|0.469|6.421| 78.9|4.9671|  2|242|   17.8| 396.9| 9.14|21.6|
    |0.02729| 0.0| 7.07|   0|0.469|7.185| 61.1|4.9671|  2|242|   17.8|392.83| 4.03|34.7|
    |0.03237| 0.0| 2.18|   0|0.458|6.998| 45.8|6.0622|  3|222|   18.7|394.63| 2.94|33.4|
    |0.06905| 0.0| 2.18|   0|0.458|7.147| 54.2|6.0622|  3|222|   18.7| 396.9| 5.33|36.2|
    |0.02985| 0.0| 2.18|   0|0.458| 6.43| 58.7|6.0622|  3|222|   18.7|394.12| 5.21|28.7|
    |0.08829|12.5| 7.87|   0|0.524|6.012| 66.6|5.5605|  5|311|   15.2| 395.6|12.43|22.9|
    |0.14455|12.5| 7.87|   0|0.524|6.172| 96.1|5.9505|  5|311|   15.2| 396.9|19.15|27.1|
    |0.21124|12.5| 7.87|   0|0.524|5.631|100.0|6.0821|  5|311|   15.2|386.63|29.93|16.5|
    |0.17004|12.5| 7.87|   0|0.524|6.004| 85.9|6.5921|  5|311|   15.2|386.71| 17.1|18.9|
    |0.22489|12.5| 7.87|   0|0.524|6.377| 94.3|6.3467|  5|311|   15.2|392.52|20.45|15.0|
    |0.11747|12.5| 7.87|   0|0.524|6.009| 82.9|6.2267|  5|311|   15.2| 396.9|13.27|18.9|
    |0.09378|12.5| 7.87|   0|0.524|5.889| 39.0|5.4509|  5|311|   15.2| 390.5|15.71|21.7|
    |0.62976| 0.0| 8.14|   0|0.538|5.949| 61.8|4.7075|  4|307|   21.0| 396.9| 8.26|20.4|
    |0.63796| 0.0| 8.14|   0|0.538|6.096| 84.5|4.4619|  4|307|   21.0|380.02|10.26|18.2|
    |0.62739| 0.0| 8.14|   0|0.538|5.834| 56.5|4.4986|  4|307|   21.0|395.62| 8.47|19.9|
    |1.05393| 0.0| 8.14|   0|0.538|5.935| 29.3|4.4986|  4|307|   21.0|386.85| 6.58|23.1|
    | 0.7842| 0.0| 8.14|   0|0.538| 5.99| 81.7|4.2579|  4|307|   21.0|386.75|14.67|17.5|
    |0.80271| 0.0| 8.14|   0|0.538|5.456| 36.6|3.7965|  4|307|   21.0|288.99|11.69|20.2|
    | 0.7258| 0.0| 8.14|   0|0.538|5.727| 69.5|3.7965|  4|307|   21.0|390.95|11.28|18.2|
    +-------+----+-----+----+-----+-----+-----+------+---+---+-------+------+-----+----+
    only showing top 20 rows
    


## 피쳐 벡터를 만들기


```python
from pyspark.ml.feature import VectorAssembler

# medv빼고 전부
feature_columns = data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
```


```python
feature_columns
```




    ['crim',
     'zn',
     'indus',
     'chas',
     'nox',
     'rm',
     'age',
     'dis',
     'rad',
     'tax',
     'ptratio',
     'b',
     'lstat']




```python
data_2 = assembler.transform(data)
```


```python
data_2.show()
```

    +-------+----+-----+----+-----+-----+-----+------+---+---+-------+------+-----+----+--------------------+
    |   crim|  zn|indus|chas|  nox|   rm|  age|   dis|rad|tax|ptratio|     b|lstat|medv|            features|
    +-------+----+-----+----+-----+-----+-----+------+---+---+-------+------+-----+----+--------------------+
    |0.00632|18.0| 2.31|   0|0.538|6.575| 65.2|  4.09|  1|296|   15.3| 396.9| 4.98|24.0|[0.00632,18.0,2.3...|
    |0.02731| 0.0| 7.07|   0|0.469|6.421| 78.9|4.9671|  2|242|   17.8| 396.9| 9.14|21.6|[0.02731,0.0,7.07...|
    |0.02729| 0.0| 7.07|   0|0.469|7.185| 61.1|4.9671|  2|242|   17.8|392.83| 4.03|34.7|[0.02729,0.0,7.07...|
    |0.03237| 0.0| 2.18|   0|0.458|6.998| 45.8|6.0622|  3|222|   18.7|394.63| 2.94|33.4|[0.03237,0.0,2.18...|
    |0.06905| 0.0| 2.18|   0|0.458|7.147| 54.2|6.0622|  3|222|   18.7| 396.9| 5.33|36.2|[0.06905,0.0,2.18...|
    |0.02985| 0.0| 2.18|   0|0.458| 6.43| 58.7|6.0622|  3|222|   18.7|394.12| 5.21|28.7|[0.02985,0.0,2.18...|
    |0.08829|12.5| 7.87|   0|0.524|6.012| 66.6|5.5605|  5|311|   15.2| 395.6|12.43|22.9|[0.08829,12.5,7.8...|
    |0.14455|12.5| 7.87|   0|0.524|6.172| 96.1|5.9505|  5|311|   15.2| 396.9|19.15|27.1|[0.14455,12.5,7.8...|
    |0.21124|12.5| 7.87|   0|0.524|5.631|100.0|6.0821|  5|311|   15.2|386.63|29.93|16.5|[0.21124,12.5,7.8...|
    |0.17004|12.5| 7.87|   0|0.524|6.004| 85.9|6.5921|  5|311|   15.2|386.71| 17.1|18.9|[0.17004,12.5,7.8...|
    |0.22489|12.5| 7.87|   0|0.524|6.377| 94.3|6.3467|  5|311|   15.2|392.52|20.45|15.0|[0.22489,12.5,7.8...|
    |0.11747|12.5| 7.87|   0|0.524|6.009| 82.9|6.2267|  5|311|   15.2| 396.9|13.27|18.9|[0.11747,12.5,7.8...|
    |0.09378|12.5| 7.87|   0|0.524|5.889| 39.0|5.4509|  5|311|   15.2| 390.5|15.71|21.7|[0.09378,12.5,7.8...|
    |0.62976| 0.0| 8.14|   0|0.538|5.949| 61.8|4.7075|  4|307|   21.0| 396.9| 8.26|20.4|[0.62976,0.0,8.14...|
    |0.63796| 0.0| 8.14|   0|0.538|6.096| 84.5|4.4619|  4|307|   21.0|380.02|10.26|18.2|[0.63796,0.0,8.14...|
    |0.62739| 0.0| 8.14|   0|0.538|5.834| 56.5|4.4986|  4|307|   21.0|395.62| 8.47|19.9|[0.62739,0.0,8.14...|
    |1.05393| 0.0| 8.14|   0|0.538|5.935| 29.3|4.4986|  4|307|   21.0|386.85| 6.58|23.1|[1.05393,0.0,8.14...|
    | 0.7842| 0.0| 8.14|   0|0.538| 5.99| 81.7|4.2579|  4|307|   21.0|386.75|14.67|17.5|[0.7842,0.0,8.14,...|
    |0.80271| 0.0| 8.14|   0|0.538|5.456| 36.6|3.7965|  4|307|   21.0|288.99|11.69|20.2|[0.80271,0.0,8.14...|
    | 0.7258| 0.0| 8.14|   0|0.538|5.727| 69.5|3.7965|  4|307|   21.0|390.95|11.28|18.2|[0.7258,0.0,8.14,...|
    +-------+----+-----+----+-----+-----+-----+------+---+---+-------+------+-----+----+--------------------+
    only showing top 20 rows
    


## 훈련용과 테스트용 데이터를 나누고 Linear Regression 모델을 하나 만든다


```python
train, test = data_2.randomSplit([0.7, 0.3])
```


```python
from pyspark.ml.regression import LinearRegression

algo = LinearRegression(featuresCol="features", labelCol="medv")
model = algo.fit(train)
```

## 모델 성능 측정


```python
evaluation_summary = model.evaluate(test)
```


```python
evaluation_summary
```




    <pyspark.ml.regression.LinearRegressionSummary at 0x7f750d748f98>




```python
evaluation_summary.meanAbsoluteError
```




    3.4934514393357463




```python
evaluation_summary.rootMeanSquaredError
```




    5.197239963691187




```python
evaluation_summary.r2
```




    0.7319277591330025



## 모델 예측값 살펴보기 


```python
predictions = model.transform(test)
```


```python
predictions.show()
```

    +-------+----+-----+----+------+-----+----+------+---+---+-------+------+-----+----+--------------------+------------------+
    |   crim|  zn|indus|chas|   nox|   rm| age|   dis|rad|tax|ptratio|     b|lstat|medv|            features|        prediction|
    +-------+----+-----+----+------+-----+----+------+---+---+-------+------+-----+----+--------------------+------------------+
    |0.00632|18.0| 2.31|   0| 0.538|6.575|65.2|  4.09|  1|296|   15.3| 396.9| 4.98|24.0|[0.00632,18.0,2.3...| 29.49164934817386|
    | 0.0136|75.0|  4.0|   0|  0.41|5.888|47.6|7.3197|  3|469|   21.1| 396.9| 14.8|18.9|[0.0136,75.0,4.0,...| 15.97366486003298|
    |0.01439|60.0| 2.93|   0| 0.401|6.604|18.8|6.2196|  1|265|   15.6| 376.7| 4.38|29.1|[0.01439,60.0,2.9...| 31.53442460758399|
    |0.01965|80.0| 1.76|   0| 0.385| 6.23|31.5|9.0892|  1|241|   18.2| 341.6|12.93|20.1|[0.01965,80.0,1.7...|20.894740919794515|
    |0.02009|95.0| 2.68|   0|0.4161|8.034|31.9| 5.118|  4|224|   14.7|390.55| 2.88|50.0|[0.02009,95.0,2.6...|43.293958640441325|
    |0.02055|85.0| 0.74|   0|  0.41|6.383|35.7|9.1876|  2|313|   17.3| 396.9| 5.77|24.7|[0.02055,85.0,0.7...| 25.04819322105935|
    |0.02875|28.0|15.04|   0| 0.464|6.211|28.9|3.6659|  4|270|   18.2|396.33| 6.21|25.0|[0.02875,28.0,15....| 28.79867786735999|
    |0.03049|55.0| 3.78|   0| 0.484|6.874|28.1|6.4654|  5|370|   17.6|387.97| 4.61|31.2|[0.03049,55.0,3.7...| 28.42598794887642|
    |0.03113| 0.0| 4.39|   0| 0.442|6.014|48.5|8.0136|  3|352|   18.8|385.64|10.53|17.5|[0.03113,0.0,4.39...| 16.22819860106943|
    | 0.0315|95.0| 1.47|   0| 0.403|6.975|15.3|7.6534|  3|402|   17.0| 396.9| 4.56|34.9|[0.0315,95.0,1.47...|30.248456536297894|
    |0.03237| 0.0| 2.18|   0| 0.458|6.998|45.8|6.0622|  3|222|   18.7|394.63| 2.94|33.4|[0.03237,0.0,2.18...| 28.01438147295317|
    |0.03306| 0.0| 5.19|   0| 0.515|6.059|37.3|4.8122|  5|224|   20.2|396.14| 8.51|20.6|[0.03306,0.0,5.19...|22.151794828698733|
    |0.03359|75.0| 2.95|   0| 0.428|7.024|15.8|5.4011|  3|252|   18.3|395.62| 1.98|34.9|[0.03359,75.0,2.9...| 34.32733571024841|
    |0.03466|35.0| 6.06|   0|0.4379|6.031|23.3|6.6407|  1|304|   16.9|362.25| 7.83|19.4|[0.03466,35.0,6.0...| 23.27875714087658|
    |0.03502|80.0| 4.95|   0| 0.411|6.861|27.9|5.1167|  4|245|   19.2| 396.9| 3.33|28.5|[0.03502,80.0,4.9...| 33.65946129021973|
    |0.03548|80.0| 3.64|   0| 0.392|5.876|19.1|9.2203|  1|315|   16.4|395.18| 9.25|20.9|[0.03548,80.0,3.6...|22.043255459132656|
    |0.03578|20.0| 3.33|   0|0.4429| 7.82|64.5|4.6947|  5|216|   14.9|387.31| 3.76|45.4|[0.03578,20.0,3.3...| 38.19010144015952|
    |0.03584|80.0| 3.37|   0| 0.398| 6.29|17.8|6.6115|  4|337|   16.1| 396.9| 4.67|23.5|[0.03584,80.0,3.3...|30.507918460123335|
    |0.03705|20.0| 3.33|   0|0.4429|6.968|37.2|5.2447|  5|216|   14.9|392.23| 4.59|35.4|[0.03705,20.0,3.3...| 33.96252998441784|
    |0.04297|52.5| 5.32|   0| 0.405|6.565|22.9|7.3172|  6|293|   16.6|371.72| 9.51|24.8|[0.04297,52.5,5.3...|27.135407022957597|
    +-------+----+-----+----+------+-----+----+------+---+---+-------+------+-----+----+--------------------+------------------+
    only showing top 20 rows
    



```python
predictions.select(predictions.columns[13:]).show()
```

    +----+--------------------+------------------+
    |medv|            features|        prediction|
    +----+--------------------+------------------+
    |24.0|[0.00632,18.0,2.3...| 29.49164934817386|
    |18.9|[0.0136,75.0,4.0,...| 15.97366486003298|
    |29.1|[0.01439,60.0,2.9...| 31.53442460758399|
    |20.1|[0.01965,80.0,1.7...|20.894740919794515|
    |50.0|[0.02009,95.0,2.6...|43.293958640441325|
    |24.7|[0.02055,85.0,0.7...| 25.04819322105935|
    |25.0|[0.02875,28.0,15....| 28.79867786735999|
    |31.2|[0.03049,55.0,3.7...| 28.42598794887642|
    |17.5|[0.03113,0.0,4.39...| 16.22819860106943|
    |34.9|[0.0315,95.0,1.47...|30.248456536297894|
    |33.4|[0.03237,0.0,2.18...| 28.01438147295317|
    |20.6|[0.03306,0.0,5.19...|22.151794828698733|
    |34.9|[0.03359,75.0,2.9...| 34.32733571024841|
    |19.4|[0.03466,35.0,6.0...| 23.27875714087658|
    |28.5|[0.03502,80.0,4.9...| 33.65946129021973|
    |20.9|[0.03548,80.0,3.6...|22.043255459132656|
    |45.4|[0.03578,20.0,3.3...| 38.19010144015952|
    |23.5|[0.03584,80.0,3.3...|30.507918460123335|
    |35.4|[0.03705,20.0,3.3...| 33.96252998441784|
    |24.8|[0.04297,52.5,5.3...|27.135407022957597|
    +----+--------------------+------------------+
    only showing top 20 rows
    



```python
model.save("boston_housing_model")
```


```
!ls boston_housing_model
```

    data  metadata



```
!ls -tl boston_housing_model
```

    total 8
    drwxr-xr-x 2 root root 4096 Feb  4 06:50 data
    drwxr-xr-x 2 root root 4096 Feb  4 06:50 metadata



```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive



```python
model_save_name = "boston_housing_model"
path = F"/content/gdrive/My Drive/boston_housing_model2" 
model.save(path)
```


```python
from pyspark.ml.regression import LinearRegressionModel

loaded_model = LinearRegressionModel.load(path)  # "boston_housing_model")
```


```python
predictions2 = loaded_model.transform(test)
```


```python
predictions2.select(predictions.columns[13:]).show()
```

    +----+--------------------+------------------+
    |medv|            features|        prediction|
    +----+--------------------+------------------+
    |24.0|[0.00632,18.0,2.3...| 29.49164934817386|
    |18.9|[0.0136,75.0,4.0,...| 15.97366486003298|
    |29.1|[0.01439,60.0,2.9...| 31.53442460758399|
    |20.1|[0.01965,80.0,1.7...|20.894740919794515|
    |50.0|[0.02009,95.0,2.6...|43.293958640441325|
    |24.7|[0.02055,85.0,0.7...| 25.04819322105935|
    |25.0|[0.02875,28.0,15....| 28.79867786735999|
    |31.2|[0.03049,55.0,3.7...| 28.42598794887642|
    |17.5|[0.03113,0.0,4.39...| 16.22819860106943|
    |34.9|[0.0315,95.0,1.47...|30.248456536297894|
    |33.4|[0.03237,0.0,2.18...| 28.01438147295317|
    |20.6|[0.03306,0.0,5.19...|22.151794828698733|
    |34.9|[0.03359,75.0,2.9...| 34.32733571024841|
    |19.4|[0.03466,35.0,6.0...| 23.27875714087658|
    |28.5|[0.03502,80.0,4.9...| 33.65946129021973|
    |20.9|[0.03548,80.0,3.6...|22.043255459132656|
    |45.4|[0.03578,20.0,3.3...| 38.19010144015952|
    |23.5|[0.03584,80.0,3.3...|30.507918460123335|
    |35.4|[0.03705,20.0,3.3...| 33.96252998441784|
    |24.8|[0.04297,52.5,5.3...|27.135407022957597|
    +----+--------------------+------------------+
    only showing top 20 rows
    