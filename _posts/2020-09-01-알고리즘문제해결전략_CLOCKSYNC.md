---
category: Algorithm
tags : [알고리즘문제해결전략, CLOCKSYNC]
---

## CLOCKSYNC 문제 개요

#### 문제
![example](http://algospot.com/media/judge-attachments/d3428bd7a9a425b85c9d3c042b674728/clocks.PNG "exmaple")  
그림과 같이 4 x 4 개의 격자 형태로 배치된 16개의 시계가 있다. 이 시계들은 모두 12시, 3시, 6시, 혹은 9시를 가리키고 있다. 이 시계들이 모두 12시를 가리키도록 바꾸고 싶다.  

시계의 시간을 조작하는 유일한 방법은 모두 10개 있는 스위치들을 조작하는 것으로, 각 스위치들은 모두 적게는 3개에서 많게는 5개의 시계에 연결되어 있다. 한 스위치를 누를 때마다, 해당 스위치와 연결된 시계들의 시간은 3시간씩 앞으로 움직인다. 스위치들과 그들이 연결된 시계들의 목록은 다음과 같다.  

|스위치 번호|연결된 시계들|
|--------|----------|
|0|0, 1, 2|
|1|3, 7, 9, 11|
|2|4, 10, 14, 15|
|3|0, 4, 5, 6, 7|
|4|6, 7, 8, 10, 12|
|5|0, 2, 14, 15|
|6|3, 14, 15|
|7|4, 5, 7, 14, 15|
|8|1, 2, 3, 4, 5|
|9|3, 4, 5, 9, 13|  

시계들은 맨 윗줄부터, 왼쪽에서 오른쪽으로 순서대로 번호가 매겨졌다고 가정하자. 시계들이 현재 가리키는 시간들이 주어졌을 때, 모든 시계를 12시로 돌리기 위해 최소한 눌러야 할 스위치의 수를 계산하는 프로그램을 작성하시오.  
<br>

#### 입력
첫 줄에 테스트 케이스의 개수 C (<= 30) 가 주어진다.
각 테스트 케이스는 한 줄에 16개의 정수로 주어지며, 각 정수는 0번부터 15번까지 각 시계가 가리키고 있는 시간을 12, 3, 6, 9 중 하나로 표현한다. 
<br>

#### 출력
각 테스트 케이스당 한 줄을 출력한다. 시계들을 모두 12시로 돌려놓기 위해 눌러야 할 스위치의 최소 수를 출력한다. 만약 이것이 불가능할 경우 -1 을 출력한다.  
<br>

#### 예제 입력
2  
12 6 6 6 6 6 12 12 12 12 12 12 12 12 12 12  
12 9 3 12 6 6 9 3 12 9 12 9 12 12 6 6  

#### 예제 출력
2  
9

출처: [Algospot](https://algospot.com/judge/problem/read/CLOCKSYNC)

## 구현

#### 설계
전에 구현한 PICNIC이나 BOARDCOVER와 같이 완전 탐색 방식으로 하여 구현하려고 하였다.

1. 시계의 스위치를 눌렀을 때, 4번 누르면 다시 원래의 상태로 돌아오기 때문에, 3번 누르는 경우까지만 생각해 보면 된다.
2. 또한, 여타의 다른 완전 탐색 문제처럼 스위치를 누르는 순서는 중요하지 않다. 3번을 누르고 9번을 누르는 것과 9번을 누르고 3번을 누르는 경우의 상태는 같다. 


#### 전체 코드


```cpp
#include <iostream>

using namespace std;

int TC, BestCase;
const int Switch[10][16]=
    {
        {3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,3,0,0,0,3,0,3,0,3,0,0,0,0},
        {0,0,0,0,3,0,0,0,0,0,3,0,0,0,3,3},
        {3,0,0,0,3,3,3,3,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,3,3,3,0,3,0,3,0,0,0},
        {3,0,3,0,0,0,0,0,0,0,0,0,0,0,3,3},
        {0,0,0,3,0,0,0,0,0,0,0,0,0,0,3,3},
        {0,0,0,0,3,3,0,3,0,0,0,0,0,0,3,3},
        {0,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,3,3,3,0,0,0,3,0,0,0,3,0,0},
    };

bool IsAllTwelve(int* Clock)
{
    for(int i = 0; i < 16; i++)
    {
        if(Clock[i] != 12)
            return 0;
    }
    return 1;
}

void ClockSync(int count, int point_switch, int switch_count, int* Clock)
{
    if(BestCase != 0 && BestCase <= count)
        return;

    if(IsAllTwelve(Clock))
    {
        if(BestCase == 0)
            BestCase = count;
        else if(BestCase > count)
            BestCase = count;
        return;
    }

    if(switch_count < 4)
    {
        for(int i = 0; i < 16; i++)
        {
            Clock[i] += Switch[point_switch][i];
            if(Clock[i] > 12)
                Clock[i] -= 12;
        }
        ClockSync(++count, point_switch, ++switch_count, Clock);
        --count;
        for(int i = 0; i < 16; i++)
        {
            Clock[i] -= Switch[point_switch][i];
            if(Clock[i] == 0)
                Clock[i] = 12;
        }
    }

    for(int i = point_switch + 1; i < 10; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            Clock[j] += Switch[i][j];
            if(Clock[j] > 12)
                Clock[j] -= 12;
        }
        ClockSync(++count, i, 1, Clock);
        --count;
        for(int j = 0; j < 16; j++)
        {
            Clock[j] -= Switch[i][j];
            if(Clock[j] == 0)
                Clock[j] = 12;
        }
    }
}


int main(void)
{   
    cin >> TC;
    for(int i = 0; i < TC; i++)
    {
        int Clock[16];
        for(int j = 0; j < 16; j++)
        {
            cin >> Clock[j];
        }

        BestCase = 0;
        int ret;
        if(IsAllTwelve(Clock))
        {
            ret = 0;
        }
        else
        {
            ClockSync(0, 0, 0, Clock);
            if(BestCase == 0)
                ret = -1;
            else
            {
                ret = BestCase;
            }   
        }
        cout<<ret<<endl;   
    }
    return 0;
}
```
<br>

#### main함수와 보조함수(IsAllTwelve) 및 Switch배열
```cpp

const int Switch[10][16]=
    {
        {3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,3,0,0,0,3,0,3,0,3,0,0,0,0},
        {0,0,0,0,3,0,0,0,0,0,3,0,0,0,3,3},
        {3,0,0,0,3,3,3,3,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,3,3,3,0,3,0,3,0,0,0},
        {3,0,3,0,0,0,0,0,0,0,0,0,0,0,3,3},
        {0,0,0,3,0,0,0,0,0,0,0,0,0,0,3,3},
        {0,0,0,0,3,3,0,3,0,0,0,0,0,0,3,3},
        {0,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,3,3,3,0,0,0,3,0,0,0,3,0,0},
    };

bool IsAllTwelve(int* Clock)
{
    for(int i = 0; i < 16; i++)
    {
        if(Clock[i] != 12)
            return 0;
    }
    return 1;
}

int main(void)
{   
    cin >> TC;
    for(int i = 0; i < TC; i++)
    {
        int Clock[16];
        for(int j = 0; j < 16; j++)
        {
            cin >> Clock[j];
        }

        BestCase = 0;
        int ret;
        if(IsAllTwelve(Clock))
        {
            ret = 0;
        }
        else
        {
            ClockSync(0, 0, 0, Clock);
            if(BestCase == 0)
                ret = -1;
            else
            {
                ret = BestCase;
            }   
        }
        cout<<ret<<endl;   
    }
    return 0;
}
```

Switch배열의 경우 행은 스위치번호를 나타내며, 열의 경우에는 0일경우 스위치가 눌러졌을 때, 변하지 않는 시계 번호이고 3인 경우에는 변하는 시계의 번호이다. 예를들어 'Switch[1][3]'의 경우 1번 스위치가 3번 시계를 움직일 수 있는지 없는지를 보여준다. 이경우 3번 시계가 1번 스위치와 연결 되어있으므로 3이 된다.  
IsAllTwelve함수는 모든 시계가 12를 가리키는지를 확인하는 함수이다.  
main함수는 이전의 구현과 크게 다르지 않다. Testcase룰 받고 그 수 만큼 함수를 실행시킨다. 한가지 생각했던 점은 처음부터 모든 시계가 12를 가리키는 경우 BestCase가 0이 되므로, 이때 -1이 return된다. 따라서 이경우는 미리 확인하여 0을 return하도록 하였다.(실제 채점에서 의미가 있었는지 모르곘다.)  
Switch배열의 초기 구현은 시계와 스위치가 연결되어있는 경우 3이 아니라 1을 가지도록 하고 int가 아닌 bool의 배열이었지만, 이번 문제에서의 가장 큰 문제는 시간이었다. 채점을 했을 때, 자꾸 시간초과가 떠서 최대한 시간을 줄이려고 하였다. 그 과정에서 Switch배열의 값이 1인지 0인지 if문을 통해 확인하고, 1인 경우 시계의 값들을 변경해 주는 방식의 구현이 단순히 3을 더해주고 빼주어서 시계의 값들을 변경하는 것으로 변경 되게 되었다.  

#### Recursive 함수(ClockSync)

```cpp
void ClockSync(int count, int point_switch, int switch_count, int* Clock)
{
    if(BestCase != 0 && BestCase <= count)
        return;

    if(IsAllTwelve(Clock))
    {
        if(BestCase == 0)
            BestCase = count;
        else if(BestCase > count)
            BestCase = count;
        return;
    }

    if(switch_count < 4)
    {
        for(int i = 0; i < 16; i++)
        {
            Clock[i] += Switch[point_switch][i];
            if(Clock[i] > 12)
                Clock[i] -= 12;
        }
        ClockSync(++count, point_switch, ++switch_count, Clock);
        --count;
        for(int i = 0; i < 16; i++)
        {
            Clock[i] -= Switch[point_switch][i];
            if(Clock[i] == 0)
                Clock[i] = 12;
        }
    }

    for(int i = point_switch + 1; i < 10; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            Clock[j] += Switch[i][j];
            if(Clock[j] > 12)
                Clock[j] -= 12;
        }
        ClockSync(++count, i, 1, Clock);
        --count;
        for(int j = 0; j < 16; j++)
        {
            Clock[j] -= Switch[i][j];
            if(Clock[j] == 0)
                Clock[j] = 12;
        }
    }
}
```

이번 문제에서의 기저 사례는 당연히 모든 시계의 값이 12가 되는 경우이다. 그에 앞서서 탐색 시간을 절약하기 위해 BestCase가 이미 하나 들어왔고, 그 값을 넘어서는 경우는 더이상 살펴볼 필요가 없기 때문에, 함수를 끝내도록 구현하였다. 처음 구현에는 없는 부분이었지만, 시간 절약을 위해서 추가하였다. 나머지 부분은 모든 Recursive함수를 통한 완전 탐색의 알고리즘 처럼 구현하였다. Recursive함수를 불러올 때, 현재 어느 스위치를 조작하고 있는지, 또 몇번 조작했는지의 값들을 계속 넘겨주도록 하였다.  
  
<br>


## 정리
1. 대부분의 완전 탐색 알고리즘을 구현할 때, 순서와 상관이 있는지 없는지를 반드시 먼저 살펴보아야 한다는 것을 기억해야겠다. 처음에 순서와 상관이 없다는 것을 알기까지의 구현이 너무 막막하게 느껴졌었는데, 이러한 특징 하나만으로도 구현을 쉽게 할 수 있게 만들어 주었다.

2. 처음으로 채점할 때, 시간초과가 나와서 당황했었다. 답이 틀린 경우보다 시간이 초과되는 경우가 더 막막하게 느껴지는데, 앞으로 이를 더 신경써서 구현해야겠다.

지금까지 구현했던 것보다 난이도가 있는 문제였던 것 같다. 구현 자체도 오래걸렸을 뿐만 아니라 시간을 맞추는데에도 많은 시간이 들었다.  

끝!