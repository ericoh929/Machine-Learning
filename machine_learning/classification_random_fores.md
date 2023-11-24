해당 파일은 synthetic data임.


```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as randomforest
from sklearn.tree import DecisionTreeClassifier as decisiontree
import matplotlib.pyplot as plt
from tqdm import tqdm
```

### 데이터 전처리


```python
df = pd.read_excel('/content/sample_data/private_file.xlsx', header = 1)
df = df[:100]
df = df.rename(columns = {'Unnamed: 0': '이름', 'Unnamed: 13': '실제등급', 'Unnamed: 14': '종속변수'})
df = df.set_index('이름')
df = df.astype({'종속변수':int}) #타입 정수로 바꿔주기
df

```





  <div id="df-950dc16e-288b-4da4-b424-ac539cc8ac74" class="colab-df-container">
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
      <th>병무청</th>
      <th>훈련소</th>
      <th>자대</th>
      <th>관계유형</th>
      <th>자살시도</th>
      <th>경력</th>
      <th>가정 및 이성</th>
      <th>건강</th>
      <th>훈련소 의견</th>
      <th>상담</th>
      <th>자해,자살</th>
      <th>병영폭력</th>
      <th>실제등급</th>
      <th>종속변수</th>
    </tr>
    <tr>
      <th>이름</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>병장 강O건</th>
      <td>양호</td>
      <td>양호</td>
      <td>양호</td>
      <td>평균형</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>일반 용사</td>
      <td>0</td>
    </tr>
    <tr>
      <th>병장 강O경</th>
      <td>사고예측</td>
      <td>양호</td>
      <td>양호</td>
      <td>인기형</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>가정(이혼)</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>배려 용사</td>
      <td>1</td>
    </tr>
    <tr>
      <th>병장 강O규</th>
      <td>양호</td>
      <td>양호</td>
      <td>양호</td>
      <td>주변형</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>부정적</td>
      <td>부정적</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>배려 용사</td>
      <td>1</td>
    </tr>
    <tr>
      <th>병장 강O덕</th>
      <td>양호</td>
      <td>양호</td>
      <td>양호</td>
      <td>인기형</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>자살시도</td>
      <td>미해당</td>
      <td>현역복무부적합 대상</td>
      <td>3</td>
    </tr>
    <tr>
      <th>병장 강O변</th>
      <td>양호</td>
      <td>양호</td>
      <td>양호</td>
      <td>평균형</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>자해시도</td>
      <td>미해당</td>
      <td>도움용사</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>이병 허O우</th>
      <td>양호</td>
      <td>양호</td>
      <td>양호</td>
      <td>인기형</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>부정적</td>
      <td>미해당</td>
      <td>피해</td>
      <td>배려 용사</td>
      <td>1</td>
    </tr>
    <tr>
      <th>이병 허O진</th>
      <td>양호</td>
      <td>R(즉각의뢰)</td>
      <td>양호</td>
      <td>평균형</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>비만</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>배려 용사</td>
      <td>1</td>
    </tr>
    <tr>
      <th>이병 허O철</th>
      <td>특수</td>
      <td>Y(주의)</td>
      <td>양호</td>
      <td>인기형</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>비만</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>도움 용사</td>
      <td>2</td>
    </tr>
    <tr>
      <th>이병 허O후</th>
      <td>양호</td>
      <td>양호</td>
      <td>양호</td>
      <td>인기형</td>
      <td>해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>배려 용사</td>
      <td>1</td>
    </tr>
    <tr>
      <th>이병 허O훈</th>
      <td>양호</td>
      <td>양호</td>
      <td>양호</td>
      <td>인기형</td>
      <td>해당</td>
      <td>미해당</td>
      <td>가정(이혼)</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>미해당</td>
      <td>배려 용사</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 14 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-950dc16e-288b-4da4-b424-ac539cc8ac74')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-950dc16e-288b-4da4-b424-ac539cc8ac74 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-950dc16e-288b-4da4-b424-ac539cc8ac74');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4d9845e9-7668-45de-bc96-41f1f58084e6">
  <button class="colab-df-quickchart" onclick="quickchart('df-4d9845e9-7668-45de-bc96-41f1f58084e6')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4d9845e9-7668-45de-bc96-41f1f58084e6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df['자해,자살'].unique()
```




    array(['미해당', '자살시도', '자해시도', '자살 시도', '자해 시도'], dtype=object)




```python
df['자해,자살'] = [i.replace(' ', '') for i in df['자해,자살']]
```


```python
df['병영폭력'].unique()
```




    array(['미해당', '피해', '가해'], dtype=object)




```python
df['종속변수'].unique()
```




    array([0, 1, 3, 2])




```python
df.columns #컬럼 이름 확인
```




    Index(['병무청', '훈련소', '자대', '관계유형', '자살시도', '경력', '가정 및 이성', '건강', '훈련소 의견',
           '상담', '자해,자살', '병영폭력', '실제등급', '종속변수'],
          dtype='object')




```python
X = pd.get_dummies(df, columns = ['병무청', '훈련소', '자대', '관계유형', '자살시도', '경력', '가정 및 이성', '건강', '훈련소 의견', '상담', '자해,자살', '병영폭력']) #one-hot vector로 만들어주기
X.drop(columns = ['실제등급', '종속변수'], inplace = True)
y = df['종속변수']
```


```python
X
```





  <div id="df-6d67ee7a-8043-4e99-a8ce-2c88ba70af7c" class="colab-df-container">
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
      <th>병무청_사고예측</th>
      <th>병무청_양호</th>
      <th>병무청_정밀</th>
      <th>병무청_특수</th>
      <th>훈련소_R(즉각의뢰)</th>
      <th>훈련소_Y(주의)</th>
      <th>훈련소_양호</th>
      <th>자대_부적응</th>
      <th>자대_양호</th>
      <th>자대_위험</th>
      <th>...</th>
      <th>훈련소 의견_미해당</th>
      <th>훈련소 의견_부정적</th>
      <th>상담_미해당</th>
      <th>상담_부정적</th>
      <th>자해,자살_미해당</th>
      <th>자해,자살_자살시도</th>
      <th>자해,자살_자해시도</th>
      <th>병영폭력_가해</th>
      <th>병영폭력_미해당</th>
      <th>병영폭력_피해</th>
    </tr>
    <tr>
      <th>이름</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>병장 강O건</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>병장 강O경</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>병장 강O규</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>병장 강O덕</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>병장 강O변</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>이병 허O우</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>이병 허O진</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>이병 허O철</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>이병 허O후</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>이병 허O훈</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 39 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6d67ee7a-8043-4e99-a8ce-2c88ba70af7c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6d67ee7a-8043-4e99-a8ce-2c88ba70af7c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6d67ee7a-8043-4e99-a8ce-2c88ba70af7c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-fe6762c0-9e5e-4532-a42d-2a318eee710d">
  <button class="colab-df-quickchart" onclick="quickchart('df-fe6762c0-9e5e-4532-a42d-2a318eee710d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-fe6762c0-9e5e-4532-a42d-2a318eee710d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
X.describe()
```





  <div id="df-06486245-df34-49e8-85fd-690942dc382c" class="colab-df-container">
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
      <th>병무청_사고예측</th>
      <th>병무청_양호</th>
      <th>병무청_정밀</th>
      <th>병무청_특수</th>
      <th>훈련소_R(즉각의뢰)</th>
      <th>훈련소_Y(주의)</th>
      <th>훈련소_양호</th>
      <th>자대_부적응</th>
      <th>자대_양호</th>
      <th>자대_위험</th>
      <th>...</th>
      <th>훈련소 의견_미해당</th>
      <th>훈련소 의견_부정적</th>
      <th>상담_미해당</th>
      <th>상담_부정적</th>
      <th>자해,자살_미해당</th>
      <th>자해,자살_자살시도</th>
      <th>자해,자살_자해시도</th>
      <th>병영폭력_가해</th>
      <th>병영폭력_미해당</th>
      <th>병영폭력_피해</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.050000</td>
      <td>0.780000</td>
      <td>0.090000</td>
      <td>0.08000</td>
      <td>0.100000</td>
      <td>0.08000</td>
      <td>0.820000</td>
      <td>0.08000</td>
      <td>0.880000</td>
      <td>0.040000</td>
      <td>...</td>
      <td>0.880000</td>
      <td>0.120000</td>
      <td>0.770000</td>
      <td>0.230000</td>
      <td>0.870000</td>
      <td>0.060000</td>
      <td>0.070000</td>
      <td>0.050000</td>
      <td>0.820000</td>
      <td>0.130000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.219043</td>
      <td>0.416333</td>
      <td>0.287623</td>
      <td>0.27266</td>
      <td>0.301511</td>
      <td>0.27266</td>
      <td>0.386123</td>
      <td>0.27266</td>
      <td>0.326599</td>
      <td>0.196946</td>
      <td>...</td>
      <td>0.326599</td>
      <td>0.326599</td>
      <td>0.422953</td>
      <td>0.422953</td>
      <td>0.337998</td>
      <td>0.238683</td>
      <td>0.256432</td>
      <td>0.219043</td>
      <td>0.386123</td>
      <td>0.337998</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 39 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-06486245-df34-49e8-85fd-690942dc382c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-06486245-df34-49e8-85fd-690942dc382c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-06486245-df34-49e8-85fd-690942dc382c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e592f237-6eed-4daf-bd43-6ce2c3ee5378">
  <button class="colab-df-quickchart" onclick="quickchart('df-e592f237-6eed-4daf-bd43-6ce2c3ee5378')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e592f237-6eed-4daf-bd43-6ce2c3ee5378 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
X.isna().sum()
```




    병무청_사고예측          0
    병무청_양호            0
    병무청_정밀            0
    병무청_특수            0
    훈련소_R(즉각의뢰)       0
    훈련소_Y(주의)         0
    훈련소_양호            0
    자대_부적응            0
    자대_양호             0
    자대_위험             0
    관계유형_배척형          0
    관계유형_인기형          0
    관계유형_주변형          0
    관계유형_평균형          0
    자살시도_미해당          0
    자살시도_해당           0
    경력_미해당            0
    경력_범법             0
    경력_폭력             0
    가정 및 이성_가정(이혼)    0
    가정 및 이성_미해당       0
    가정 및 이성_이성(결별)    0
    가정 및 이성_평범        0
    건강_갑상선 저하         0
    건강_고혈압            0
    건강_미해당            0
    건강_비만             0
    건강_심장통증           0
    건강_폐렴             0
    훈련소 의견_미해당        0
    훈련소 의견_부정적        0
    상담_미해당            0
    상담_부정적            0
    자해,자살_미해당         0
    자해,자살_자살시도        0
    자해,자살_자해시도        0
    병영폭력_가해           0
    병영폭력_미해당          0
    병영폭력_피해           0
    dtype: int64



### 데이터 나누기


```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import seaborn as sns
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
```


```python
plt.figure(figsize = (7, 5))
plt.subplot(121)
plt.bar(y_train.value_counts().index, y_train.value_counts())
plt.subplot(122)
plt.bar(y_test.value_counts().index, y_test.value_counts())
```




    <BarContainer object of 4 artists>




    
![png](classification_random_fores_files/classification_random_fores_16_1.png)
    



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)
```


```python
y_train.value_counts()
```




    1    34
    0    15
    2    14
    3     7
    Name: 종속변수, dtype: int64




```python
plt.figure(figsize = (7, 5))
plt.subplot(121)
plt.bar(y_train.value_counts().index, y_train.value_counts())
plt.subplot(122)
plt.bar(y_test.value_counts().index, y_test.value_counts())
```




    <BarContainer object of 4 artists>




    
![png](classification_random_fores_files/classification_random_fores_19_1.png)
    


### RandomForest Model

### 그냥 일반적으로 학습시키면 아래와 같이 할 수 있음


```python
model_fr = randomforest(random_state = 1, n_estimators = 5, max_features = 4) #random state = random 성 부여, n_estimators 트리 개수, max features 무작위 선택 노드 수
model_fr.fit(X_train, y_train) #학습
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(max_features=4, n_estimators=5, random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_features=4, n_estimators=5, random_state=1)</pre></div></div></div></div></div>




```python
y_train_pred = model_fr.predict(X_train) # 훈련 데이터셋 예측
y_pred = model_fr.predict(X_test) # 테스트 데이터셋 예측
```


```python
train_score = accuracy_score(y_train, y_train_pred)
test_score = accuracy_score(y_test, y_pred)

print('train score: ', train_score)
print('test score: ', test_score)
```

    train score:  0.9714285714285714
    test score:  0.6


### 하지만, 데이터가 100개 밖에 안되는 이유로 K-fold cross-validation 실시


```python
x = []
accuracy_list = []
best_accuracy = [0, 0]
best_model = None
for n in tqdm(range(200)):
    x.append(n)
    model_fr = randomforest(random_state = 1, n_estimators = 1 + n, max_features = 4)
    accuracy = cross_val_score(model_fr, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
    if accuracy >= best_accuracy[1]:
        best_accuracy[1] = accuracy
        best_accuracy[0] = n
        best_model = model_fr
    accuracy_list.append(accuracy)
print('\nbest number of trees: ', best_accuracy[0], 'best accuracy: ', best_accuracy[1])

```

    100%|██████████| 200/200 [03:07<00:00,  1.07it/s]

    
    best number of trees:  17 best accuracy:  0.7142857142857142


    



```python
model_fr = randomforest(random_state = 1, n_estimators = 17, max_features = 4) #random state = random 성 부여, n_estimators 트리 개수, max features 무작위 선택 노드 수
model_fr.fit(X_train, y_train) #학습

y_pred = model_fr.predict(X_test)
test_score = accuracy_score(y_test, y_pred)
print('test accuracy: ', test_score)
```

    test accuracy:  0.7333333333333333



```python
plt.figure(figsize = (10, 5))
plt.plot(x, accuracy_list, marker = 'o', linestyle = 'dashed')
plt.axvline(best_accuracy[0], linestyle = 'dashed', color = 'r')
plt.title('Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Accuracy')
plt.show()
```


    
![png](classification_random_fores_files/classification_random_fores_28_0.png)
    


### 데이터 편향성 고려 Stratified K-fold cross validation 사용


```python
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

x = []
accuracy_list = []
best_accuracy = [0, 0]
best_model = None
for n in tqdm(range(200)):
    x.append(n)
    model_fr = randomforest(random_state = 1, n_estimators = 1 + n, max_features = 4)
    accuracy = cross_val_score(model_fr, X, y, scoring = 'accuracy', cv = skf).mean()
    if accuracy >= best_accuracy[1]:
        best_accuracy[1] = accuracy
        best_accuracy[0] = n
        best_model = model_fr
    accuracy_list.append(accuracy)
print('\nbest number of trees: ', best_accuracy[0], 'best accuracy: ', best_accuracy[1])
```

    100%|██████████| 200/200 [03:04<00:00,  1.09it/s]

    
    best number of trees:  191 best accuracy:  0.75


    



```python
model_fr = randomforest(random_state = 1, n_estimators = 191, max_features = 4) #random state = random 성 부여, n_estimators 트리 개수, max features 무작위 선택 노드 수
model_fr.fit(X_train, y_train) #학습

y_pred = model_fr.predict(X_test)
test_score = accuracy_score(y_test, y_pred)
print('test accuracy: ', test_score)
```

    test accuracy:  0.7



```python
plt.figure(figsize = (10, 5))
plt.plot(x, accuracy_list, marker = 'o', linestyle = 'dashed')
plt.axvline(best_accuracy[0], linestyle = 'dashed', color = 'r')
plt.title('Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Accuracy')
plt.show()
```


    
![png](classification_random_fores_files/classification_random_fores_32_0.png)
    


### 아래는 참고


```python
model_fr.feature_importances_
```




    array([0.00732617, 0.03263134, 0.01322627, 0.01668803, 0.01706519,
           0.01334391, 0.04631327, 0.01835436, 0.02716951, 0.01510435,
           0.01498149, 0.02710574, 0.02746434, 0.02251347, 0.03280103,
           0.02678933, 0.02814757, 0.03193347, 0.01045033, 0.00726048,
           0.05189492, 0.04066492, 0.01101234, 0.00920979, 0.01887209,
           0.05901682, 0.01546502, 0.        , 0.01430969, 0.0165478 ,
           0.01815684, 0.04867158, 0.04389351, 0.06802216, 0.03854181,
           0.03406979, 0.01171209, 0.03426255, 0.02900662])



### Decision Tree model


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
!jupyter nbconvert --to markdown "/content/drive/MyDrive/NLP/HuggingFace/트랜스포머를 활용한 자연어처리/text_classification.ipynb"
```
