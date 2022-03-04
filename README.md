# 自作の線形サポートベクター分類（ソフトマージン）
## 逐次最小問題最適化法（SMO）を用いたパラメータの学習

### 制作背景
SVMの計算時間の改善を研究したい。そのためにはSVMの理解が必須であり、SVMのパラメータの学習アルゴリズムを理解せず既存の関数に頼るのはよくない。
なのでSVMのパラメータの学習アルゴリズムを理解するために、そのアルゴリズムの一つである逐次最小問題最適化法（SMO）を自作した。

### 線形SVMの簡単な説明

![svm_example1](https://user-images.githubusercontent.com/91111835/156761372-3121394b-7442-4f1e-bd94-d292ab417491.png)

上図のようなデータを分割する直線をf(x)とします。このときf(x)は以下となる。

<img src="http://latex.codecogs.com/svg.latex?f(\mathbf{x})=\beta^\mathsf{T}\mathbf{x}&plus;\beta_0" />

<img src="http://latex.codecogs.com/svg.latex?\beta=\sum&space;^n_{i=1}\alpha&space;y_i&space;\mathbf{x}_i" title="http://latex.codecogs.com/svg.latex?\beta=\sum ^n_{i=1}\alpha y_i \mathbf{x}_i" />

<img src="https://latex.codecogs.com/svg.image?\alpha&space;=&space;\begin{bmatrix}&space;\alpha_1\\&space;\vdots\\&space;\alpha_n\end{bmatrix}" title="\alpha = \begin{bmatrix} \alpha_1\\ \vdots\\ \alpha_n\end{bmatrix}" />
<br>
<br>

つまり <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " /> がわかればデータを分割する関数 <img src="http://latex.codecogs.com/svg.latex?f(\mathbf{x})" title="http://latex.codecogs.com/svg.latex?f(\mathbf{x})" /> がわかる。

この <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " /> を逐次最小問題最適化法（SMO）を用いて見つける。
<br>
<br>
<br>

### 逐次最小問題最適化法（SMO）の簡単な説明

以下のステップを順に行う。

setp1, <img src="http://latex.codecogs.com/svg.latex?\alpha=0&space;" title="http://latex.codecogs.com/svg.latex?\alpha=0 " /> で初期化する。

step2, 
最適ではない可能性が高い <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " /> のペア <img src="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" title="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" /> を見つける。

step3, 
その <img src="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" title="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" /> に対して制約を無視したときの最適な <img src="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" title="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" /> を求める。

step4, 
制約を満たすように <img src="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" title="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" /> を修正する。

step5,
<img src="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" title="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" /> が改善しなくなるまでstep2へ。そうでない場合、得られた <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " /> から関数f(x)のパラメータ <img src="http://latex.codecogs.com/svg.latex?\beta,&space;\beta_0&space;" title="http://latex.codecogs.com/svg.latex?\beta, \beta_0 " /> を求める。
<br>
<br>
<br>

### コード(SMO_SVM.py)に出てくる関数の説明

- initialize():

データの数や次元、データを分割する関数（超平面）のパラメータなどを初期化する関数。
<br>
<br>
<br>

- gradient_f()

データを分割する関数（超平面）の勾配ベクトルを求める関数。
<br>
<br>
<br>

- inappropriate_index()

step2で用いる、最も適切ではなさそうな <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " /> を見つける関数。
<br>
<br>
<br>

- optimize_alpha()

step3,step4で用いる、<img src="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" title="http://latex.codecogs.com/svg.latex?\alpha_i,\alpha_j" /> を最適化する関数。
<br>
<br>
<br>

- f()

描画に用いる、データを分割する関数（超平面）を求める関数。
<br>
<br>
<br>

- fit()

データを分割する関数（超平面）のパラメータを求める関数。
<br>
<br>
<br>

- predict()

与えられたデータからそのデータが-1と1のどちらに属するかを予測する関数。
<br>
<br>
<br>

- plot_scatter()

データと超平面をプロットする関数。2次元のデータまたは3次元のデータにしか使えない。
<br>
<br>
<br>

### 実行例(用いたデータはにあります。)
1. 2次元のデータ(train_svm1.csv)を用いた場合（データの数:100, 変数の数:2）

import pandas as pd

import numpy as np

import SMO_SVM

<br>

df = pd.read_csv("train_svm1.csv")

X=np.array(df.iloc[:,:-1])

y=np.array(df.iloc[:,-1])

<br>

model = SMO_SVM.SMO_SVM()

model.fit(X,y)

model.plot_scatter()
<br>
<br>

出力

<img src="https://user-images.githubusercontent.com/91111835/154817914-290ce910-f775-4884-b107-35e96b8d50f3.png" width="300px">
<br>
<br>
<br>

2. 自分でデータを用意する場合（データの数:100, 変数の数:2）

import pandas

import numpy

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
<br>

AS = ActiveSet(100, 2, data_set="train_svm.xlsx")

AS.optimize_alpha()
<br>
<br>

出力

<img src="https://user-images.githubusercontent.com/91111835/154819020-9e1d12cd-af16-4749-ba01-02c1757d7b6c.png" width="300px">
