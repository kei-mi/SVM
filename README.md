# 自作の線形サポートベクター分類（ソフトマージン）
## 逐次最小問題最適化法（SMO）を用いたパラメータの学習

### 制作背景
サポートベクターマシン(以下SVM)の計算時間の改善を研究したい。そのためにはSVMの理解が必須であり、SVMのパラメータの学習アルゴリズムを理解せず既存の関数に頼るのはよくない。
なのでSVMのパラメータの学習アルゴリズムを理解するために、そのアルゴリズムの一つである逐次最小問題最適化法（SMO）を自作した。

### 線形SVMの簡単な説明

![svm_example1](https://user-images.githubusercontent.com/91111835/157197275-575608e5-9f4a-4d54-b60d-935524fcb1bb.png)

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

### 実行例(用いたデータは同じリポジトリにあります。)
1. 2次元のデータ(train_svm1.csv)を用いた場合（データの数:100, 変数の数:2）

入力

import pandas as pd

import numpy as np

import SMO_SVM

<br>

df = pd.read_csv("train_svm1.csv")

X = np.array(df.iloc[:, :-1])

y = np.array(df.iloc[:, -1])

<br>

model = SMO_SVM.SMO_SVM()

model.fit(X, y)

model.plot_scatter()
<br>
<br>

出力

<img src="https://user-images.githubusercontent.com/91111835/157197714-b1abbf20-1e05-4cd5-b669-6ba3573fd487.png" width="300px">
<br>
<br>
<br>

2. 3次元のデータ(train_svm2.csv)を用いた場合（データの数:100, 変数の数:3）

import pandas as pd

import numpy as np

import SMO_SVM

<br>

df = pd.read_csv("train_svm2.csv")

X = np.array(df.iloc[:, :-1])

y = np.array(df.iloc[:, -1])

<br>

model = SMO_SVM.SMO_SVM()

model.fit(X, y)

model.plot_scatter()
<br>
<br>

出力

<img src="https://user-images.githubusercontent.com/91111835/157199120-ced6a82a-6102-4a84-9aa5-4eafd4c67897.png" width="300px">
<img src="https://user-images.githubusercontent.com/91111835/156775003-c4830c4c-4131-470c-805b-7eb424f66955.png" width="300px">
