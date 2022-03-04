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

### コード(active_set_method.py)に出てくる関数の説明

前提：あらかじめnumpyとmatplotlib.pyplot、pandas、scaleのインポートが必要
<br>

- calc_hyperplane():

<img src="https://latex.codecogs.com/svg.image?f(\mathbf{x})=\sum_{i=1}^{N}&space;\alpha_i&space;y_i&space;\mathbf{x}_i^\mathsf{T}\mathbf{x}&plus;\beta" title="f(\mathbf{x})=\sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i^\mathsf{T}\mathbf{x}+\beta" />

に　<img src="https://latex.codecogs.com/svg.image?\mathbf{x}=\rm{list\_type}" title="\mathbf{x}=\rm{list\_type}" />　を代入したときの値を返す関数。
<br>
<br>
<br>

- make_hyperplane()

描画用の
<img src="https://latex.codecogs.com/svg.image?f(\mathbf{x})" title="f(\mathbf{x})" />
を返す関数。
<br>
<br>
<br>

- inappropriate_data()

step2で用いる、適していない集合に属するデータの個数を返す関数。
<br>
<br>
<br>

- transfer_data()

step2,step4で用いる、データを別の集合に移す関数。
<br>
<br>
<br>

- solve()

<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />(仮)を作る関数。より具体的には

<img src="https://latex.codecogs.com/svg.image?\mathbf{Q}_M=\begin{bmatrix}\mathbf{Q}_M_{i,j}\end{bmatrix}" title="\mathbf{Q}_M=\begin{bmatrix}\mathbf{Q}_M_{i,j}\end{bmatrix}" />

<img src="https://latex.codecogs.com/svg.image?\mathbf{Q}_M_{i,j}=y_i&space;y_j&space;\mathbf{x}_i^\mathsf{T}&space;\mathbf{x}_j" title="\mathbf{Q}_M_{i,j}=y_i y_j \mathbf{x}_i^\mathsf{T} \mathbf{x}_j" />　　<img src="https://latex.codecogs.com/svg.image?i,j&space;\in&space;\textit{M}" title="i,j \in \textit{M}" />

<img src="https://latex.codecogs.com/svg.image?\mathbf{Q}_{MIi,j}=y_iy_j\mathbf{x}_i^\mathsf{T}\mathbf{x}_j" title="\mathbf{Q}_{MIi,j}=y_iy_j\mathbf{x}_i^\mathsf{T}\mathbf{x}_j" />　　<img src="https://latex.codecogs.com/svg.image?i\in&space;\textit{M},j\in&space;\textit{I}" title="i\in \textit{M},j\in \textit{I}" />

と定義する。このとき以下の式

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;\mathbf{Q}_M&y_M&space;&space;\\&space;y_M^\mathsf{T}&0&space;&space;\\\end{bmatrix}\begin{bmatrix}&space;\alpha_M\\&space;\beta\end{bmatrix}&space;=-C\begin{bmatrix}\mathbf{Q}_{MI}&space;\mathbf{1}&space;\\\textbf{1}^\mathsf{T}y_I\end{bmatrix}&space;&plus;\begin{bmatrix}\textbf{1}&space;\\0\end{bmatrix}&space;" title="\begin{bmatrix} \mathbf{Q}_M&y_M \\ y_M^\mathsf{T}&0 \\\end{bmatrix}\begin{bmatrix} \alpha_M\\ \beta\end{bmatrix} =-C\begin{bmatrix}\mathbf{Q}_{MI} \mathbf{1} \\\textbf{1}^\mathsf{T}y_I\end{bmatrix} +\begin{bmatrix}\textbf{1} \\0\end{bmatrix} " />

を
<img src="https://latex.codecogs.com/svg.image?\alpha,\beta" title="\alpha,\beta" />
について解いている。
<br>
<br>
<br>

- calc_eta()

step3の
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
を
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />(仮)
に近づける度合いを返す関数。
<br>
<br>
<br>

- plot_scatter()

データだけ（超平面はなし）をプロットする関数。
<br>
<br>
<br>

- plot_scatter_hyperplane()

データと超平面をプロットする関数。
<br>
<br>
<br>

- optimize_alpha()

step1~step5までを行う関数。
<br>
<br>
<br>

### 実行例
1. クラス内で用意されたデータを用いる場合（データの数:50, 変数の数:2）

import pandas

import numpy

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
<br>

AS = ActiveSet(50, 2)

AS.optimize_alpha()
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
