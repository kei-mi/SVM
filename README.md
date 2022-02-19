# 自作の線形サポートベクター分類（ソフトマージン）
## 有効制約法(Active Set Method)を用いたパラメータの学習

### 線形SVMの簡単な説明

![svm_example](https://user-images.githubusercontent.com/91111835/154810400-403d050e-71d0-4824-b477-81b90a37cb60.png)

データを分割する青い直線をf(x)とします。このときf(x)は以下となります。

<img src="https://latex.codecogs.com/svg.image?f(\mathbf{x})=\sum_{i=1}^{N}&space;\alpha_i&space;y_i&space;\mathbf{x}_i^\mathsf{T}\mathbf{x}&plus;\beta" title="f(\mathbf{x})=\sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i^\mathsf{T}\mathbf{x}+\beta" />

<img src="https://latex.codecogs.com/svg.image?\alpha&space;=&space;\begin{bmatrix}&space;\alpha_1\\&space;\vdots\\&space;\alpha_3\end{bmatrix}" title="\alpha = \begin{bmatrix} \alpha_1\\ \vdots\\ \alpha_3\end{bmatrix}" />
<br>
<br>

データ
<img src="https://latex.codecogs.com/svg.image?(x_i,y_i)" title="(x_i,y_i)" />
を分割する関数fのパラメータ
<img src="https://latex.codecogs.com/svg.image?\alpha,\beta" title="\alpha,\beta" />
は以下の定理を満たす。         


定理の図


この定理を満たす
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
を有効制約法(Active Set Method)を用いて見つける。（
<img src="https://latex.codecogs.com/svg.image?\beta" title="\beta" />
は
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
から作れる）
<br>
<br>
<br>

### 有効制約法(Active Set Method)の簡単な説明

以下のステップを順に行う。

setp1, すべてのデータが集合
<img src="https://latex.codecogs.com/svg.image?\textit{O}" title="\textit{O}" />
に属する状態、かつ
<img src="https://latex.codecogs.com/svg.image?\alpha=0" title="\alpha=0" />
から始める。

step2, 
<img src="https://latex.codecogs.com/svg.image?\textit{I}" title="\textit{I}" />
に属するデータの中で最も
<img src="https://latex.codecogs.com/svg.image?\textit{I}" title="\textit{I}" />
に適していないデータを
<img src="https://latex.codecogs.com/svg.image?\textit{M}" title="\textit{M}" />
に移す、または
<img src="https://latex.codecogs.com/svg.image?\textit{O}" title="\textit{O}" />
に属するデータの中で最も
<img src="https://latex.codecogs.com/svg.image?\textit{O}" title="\textit{O}" />
に適していないデータを
<img src="https://latex.codecogs.com/svg.image?\textit{M}" title="\textit{M}" />
に移す。

step3, 
<img src="https://latex.codecogs.com/svg.image?\textit{M}" title="\textit{M}" />
に属するデータを用いて
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
(仮)を作る。
<img src="https://latex.codecogs.com/svg.image?0\leq&space;\alpha\leq&space;C" title="0\leq \alpha\leq C" />
を満たす範囲で
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
を
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
(仮)に近づける。

step4, 
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
(仮)に近づけた
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
と帳尻が合うように
<img src="https://latex.codecogs.com/svg.image?\textit{M}" title="\textit{M}" />
が属するデータを
<img src="https://latex.codecogs.com/svg.image?\textit{O}" title="\textit{O}" />
または
<img src="https://latex.codecogs.com/svg.image?\textit{I}" title="\textit{I}" />
に移す。

step5, すべてのデータがそれぞれ適した集合
<img src="https://latex.codecogs.com/svg.image?\textit{O}" title="\textit{O}" />
,
<img src="https://latex.codecogs.com/svg.image?\textit{I}" title="\textit{I}" />
,
<img src="https://latex.codecogs.com/svg.image?\textit{M}" title="\textit{M}" />
に属していたら終了（終了時の
<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />
が解）。
そうでなければsetp2へ
<br>
<br>
<br>

### コードに出てくる関数の説明

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

データだけのプロット（超平面はなし）する関数。
<br>
<br>
<br>

- plot_scatter_hyperplane()

データと超平面のプロットする関数。
<br>
<br>
<br>

- optimize_alpha()

step1~step5までを行う関数。

