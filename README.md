# 自作の線形サポートベクター分類（ソフトマージン）
**有効制約法を用いたパラメータの学習**

SVMの簡単な説明

![svm_example](https://user-images.githubusercontent.com/91111835/154810400-403d050e-71d0-4824-b477-81b90a37cb60.png)

データを分割する青い直線をf(x)とします。このときf(x)は以下となります。

<img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\sum^N_{i=1}&space;\alpha_i&space;y_i&space;\mathbf{x}_i^\mathsf{T}&space;\mathbf{x}&plus;\beta" title="f(x) = \sum^N_{i=1} \alpha_i y_i \mathbf{x}_i^\mathsf{T} \mathbf{x}+\beta" />

<img src="https://latex.codecogs.com/svg.image?\alpha&space;=&space;\begin{bmatrix}&space;\alpha_1\\&space;\vdots\\&space;\alpha_3\end{bmatrix}" title="\alpha = \begin{bmatrix} \alpha_1\\ \vdots\\ \alpha_3\end{bmatrix}" />

データ
<img src="https://latex.codecogs.com/svg.image?(x_i,y_i)" title="(x_i,y_i)" />
を分割する関数fのパラメータ
<img src="https://latex.codecogs.com/svg.image?\alpha,\beta" title="\alpha,\beta" />
は以下の定理を満たす。

