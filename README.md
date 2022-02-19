# 自作の線形サポートベクター分類（ソフトマージン）
**有効制約法を用いたパラメータの学習**

SVMの簡単な説明

![svm_example](https://user-images.githubusercontent.com/91111835/154810400-403d050e-71d0-4824-b477-81b90a37cb60.png)

データを分割する青い直線をf(x)とします。このときf(x)は以下となります。
<img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\sum^N_{i=1}&space;\alpha_i&space;y_i&space;\mathbf{x}_i^\mathsf{T}&space;\mathbf{x}&plus;\beta" title="f(x) = \sum^N_{i=1} \alpha_i y_i \mathbf{x}_i^\mathsf{T} \mathbf{x}+\beta" />
