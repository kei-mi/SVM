import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#逐次最小問題最適化法（SMO）を用いた線形SV分類（ソフトマージン）
class SMO_SVM:
  def __init__(self, C=1.):
    self.C=C          #マージン

  def initialize(self, X, y):
    self.y = y  #応答変数
    self.X = X  #予測変数
    self.N = self.X.shape[0]        #データの数
    self.dim = self.X.shape[1]    #変数の数（次元）
    self.beta = np.zeros(self.dim) #超平面の係数
    self.beta0 = 0.  #超平面の切片
    self.alpha = np.zeros(self.N)  #beta を構成するパラメータ

  #fの勾配ベクトルを計算
  def gradient_f(self):  
     return self.y * (1-np.dot(self.y_x, self.alpha_y_x.T))

  #最適なalphaの可能性が低いインデックスを返す
  def inappropriate_index(self):
    index_1 = self.all_index[((self.alpha<self.C)&(self.y<0)) | ((self.alpha>0)&(self.y>0))]
    index_2 = self.all_index[((self.alpha<self.C)&(self.y>0)) | ((self.alpha>0)&(self.y<0))]
    i = index_1[np.argmin(self.gradient_f()[index_1])]
    j = index_2[np.argmax(self.gradient_f()[index_2])]
    return i, j

  #最適なalpha[i],[j]を計算する
  def optimize_alpha(self, i, j):
    alphay_reduce_ij = self.alpha_y - self.y[i]*self.alpha[i] - self.y[j]*self.alpha[j]
    alphayx_reduce_ij = self.alpha_y_x - self.y[i]*self.alpha[i]*self.X[i,:] - self.y[j]*self.alpha[j]*self.X[j,:]
    #制約を無視した最適なalpha[i]の計算
    alpha_i = ((1-self.y[i]*self.y[j]+self.y[i]*np.dot(self.X[i,:]-self.X[j,:],self.X[j,:]*alphay_reduce_ij-alphayx_reduce_ij)) / ((self.X[i]-self.X[j])**2).sum())
    #制約ありの最適なalpha[i],[j]に近づける
    if alpha_i < 0:
      alpha_i = 0
    elif alpha_i > self.C:
      alpha_i = self.C
    alpha_j = (-alpha_i*self.y[i]-alphay_reduce_ij) * self.y[j]
    if alpha_j < 0:
      alpha_j = 0
      alpha_i = (-alpha_j*self.y[j]-alphay_reduce_ij) * self.y[i]
    elif alpha_j > self.C:
      alpha_j = self.C
      alpha_i = (-alpha_j*self.y[j]-alphay_reduce_ij) * self.y[i]
    return alpha_i, alpha_j

  #超平面（境界）を計算
  def f(self, x):
    return -(self.beta0+np.dot(self.beta[:-1],x)) / self.beta[-1]

  #超平面のパラメータを計算
  def fit(self, X, y):
    #初期化
    self.initialize(X,y)
    self.alpha_y = 0  #alpha と y の内積
    self.alpha_y_x = np.zeros(self.dim)  #alpha と y、x の積
    self.y_x = y.reshape(-1, 1) * X  #y と x の積
    self.all_index = np.arange(X.shape[0])  #indexを表す配列
    #最適ではないalphaが存在する限り繰り返す
    while True:
      #最適なalphaである可能性が低いindexを返す
      i, j = self.inappropriate_index()
      #alpha[i],[j]が最適な場合
      if self.gradient_f()[i] >= self.gradient_f()[j]:
        break
      #alpha[i],[j]を最適化する
      alpha_i, alpha_j = self.optimize_alpha(i, j)
      #新しいalpha[i],[j]に合わせてalphaを用いた式を修正
      self.alpha_y += self.y[i]*(alpha_i-self.alpha[i]) + self.y[j]*(alpha_j-self.alpha[j])
      self.alpha_y_x += self.y[i]*(alpha_i-self.alpha[i])*self.X[i,:] + self.y[j]*(alpha_j-self.alpha[j])*self.X[j,:]
      #alpha[i]が最適な場合
      if alpha_i == self.alpha[i]:
        break
      #alpha[i],[j]をalphaに反映する
      self.alpha[i] = alpha_i
      self.alpha[j] = alpha_j
    #超平面（境界）の係数を計算する
    index_support_vector = self.alpha!=0.
    self.beta = ((self.alpha[index_support_vector]*self.y[index_support_vector]).reshape(-1,1)*self.X[index_support_vector,:]).sum(axis=0)
    self.beta0 = (self.y[index_support_vector]-np.dot(self.X[index_support_vector,:],self.beta)).sum() / index_support_vector.sum()
    
  #与えられたデータから分類する
  def predict(self, X):
    return np.sign(self.beta0 + np.dot(X,self.beta))      

  #データと超平面（境界）のプロット(2次元 or 3次元のデータのみ)
  def plot_scatter(self):
    if self.dim == 2:
      x0_min = min(self.X[:,0])
      x0_max = max(self.X[:,0])
      x1_min = min(self.X[:,1])
      x1_max = max(self.X[:,1])
      plt.scatter(self.X[self.y==1,0], self.X[self.y==1,1], color="red", marker="+")
      plt.scatter(self.X[self.y==-1,0], self.X[self.y==-1,1], color="blue", marker="*")
      plt.plot([x0_min,x0_max], [self.f(x0_min),self.f(x0_max)], color="k")
      plt.scatter(self.X[self.alpha!=0,0], self.X[self.alpha !=0,1], s=200, color=(0,0,0,0), edgecolor="k", marker="o")
      plt.show()
    elif self.dim == 3:
      x0_min = min(self.X[:,0])
      x0_max = max(self.X[:,0])
      x1_min = min(self.X[:,1])
      x1_max = max(self.X[:,1])
      xx0 = np.array([x0_min,x0_min,x0_max,x0_max])
      xx1 = np.array([x1_min,x1_max,x1_min,x1_max])
      xx2 = np.array(self.f(np.r_[[xx0],[xx1]]))
      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      ax.scatter(self.X[self.y==1,0], self.X[self.y==1,1], self.X[self.y==1,2], color="red", marker="+")
      ax.scatter(self.X[self.y==-1,0], self.X[self.y==-1,1], self.X[self.y==-1,2], color="blue", marker="*")
      ax.plot_surface(xx0.reshape(2,2), xx1.reshape(2,2), xx2.reshape(2,2), color="green", alpha=0.3)
      plt.show()
    else:
      print("プロットできるデータは2次元か3次元のみです")