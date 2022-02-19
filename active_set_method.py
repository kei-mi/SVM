#Active Set Method（有効制約法）を用いた線形SV分類（ソフトマージン）
class ActiveSet():
  #最初に必要なimport
  #import pandas
  #import numpy
  #import matplotlib.pyplot as plt
  #from sklearn.preprocessing import scale

  def __init__(self, N, dim, data_set = None):
    self.N = N        #データの数
    self.dim = dim    #変数の数（次元）
    
    if self.dim!=2 and data_set is None:
      print("データセットを与えていないため、2次元のデータセットを使います")
      self.dim = 2
    elif data_set is None:
      print("2次元のデータセットを使います")

    #データを与えない場合は自動でデータを作る
    if data_set is None:
      numpy.random.seed(4)
      self.X = numpy.random.randn(N, self.dim)
      self.Y = numpy.array([1 if y+x > 0 else - 1 for x, y in self.X])
      self.X[self.Y == 1, 1] -= 0.3
      self.X[self.Y == -1, 1] += 0.3
    else:
      print("与えられたデータセットを使います")
      df = pandas.read_excel(data_set)
      self.N=len(df)
      self.dim=len(df.columns)-1
      for i in range(len(df.columns)-1):
        df.iloc[:,i] = scale(df.iloc[:,i])
      self.X = numpy.array(df.iloc[:, :-1])
      self.Y = numpy.array(df.iloc[:, -1])

    self.C = 1.0 #マージン
    #初期値を与える
    self.alpha = numpy.zeros(N) 
    self.list_I = [] #alpha == C となる点のリスト
    self.list_M = [] #0 < alpha < 1 となる点のリスト
    self.list_O = [i for i in range(N)] #alpha == 0 となる点のリスト
    self.I = set(self.list_I)
    self.M = set(self.list_M)
    self.O = set(self.list_O)
    self.beta = 0 #超平面の切片
  
    self.calc_num = 0  #ループの回数
    self.calc_limit = pow(10, 3) #ループの上限

  #データと超平面との距離
  def calc_hyperplane(self, list_type):
    dis = self.beta
    for i in range(self.N):
      dis += self.alpha[i] * self.Y[i] * self.X[i].dot(self.X[list_type].T)
    return dis

  #超平面を生成
  def make_hyperplane(self, x):
    res = self.beta
    tmp = 0
    for i in range(self.N):
      tmp += self.alpha[i] * self.Y[i] * self.X[i][1]
    for i in range(self.N):
      res += self.alpha[i] * self.Y[i] * self.X[i][0]
    res = -res/tmp *x
    return res

  #誤った集合に属するデータの数
  def inappropriate_data(self, list_type):
    if list_type == self.list_O:
      #list_Oに誤って属するデータの数
      return len(numpy.argwhere(self.Y[list_type] * self.calc_hyperplane(list_type) < 1))
    elif list_type == self.list_I:
      #list_Iに誤って属するデータの数
      return len(numpy.argwhere(self.Y[list_type] * self.calc_hyperplane(list_type) > 1))
    else:
      print("wrong list")
      exit()

  #データを別の集合に移す
  def transfer_data(self, from_set, from_list, to_set, move_index):
    to_set.add(from_list[move_index])
    from_set.remove(from_list[move_index])

  #線形方程式を解く
  def solve(self):
    Y_I = self.Y[self.list_I]
    Y_M = self.Y[self.list_M]
    X_I = self.X[self.list_I]
    X_M = self.X[self.list_M]
    Q_M = (Y_M.reshape(-1, 1) * X_M).dot(X_M.T * Y_M)
    A = numpy.vstack((numpy.hstack((Q_M, Y_M.reshape(-1, 1))), numpy.hstack((Y_M, 0))))
    if len(self.list_I) != 0:
      Q_MI = (Y_M.reshape(-1, 1) * X_M).dot(X_I.T * Y_I)
      B = -self.C * numpy.vstack((Q_MI.sum(axis = 1).reshape(-1, 1), Y_I.sum()))
      B[:-1] += 1
    else:
      B=[[1] for _ in range(len(self.list_M))]
      B.append([0])
    #res = [alpha, beta]
    res = numpy.linalg.pinv(A).dot(B)
    return res

  #ステップ幅を求める
  def calc_eta(self, d, alpha_old):
    eta = [numpy.inf, None]
    for i in range(len(self.list_M)):
      if d[i] == 0:
        continue
      elif d[i] < 0:
        tmp = alpha_old[i] / (-d[i])
        if eta[0] > tmp:
          eta = [tmp, i]
      else:
        tmp = (self.C-alpha_old[i]) / d[i]
        if eta[0] > tmp:
          eta = [tmp, i]
    return eta

  #データのプロット(2次元)
  def plot_scatter(self):
    for i in range(self.N):
      if self.Y[i] == -1:
        plt.scatter(self.X[i][0], self.X[i][1], color='red')
      else:
        plt.scatter(self.X[i][0], self.X[i][1], color='blue')
    plt.show()

  #データと超平面のプロット(2次元)
  def plot_scatter_hyperplane(self):
    #データのプロット
    plt.xlim(self.X[:,0].min()-1, self.X[:,0].max()+1)
    plt.ylim(self.X[:,1].min()-1, self.X[:,1].max()+1)
    for i in range(self.N):
      if self.Y[i] == -1:
        plt.scatter(self.X[i][0], self.X[i][1], color='red')
      else:
        plt.scatter(self.X[i][0], self.X[i][1], color='blue')
    
    #超平面のプロット
    p = numpy.linspace(self.X[:,0].min()-1, self.X[:,0].max()+1, 70)
    q = self.make_hyperplane(p)
    plt.plot(p, q)
    #plt.savefig("svm_example.png")
    plt.show()
  
  #超平面のパラメータalphaを求める
  def optimize_alpha(self):
    
    #データが誤った集合に属する限り計算する
    while self.inappropriate_data(self.list_O)>0 or self.inappropriate_data(self.list_I)>0:
      
      #最も誤った集合に属するデータを別の集合に移す
      isnot_move = True
      if len(self.list_I) == 0:
        index_O = numpy.argmin(self.Y[self.list_O]*self.calc_hyperplane(self.list_O))
        self.transfer_data(self.O, self.list_O, self.M, index_O)
        isnot_move = False
      elif len(self.list_O) == 0:
        index_I = numpy.argmax(self.Y[self.list_I]*self.calc_hyperplane(self.list_I))
        self.transfer_data(self.I, self.list_I, self.M, index_I)
        isnot_move = False
      if isnot_move:
        index_O = numpy.argmin(self.Y[self.list_O]*self.calc_hyperplane(self.list_O))
        index_I = numpy.argmax(self.Y[self.list_I]*self.calc_hyperplane(self.list_I))
        distance_O = abs(1-self.Y[self.list_O[index_O]]*self.calc_hyperplane(self.list_O[index_O]))
        distance_I = abs(1-self.Y[self.list_I[index_I]]*self.calc_hyperplane(self.list_I[index_I]))
        if distance_O >= distance_I:
          self.transfer_data(self.O, self.list_O, self.M, index_O)
        else:
          self.transfer_data(self.I, self.list_I, self.M, index_I)
      is_transfer = True
      self.list_M = list(self.M)
      
      #超平面の変更により生まれた誤った集合に属するデータが,存在する限り計算する
      while is_transfer and len(self.list_M)!=0:
        is_transfer = False
        self.list_I = list(self.I)
        self.list_M = list(self.M)
        self.list_O = list(self.O)

        #超平面のパラメータ変更
        res = self.solve()
        alpha_new = res[:-1,0]
        direction = alpha_new - self.alpha[self.list_M]
        if numpy.all(direction == 0):
          break
        if numpy.all(0 <= alpha_new) and numpy.all(alpha_new <= self.C):
          self.alpha[self.list_M] = alpha_new
        else:
          eta = self.calc_eta(direction, self.alpha[self.list_M])
          self.alpha[self.list_M] += eta[0] * direction

          #最も誤った集合に属するデータを別の集合に移す
          if direction[eta[1]] < 0:
            self.transfer_data(self.M, self.list_M, self.O, eta[1])
            is_transfer = True
          elif direction[eta[1]] > 0:
            self.transfer_data(self.M, self.list_M, self.I, eta[1])
            is_transfer = True
          self.list_I = list(self.I)
          self.list_M = list(self.M)
          self.list_O = list(self.O)
    
      self.beta = res[-1,0]

      #ループ回数が多い場合は終了
      self.calc_num += 1
      if self.calc_num > self.calc_limit:
        break 

    #結果のプロット
    if self.dim == 2:
      self.plot_scatter_hyperplane()

    return self.alpha, self.beta 