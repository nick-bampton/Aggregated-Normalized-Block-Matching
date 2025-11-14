import numpy as np
import torch as tf
import os
from PIL import Image

def nei_map(w_r,F_s):
  # MAPS IMAGE INTO TENSOR OF STATE OBSERVATIONS
  #
  # Fs    = frame dimensions
  # w_r   = window radius
  # 
  # map_r = map from pixel index (not subscript) to location in state observation tensor

  T = F_s[0]
  N = F_s[1]
  M = F_s[2]

  NM = N*M

  map_v = tf.arange(0,NM)#.to('cuda')
  map_m = tf.reshape(map_v,(N,M))#.to('cuda')

  w_d = 2*w_r+1
  map_s = tf.tensor(())#.to('cuda')
  for j in range(w_d):
    for i in range(w_d):
      map_s = tf.cat((map_s,tf.unsqueeze(map_m[j:N-w_d+j+1,i:M-w_d+i+1],0)),0)

  map_y = map_s.permute(1,2,0).int()
  map_z = map_y.unsqueeze(0).repeat(T,1,1,1)

  ind_r = (tf.arange(T)*NM)#.to('cuda')
  map_a = tf.reshape(ind_r,(T,1,1,1))

  map_r = map_z + map_a
  #print(np.array(map_r))
  return map_r

def edg_con_nbm(F_n,v_r,c1,c2):
  # CONSTRUCTS EDGE OBSERVATIONS FOR NORMALIZED BLOCK MATCHING
  #
  # F_n   = tensor of state observations (vector), dim 1-2 is vertex, dim 3 is state observation       
  # v_r   = velocity radius (v_1 in paper)
  # c_1   = NBM parameter 1 (b in paper)
  # c_2   = NBM parameter 2 (epsilon in paper)   
  # 
  # D_n   = tensor of edge observations (real number), dim 1-2 is source vertex, dim 3 is destination vertex            

  Fs = F_n.size()
  N = Fs[1]
  M = Fs[2]
  T = Fs[0]
  V = (2*v_r+1)**2

  D = tf.zeros(T-1,N,M,V)#.to('cuda')

  v_v = tf.arange(-v_r,v_r+1)#.to('cuda')
  v_m1, v_m2 = tf.meshgrid(v_v, v_v, indexing='ij')
  v_v1 = tf.flatten(v_m1)
  v_v2 = tf.flatten(v_m2)

  for k in range(V):
    j = v_v1[k].int()
    i = v_v2[k].int()
    B1 = F_n[0:T-1,  max(0,-j):min(N,N-j),max(0,-i):min(M,M-i),:]
    B2 = F_n[1:T  ,  max(0, j):min(N,N+j),max(0,+i):min(M,M+i),:]

    d1 = tf.exp(-c1*tf.sum((B1-B2)**2,dim=3))#.to('cuda') #tf.sum((B1-B2)**2,dim=3)#
    d2 = tf.ones(T-1,N,M)#.to('cuda')
    d2[:,max(0,-j):min(N,N-j),max(0,-i):min(M,M-i)] = d1
    #print(j,i)
    D[:,:,:,k] = d2#tf.unsqueeze(d2,3)
  
  D_n = D/tf.add(tf.unsqueeze(tf.sum(D,dim=3),3),c2)
  
  return D_n

def edg_nor(D):
  # EDGE OBSERVATION BACKGROUND SUBTRACTION / NORMALIZING
  #
  # D     = edge observation tensor
  #
  # D_r   = normalized edge observation tensor 

  eps = 0.00001
  div = tf.std(D,dim=0)
  div[div<eps] =  eps

  D_r = (D-tf.mean(D,dim=0)[None,:,:,:])/(div)[None,:,:,:]
  return D_r

def edg_bou(D,b):
  # EDGE OBSERVATION UPPER BOUND
  #
  # D     = edge obersvation tensor
  # b     = upper bound
  #
  # D_b   = bounded edge observation tensor

  bt = tf.tensor(b)
  if tf.numel(bt)==1:
    D_b = tf.minimum(D, bt)
  else:
    D_b = D
  return D_b

def edg_map(D_s,v_r):
  # MAP EDGE OBSERVATIONS TO BE BASED ON DESTINATION VERTEX NOT SOURCE VERTEX
  #
  # D_s   = edge observation tensor
  # v_r   = velocity radius (v_1 in paper)
  #
  # hat   = map from edge observation sorted by source vertex to sorted by destination

  N = D_s[1]
  M = D_s[2]
  v_n = D_s[3]#(2*v_r+1)**2

  indv = tf.arange(N*M)#.to('cuda')
  indm = v_n*tf.unsqueeze(tf.reshape(indv,(N,M)),2) + v_r*(v_r+1)*2
  hat = tf.tile(indm,(1,1,v_n))

  indv2 = tf.arange(N*M*v_n)#.to('cuda')
  cat = tf.reshape(indv2,(N,M,v_n))

  sad = 0 
  mad = v_n-1
  for j in range(-v_r,v_r+1):
    for i in range(-v_r,v_r+1):
      hat[max(0,-j):min(N,N-j),max(0,-i):min(M,M-i),sad] = cat[max(0, j):min(N,N+j),max(0, i):min(M,M+i),mad]
      sad = sad+1
      mad = mad-1

  return hat

def lon_pat(H_i,D_n,map_e):
  # COMPUTES LONGEST PATH (AND AVERAGE)
  #
  # H_i   = initial values (P(x_0) in paper but could be other)
  # D_n   = edge observation tensor
  # map_e = edge observation tensor map
  #
  # H_a   = LPA where H_a[k,:,:] is the LPA-k starting at t=0


  H_t = H_i
  D_s = D_n.size()
  H_a = tf.zeros(D_s[0],D_s[1],D_s[2])#.to('cuda')
  for t in range(D_s[0]):
    S_t = D_n[t] + tf.unsqueeze(H_t,2)
    S_v = tf.flatten(S_t)
    R_t = S_v[map_e]
    H_t = tf.max(R_t,dim=2).values
    H_a[t,:,:] = H_t/(t+1) #keep history of LPAs for when we have t<k

  return H_a

def exe_lpa(F,k,c1,c2,b):   #fb,N,T,c,sig_n,bet,pk,bk
  # COMPUTE LPA 
  #
  # F     = tensor of frames
  # k     = LPA parameter
  # c_1   = NBM parameter 1 (b in paper)
  # c_2   = NBM parameter 2 (epsilon in paper)  
  # b     = edge observation upper bound
  #
  # H_t   = LPA-t

  w_r = 1 #window radius
  v_r = 2 #velocity radius (bound)


  F_s = F.size()

  map_p = nei_map(w_r,F_s)
  F_v = tf.flatten(F)
  F_n = F_v[map_p]

  D_n = edg_con_nbm(F_n,v_r,c1,c2)

  D_r = edg_nor(D_n)

  D_b = edg_bou(D_r,b)
  
  D_s = D_n.size()

  map_e = edg_map(D_s,v_r)
  #print('setup')
  H_i = tf.zeros(F_s[1:3])[2:,2:]#.to('cuda') #-2 because window is 3x3 so no obs for outer pixels

  H_t = tf.zeros([F_s[0]-1,F_s[1]-2,F_s[2]-2])
  for t in range(F_s[0]-k-1):
    H_t_a = lon_pat(H_i,D_b[t:t+k+1,:,:,:],map_e)
    H_t[t+k,:,:] = H_t_a[-2,:,:] #im too lazy to change the tensor size in lon_pat, needs to be t-1 cuz edges# are vert#-1
    if t==0:
      H_t[0:k+1,:,:] = H_t_a
    #print(t+k)

  return H_t#,H_t_a

def load_data(path,T,T0):
  # LOAD DATA FROM FOLDER
  #
  # path  = path to folder
  # T     = length of data sequence ('end' goes till last file)
  # T0    = start of data sequence (0 is start file)   

  dir_list = os.listdir(path)
  dir_sort = sorted(dir_list,key=len)

  if T == 'end':
    T = np.shape(dir_sort)[0]-T0

  Xs = np.array(Image.open(os.path.join(path, dir_sort[T0])))
  sX = np.shape(Xs)

  check = Xs.ndim>2

  X1 = np.zeros([T,sX[0],sX[1]])
  for k in range(T):
    Xin = np.array(Image.open(os.path.join(path, dir_sort[k+T0])))
    if check:
      X1[k] = Xin[:,:,0]
    else:
      X1[k] = Xin

  return tf.from_numpy(X1)#,W1

if __name__ == '__main__':

  data_path = "PUT YOUR PATH TO DATA FOLDER HERE" #loads data from images in folder
  k = 50
  c1 = 1/90000
  c2 = 0.04
  b = 3  

  Fi = load_data(data_path,'end',0)
  Fo = exe_lpa(Fi,k,c1,c2,b)

