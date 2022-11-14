import torch
import numpy as np
import matplotlib.pyplot as plt

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vertex():
  def __init__(self, row, col):
    self.row=row
    self.col=col
    self.parent=None
  
  def __repr__(self):
    return 'Row: {:.2f}, Col: {:.2f}'.format(self.row, self.col)

class Env2D():
  def __init__(self, h, w, N):
    self.h=h
    self.w=w
    self.grid=torch.zeros(h,w).to(device)
    self.start=None
    self.goal=None
    self.generate_obstacles(N) #0: free space, 1: obstacles
    self.get_start_and_goal()
    self.vertices=self.get_vertices()

  def generate_obstacles(self, N):
    for _ in range(N):
      obs_r1=np.random.randint(low=0, high=self.h)
      obs_r2=np.random.randint(low=obs_r1, high=self.h)
      obs_c1=np.random.randint(low=0, high=self.w)
      obs_c2=np.random.randint(low=obs_c1, high=self.w)
      #mark obstacle grids with 1
      self.grid[obs_r1:obs_r2+1, obs_c1:obs_c2+1]=1
    return
  
  def get_vertices(self):
    vertices=[]
    for row in range(self.h):
      for col in range(self.w):
        if self.grid[row,col]==0:
          vertices.append(Vertex(row,col))
    return vertices

  def config_is_valid(self, row, col):
    validity=False
    if row>=0 and row<self.h and col>=0 and col<self.w:
      if self.grid[row, col]!=1:
        validity=True
    return validity

  def get_start_and_goal(self):
    #generate random start, goal locations
    while True:
      sr=np.random.randint(low=0, high=self.h)
      sc=np.random.randint(low=0, high=self.w)
      if self.grid[sr,sc]==0:
        break
    while True:
      gr=np.random.randint(low=0, high=self.h)
      gc=np.random.randint(low=0, high=self.w)
      if self.grid[gr,gc]==0:
        break
    #mark start w/ 2, goal w/ 3
    self.start=Vertex(sr,sc)
    self.grid[sr,sc]=2
    self.goal=Vertex(gr,gc)
    self.grid[gr,gc]=3
    return
  
  def get_distance(self, q1, q2):
    #taxicab distance
    r1=min(q1.row, q2.row)
    c1=min(q1.col, q2.col)
    r2=max(q1.row, q2.row)
    c2=max(q1.col, q2.col)
    distance=(r2-r1)+(c2-c1)
    return distance
  
  def collision_checker(self, q1, q2):
    #q1, q2: vertex class
    r1=min(q1.row, q2.row)
    c1=min(q1.col, q2.col)
    r2=max(q1.row, q2.row)
    c2=max(q1.col, q2.col)
    #consider taxicab distance in grids
    collide=False
    free_lower=True
    free_upper=True
    #upper path
    for row in range(r1,r2+1):
      if self.grid[row,c1]==1:
        free_upper=False
        break
    if free_upper:
      for col in range(c1, c2+1):
        if self.grid[r2, col]==1:
          free_upper=False
          break
    #lower path
    for row in range(r1,r2+1):
      if self.grid[row,c2]==1:
        free_lower=False
        break
    if free_lower:
      for col in range(c1, c2+1):
        if self.grid[r1, col]==1:
          free_lower=False
          break
    if free_upper==False and free_lower==False:
      collide=True
    return collide, free_upper, free_lower

class A_star():
  #A_star algorithm for 2d grid environment.
  def __init__(self, env, max_radius):
    self.env=env #contains start and goal.
    self.max_radius=max_radius
    self.Q=[] #list of vertex element

    self.coa=torch.full([self.env.h, self.env.w], 1e8).to(device)
    #initialize coa.
    self.coa[self.env.start.row, self.env.start.col]=0
    self.Q.append(self.env.start)
    #total cost of initial to goal passing certain point
    self.total_cost=torch.full([self.env.h, self.env.w], 1e8).to(device)
    self.total_cost[self.env.start.row, self.env.start.col]=self.get_ctg(self.env.start)

  
  def is_free(self, q):
    if self.env.grid[q.row, q.col]==0:
      return True
    else:
      return False

  def get_ctg(self, q):
    #direct distance regardless of obstacles.
    ctg=self.env.get_distance(self.env.goal, q)
    return ctg

  def get_neighbors(self, q, max_radius):
    #q: vertex
    neighbors=[]
    for radius in range(1, max_radius+1):
      dist_pairs=[]
      for r in range(0, radius+1):
        row_offset=r
        col_offset=radius-r
        candidates=[[q.row-row_offset, q.col-col_offset],[q.row-row_offset, q.col+col_offset], [q.row+row_offset, q.col-col_offset], [q.row+row_offset, q.col+col_offset]]
        for cand in candidates:
          if self.env.config_is_valid(cand[0], cand[1]):
            collide, _, _=self.env.collision_checker(Vertex(cand[0], cand[1]), q)
            if collide==False:
              neighbors.append(Vertex(cand[0], cand[1]))
    return neighbors
  
  def sample_config(self):
    #sample argmin total_cost
    sample_vector=torch.zeros(len(self.Q)).to(device)
    for idx, vertex in enumerate(self.Q):
      sample_vector[idx]=self.total_cost[vertex.row, vertex.col]
    argmin_idx=torch.argmin(sample_vector, dim=0)
    return argmin_idx, self.Q[argmin_idx]
  
  def get_path(self, goal):
    #grid indicating trajectories
    traj=[]
    q=goal
    result_grid=self.env.grid.detach().clone()
    while q!=None:
      result_grid[q.row, q.col]=9
      traj.insert(0, q.parent)
      q=q.parent
    return traj, result_grid

  def solve(self):
    print("Start and Goal")
    print(self.env.start)
    print(self.env.goal)
    print("---------------------")
    buffer=[self.env.start]
    step=0
    while len(self.Q)!=0:
      #sample config.
      argmin_idx, q_min=self.sample_config()
      if q_min.row==self.env.goal.row and q_min.col==self.env.goal.col:
        print("goal reached")
        return self.get_path(q_min)
      else:
        self.Q.remove(q_min)
        neighbors=self.get_neighbors(q_min, self.max_radius)
        flag=0
        for q_n in neighbors:
          #predicting distance of q_n
          coa_pred_neigh=self.coa[q_min.row, q_min.col]+self.env.get_distance(q_min, q_n)
          coa_neigh=self.coa[q_n.row, q_n.col]
          if coa_pred_neigh<coa_neigh:
            flag+=1
            #q_n becomes a parent of q_min
            self.coa[q_n.row, q_n.col]=coa_pred_neigh
            h_q_n=self.get_ctg(q_n) #approximated ctg: direct distance from q_n to q_g
            self.total_cost[q_n.row, q_n.col]=coa_pred_neigh+h_q_n
            if q_n not in self.Q:
              q_n.parent=q_min
              self.Q.append(q_n)
        step+=1
    print("Steps: {:d}".format(step))
    return "failure"

if __name__=='__main__':
    #A star algorithm using 2d grid environment.
    env=Env2D(10,10,5)
    a_star=A_star(env, 2)
    path=a_star.solve()
    print(env.grid)
    if path!='failure':
        print(path[1])
    else:
        print("invalid configuration")