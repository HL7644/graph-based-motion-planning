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

class ContinuousObstacle():
  def __init__(self, r1,c1,r2,c2):
    self.r1=r1
    self.c1=c1
    self.r2=r2
    self.c2=c2
    self.segments=[]
  
  def check_overlap(self, q):
    overlap=False
    min_r=min(self.r1, self.r2)
    max_r=max(self.r1, self.r2)
    min_c=min(self.c1, self.c2)
    max_c=max(self.c1, self.c2)
    if min_r<=q.row and max_r>=q.row and min_c<=q.col and max_c>=q.col:
      overlap=True
    return overlap
  
  def check_intersection(self, q1, q2):
    intersection=False
    #check intersection with a line: r=ac+b
    slope=(q2.row-q1.row)/(q2.col-q1.col)
    offset=q1.row-slope*q1.col
    min_qc=min(q1.col, q2.col)
    max_qc=max(q1.col, q2.col)
    for col in np.arange(min_qc, max_qc+0.01, 0.01):
      row=slope*col+offset
      q=Vertex(row, col)
      if self.check_overlap(q):
        intersection=True
        break
    return intersection

class Continuous2DEnv():
  #for RRT and FMT
  def __init__(self, h, w, N):
    self.h=h
    self.w=w
    self.obstacles=[]
    self.start=None
    self.goal=None
    self.generate_obstacles(N)
    self.generate_start_and_goal()
  
  def show_grid(self):
    plt.figure()
    #draw boundaries
    plt.xlim(-0.5, self.w+0.5)
    plt.ylim(-self.h-0.5, 0.5)
    boundary_x=[0,self.w,self.w,0,0]
    boundary_y=[0,0,-self.h,-self.h,0]
    plt.plot(boundary_x, boundary_y,'k')
    #draw start and goal
    assert self.start!=None and self.goal!=None, "start and goal must be initialized"
    plt.plot(self.start.col, -self.start.row, 'ro', label="start")
    plt.plot(self.goal.col, -self.goal.row, 'go', label="goal")
    plt.legend()
    for obst in self.obstacles:
      #plot obstacles
      obst_x=[obst.c1, obst.c2, obst.c2, obst.c1, obst.c1]
      obst_y=[-obst.r1, -obst.r1, -obst.r2, -obst.r2, -obst.r1]
      plt.plot(obst_x, obst_y, 'b')

  def generate_obstacles(self, N):
    for _ in range(N):
      #set of rectangles
      r1=np.random.rand()*self.h
      r2=np.random.rand()*self.h
      c1=np.random.rand()*self.w
      c2=np.random.rand()*self.w
      obstacle=ContinuousObstacle(r1,c1,r2,c2)
      self.obstacles.append(obstacle)
    return
  
  def generate_start_and_goal(self):
    self.start=self.sample_config(p=0)
    self.goal=self.sample_config(p=0)

  def config_is_valid(self, q):
    validity=True
    #check if config is inside grid
    if q.row>=0 and q.row<self.h and q.col>=0 and q.col<self.w:
      #check ovelap w/ obstacles
      for obst in self.obstacles:
        if obst.check_overlap(q):
          validity=False
          break
    else:
      validity=False
    return validity

  def sample_config(self, p):
    prob=np.random.rand()
    if prob<p and self.goal!=None:
      #return goal
      return self.goal
    else:
      #random sampling
      while True:
        row=np.random.rand()*self.h
        col=np.random.rand()*self.w
        vertex=Vertex(row, col)
        if self.config_is_valid(vertex):
          break
      return vertex
  
  def collision_checker(self, q1, q2):
    collision=False
    for obst in self.obstacles:
      if obst.check_intersection(q1,q2):
        collision=True
        break
    return collision
  
  def get_distance(self, q1, q2):
    dist=np.sqrt((q1.row-q2.row)**2+(q1.col-q2.col)**2)
    return dist

  def check_goal(self, q):
    dist=self.get_distance(q, self.goal)
    if dist<self.goal_thresh:
      return True
    else:
      return False

class TreeNode():
  def __init__(self, data):
    self.data=data
    self.parent=None

class Tree():
  def __init__(self, head_data):
    self.head=TreeNode(head_data)
    self.nodes=[self.head]
  
  def add_item(self, data, parent_node):
    assert parent_node in self.nodes, "invalid parent node"
    node=TreeNode(data)
    node.parent=parent_node
    self.nodes.append(node)
    return

class RRT():
  def __init__(self, env):
    self.env=env
    self.tree=Tree(self.env.start)
  
  def display_nodes(self):
    self.env.show_grid()
    x=[]
    y=[]
    for node in self.tree.nodes:
      x.append(node.data.col)
      y.append(-node.data.row)
      plt.plot(x,y,'yo',ms=1)
    return

  def get_q_near(self, q):
    #minimum distance from tree nodes
    q_near_node=self.tree.nodes[0]
    min_dist=self.env.get_distance(q, q_near_node.data)
    for idx in range(1, len(self.tree.nodes)):
      node=self.tree.nodes[idx]
      config=node.data
      dist=self.env.get_distance(q, config)
      if dist<min_dist:
        min_dist=dist
        q_near_node=node
    return q_near_node, min_dist
  
  def get_q_new(self, q, eps):
    #eps: max length of movement towards q.
    q_near_node, min_dist=self.get_q_near(q)
    q_near=q_near_node.data
    #get line r=ac+b
    #start from q_near => move towards q direction as a straight line
    theta=np.arctan2((q.row-q_near.row), (q.col-q_near.col))
    flag=False
    for d in np.arange(eps, 0, -0.01):
      new_row=q_near.row+d*np.sin(theta)
      new_col=q_near.col+d*np.cos(theta)
      q_new=Vertex(new_row, new_col)
      if self.env.config_is_valid(q_new) and self.env.collision_checker(q_new, q_near)==False:
        flag=True
        break
    return q_new, q_near_node, flag
  
  def get_path(self, q_near_node, q_goal):
    #reconstruct path
    q_node=q_near_node
    traj=[q_goal, q_near_node.data]
    while True:
      if q_node.parent==None:
        traj.append(q_node.data)
        break
      q_node=q_node.parent
      q=q_node.data
      traj.append(q)
    #apply shortcut algorithm
    rf_x, rf_y, refined_traj=self.shortcut(traj)
    #plot path
    self.env.show_grid()
    plt.plot(rf_x, rf_y, 'p-', ms=1)
    return
  
  def shortcut(self, traj):
    refined_traj=traj
    flag=False
    while flag==False:
      flag=True
      for idx, config in enumerate(traj):
        if (config.row==self.env.start.row and config.col==self.env.start.col) or (config.row==self.env.goal.row and config.col==self.env.goal.col):
          pass
        else:
          prev_config=traj[idx+1]
          next_config=traj[idx-1]
          if self.env.collision_checker(prev_config, next_config)==False:
            refined_traj.remove(config)
            flag=False
    #compute x,y for plotting
    rf_x=[]
    rf_y=[]
    for config in refined_traj:
      rf_x.append(config.col)
      rf_y.append(-config.row)
    return rf_x, rf_y, refined_traj
  
  def solve(self, eps, goal_bias):
    step=0
    while True:
      q=self.env.sample_config(p=goal_bias)
      if q.row==self.env.goal.row and q.col==self.env.goal.col:
        #check terminal condition
        q_near_node,_=self.get_q_near(q)
        q_near=q_near_node.data
        if self.env.collision_checker(q_near, q)==False:
          #q near and q_g can be connected w/o obstruction
          path=self.get_path(q_near_node, q)
          break
      else:
        q_new, q_near_node, flag=self.get_q_new(q, eps)
        if flag:
          self.tree.add_item(q_new, q_near_node)
        step+=1

if __name__=='__main__':
    #RRT Algorithm using continuous 2d env.
    c2env=Continuous2DEnv(100,150,5)
    c2env.show_grid()
    rrt=RRT(c2env)
    rrt.solve(eps=1, goal_bias=0.1)
    rrt.display_nodes()
    plt.show()