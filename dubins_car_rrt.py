#documentation link: https://github.com/AndrewWalker/pydubins/blob/master/dubins/dubins.pyx
import dubins
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy

#car settings
turning_radius = 1.0 #set turning radius as 1 for car's dynamics.
step_size = 0.03

#implement 2d continuous environment for dubin's car
class Vertex():
  def __init__(self, x,y, th):
    self.x=x
    self.y=y
    self.th=th
  
  def dubins(self):
    return (self.x, self.y, self.th)
  
  def __repr__(self):
    #convert orientation into angles
    orient_angle=self.th*np.pi/180
    return 'Row: {:.2f}, Col: {:.2f}, Orient: {:.2f}deg'.format(self.row, self.col, orient_angle)

class ContinuousObstacle():
  def __init__(self, x1,y1,x2,y2):
    self.x1=x1
    self.y1=y1
    self.x2=x2
    self.y2=y2
  
  def check_overlap(self, q):
    overlap=False
    min_x=min(self.x1, self.x2)
    max_x=max(self.x1, self.x2)
    min_y=min(self.y1, self.y2)
    max_y=max(self.y1, self.y2)
    if min_x<=q.x and max_x>=q.x and min_y<=q.y and max_y>=q.y:
      overlap=True
    return overlap
  
  def check_intersection(self, q1, q2):
    intersection=False
    min_x=min(self.x1, self.x2)
    max_x=max(self.x1, self.x2)
    min_y=min(self.y1, self.y2)
    max_y=max(self.y1, self.y2)
    #check intersection with dubins shortest path
    path=dubins.shortest_path(q1.dubins(), q2.dubins(), turning_radius)
    traj, times=path.sample_many(0.05)
    for traj_step in traj:
      config=Vertex(traj_step[0], traj_step[1], traj_step[2])
      if self.check_overlap(config):
        intersection=True
        break
    return intersection

class Continuous2DEnv():
  #for RRT
  def __init__(self, h, w, N):
    self.h=h
    self.w=w
    self.obstacles=[]
    self.start=None
    self.goal=None
    self.generate_obstacles(N)
    self.generate_start_and_goal()
  
  def show_grid(self):
    plt.figure(0)
    #draw boundaries
    plt.xlim(-0.5, self.w+0.5)
    plt.ylim(-0.5, self.h+0.5)
    boundary_x=[0,self.w,self.w,0,0]
    boundary_y=[0,0,self.h,self.h,0]
    plt.plot(boundary_x, boundary_y,'k')
    #draw start and goal
    assert self.start!=None and self.goal!=None, "start and goal must be initialized"
    plt.plot(self.start.x, self.start.y, 'ro', label="start")
    plt.plot(self.goal.x, self.goal.y, 'go', label="goal")
    plt.legend()
    for obst in self.obstacles:
      #plot obstacles
      obst_x=[obst.x1, obst.x2, obst.x2, obst.x1, obst.x1]
      obst_y=[obst.y1, obst.y1, obst.y2, obst.y2, obst.y1]
      plt.plot(obst_x, obst_y, 'b')

  def generate_obstacles(self, N):
    for _ in range(N):
      #set of rectangles
      y1=np.random.rand()*self.h
      y2=np.random.rand()*self.h
      x1=np.random.rand()*self.w
      x2=np.random.rand()*self.w
      obstacle=ContinuousObstacle(x1,y1,x2,y2)
      self.obstacles.append(obstacle)
    return
  
  def generate_start_and_goal(self):
    self.start=self.sample_config(p=0)
    self.goal=self.sample_config(p=0)

  def config_is_valid(self, q):
    validity=True
    #check if config is inside grid
    if q.y>=0 and q.y<self.h and q.x>=0 and q.x<self.w:
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
        y=np.random.rand()*self.h
        x=np.random.rand()*self.w
        th=np.random.rand()*2*np.pi-np.pi
        config=Vertex(x,y,th)
        if self.config_is_valid(config):
          break
      return config
  
  def collision_checker(self, q1, q2):
    collision=False
    for obst in self.obstacles:
      if obst.check_intersection(q1,q2):
        collision=True
        break
    return collision
  
  def get_distance(self, q1, q2):
    path=dubins.shortest_path(q1.dubins(), q2.dubins(), turning_radius)
    dist=path.path_length()
    return dist

#implement rrt on dubins car setting.
q0 = (0, 0, -np.pi/2)
q1 = (5, 5, -np.pi/2)

path = dubins.shortest_path(q0, q1, turning_radius)
print(path.path_type())
config=path.sample(0)
print(config)
configurations, times = path.sample_many(step_size)
print(configurations, times)

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

#implement RRT
class RRT():
  def __init__(self, env):
    self.env=env
    self.tree=Tree(self.env.start)
  
  def display_nodes(self):
    self.env.show_grid()
    x=[]
    y=[]
    for node in self.tree.nodes:
      x.append(node.data.x)
      y.append(node.data.y)
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
    #start from q_near => move towards q direction as w.r.t. steering rule
    path=dubins.shortest_path(q_near.dubins(), q.dubins(), turning_radius)
    traj, times=path.sample_many(step_size)
    #reverse times and traj
    times.reverse()
    traj.reverse()
    success=False
    for idx, t in enumerate(times):
      subpath=path.extract_subpath(t)
      subpath_length=subpath.path_length()
      sub_traj=traj[:idx+1]
      final_point=traj[idx]
      q_new=Vertex(final_point[0], final_point[1], final_point[2])
      if subpath_length<eps:
        #check validity of substep
        substep_valid=True
        for sub_step in sub_traj:
          if self.env.config_is_valid(Vertex(sub_step[0], sub_step[1], sub_step[2]))==False:
            substep_valid=False
            break
        if substep_valid==True:
          success=True
          break
    return q_new, q_near_node, success
  
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
    plt.plot(rf_x, rf_y, 'p-', ms=0.1)
    return
  
  def shortcut(self, traj):
    refined_traj=traj
    completion=False
    while completion==False:
      completion=True
      for idx, config in enumerate(traj):
        if (config.y==self.env.start.y and config.x==self.env.start.x) or (config.y==self.env.goal.y and config.x==self.env.goal.x):
          pass
        else:
          prev_config=traj[idx+1]
          next_config=traj[idx-1]
          #a->b->c shortest path doesnt' collide != a->c shortest path doesn't collide
          if self.env.collision_checker(prev_config, next_config)==False:
            refined_traj.remove(config)
            completion=False
    #compute x,y for plotting
    rf_x=[]
    rf_y=[]
    len_traj=len(refined_traj)
    for idx, config in enumerate(refined_traj):
      if idx<len_traj-1:
        next_config=refined_traj[idx+1]
        path=dubins.shortest_path(config.dubins(), next_config.dubins(), turning_radius)
        points, times=path.sample_many(step_size)
        for point in points:
          if self.env.config_is_valid(Vertex(point[0], point[1], point[2]))==False:
            print("fuck")
          rf_x.append(point[0])
          rf_y.append(point[1])
    #check validity of refined path
    return rf_x, rf_y, refined_traj
  
  def solve(self, eps, goal_bias):
    step=0
    while True:
      q=self.env.sample_config(p=goal_bias)
      if q.x==self.env.goal.x and q.y==self.env.goal.y:
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
  c2env=Continuous2DEnv(100,100,5)
  c2env.show_grid()

  rrt=RRT(c2env)
  rrt.solve(eps=3, goal_bias=0.1)

  rrt.display_nodes()
  plt.show()