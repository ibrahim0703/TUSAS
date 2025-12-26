import math
import random

class AircraftEnv:
"""2D ENVIRONMENT FOR AN AIRCRAFT TARGET SIMULATION"""
  def __init__(self,width = 10,height =10):
    self.width = width
    self.height = height
    self.target_pos = self._get_random_pos()
    self.agent_pos = self._get_random_pos()
    while self.agent_pos == self.target_pos:
      self.agent_pos = self._get_random_pos() # to ensure agent does not spawn exactly same coordinate with the target
      
  def get_random_pos(self):
    return [random.randint(0,self.width-1),random.randint(0,self.height-1)]
    
  def calculate_distance(self):
    dx = self.agent_pos[0] - self.target_post[0]
    dy = self.agent_pos[1] - self.target_post[1]
    distance = math.sqrt(dx**2 + dy**2)
    return distance

  def get_state(self):
    """return state vector which is [Agent_X, Agent_Y, Target_X, Target_Y, distance]"""
    distance = self.calculate_distance()
    return {"agent_pos":self.agent_pos,"target_pos":self.target_pos,"distance to the target": round(distance,2)}

  def step(self,action):
    """ executes an action and updates the environment. Action: up, down,left, right"""
    x,y = self.agent_pos
    #boundary checks to keep the agent inside the grid.
    if action =="up" and y < self.height-1:
      y +=1
    elif action == " down" and y > 0:
      y-=1
    elif action == "left" and x > 0:
      x-=1
    elif action == "right" and x < self.width -1:
      x +=1
    self.agent_pos = [x,y]
    return self.get_state()

  ##TEST##
#initialize environment
env = AircraftEnv(width = 20, height = 20)
print(f"Initial state:{env.get_state()}")
# simulate a move
action = "right"
print(f"\n-- Action taken:
{action.upper()}---")
next_state = env.step(action)
print(f"New State: {next_state}")
# check distance condition ın range r radius circle.
dist = next_state["distance_to_target"]
if dist   <1:
  print("\n[INFO] Target Destroyed!")
else: 
    print(f"\n[INFO] Target is still {dist} units away.")

  
    
