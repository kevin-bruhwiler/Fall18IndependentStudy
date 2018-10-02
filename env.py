import numpy as np
import cv2, copy, torch, random
from collections import deque
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
_CUDA = torch.cuda.is_available()


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size*2)
        self.fc2 = nn.Linear(input_size*2, input_size*2)
        self.fc3 = nn.Linear(input_size*2, output_size)

    def forward(self, x):
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       return self.fc3(x)


class Ant:
    def __init__(self, size, id):
        self.location = np.array([np.random.randint(0+4,size-4), np.random.randint(0+4,size-4)])
        self.previous_location = np.array([np.random.choice(size), np.random.choice(size)])
        self.id = id
        self.last_eaten = 300
        self.current_move = np.array([0,1])
        self.last_move = copy.copy(self.current_move)
        self.input_size = 5
        self.output_size = 4
        self.score = 0
        self.total_score = 0
        self.collision = False
        self.model = Model(self.input_size, self.output_size)
        self.target_model = Model(self.input_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_func = nn.MSELoss()
        self.iter = 0
        self.mem = deque()
        self.batch_size = 32
        self.discount = 1.0
        self.maxsize = 1000000
        #self.moves = {0:[-1,1],1:[0,1],2:[1,1],3:[1,0],4:[1,-1],5:[0,-1],6:[-1,-1],7:[-1,0]}
        #self.rev_moves = {(-1,1):0,(0,1):1,(1,1):2,(1,0):3,(1,-1):4,(0,-1):5,(-1,-1):6,(-1,0):7}
        self.moves = {0:[0,1],1:[0,-1],2:[1,0],3:[-1,0]}
        self.rev_moves = {(0,1):0,(0,-1):1,(1,0):2,(-1,0):3}
        self.obs = None
        
    def step(self, food, arr):
        self.last_eaten -= 1
        self.obs = self.make_input(food,arr)
        if self.last_eaten == 0:
            return True
        action = self.move(food,arr)
        self.last_move = copy.deepcopy(self.current_move)
        self.current_move = np.array(self.moves[action])
        self.previous_location = copy.copy(self.location)
        self.location = self.location + self.current_move
        np.clip(self.location, 0+4, size-1-4, self.location)
        if food.is_food(self.location):
            food.remove_food(self.location)
            self.last_eaten = 300
            self.score += 10
            self.total_score += 1
        self.train(self.obs, action, food, arr)
        return False
		
    def make_input(self, food, arr):
        inp = [] 
        size = arr.shape[0]
        inp.extend((self.location/size).tolist())
        inp.extend([food.cx/size, food.cy/size, self.last_eaten/300])
        #vision = np.sum(arr[self.location[0]-1:self.location[0]+1,self.location[1]-1:self.location[1]+1], axis=2)
        #inp.extend(vision.flatten().tolist())
        #print(inp)
        return np.asarray(inp)
		
    def move(self,food,arr):
        #print(self.make_input(food,arr))
        #print(self.distance_from_food(food))
        action = np.argmax(self.model.forward(self.to_variable(self.make_input(food,arr))).cpu().data.numpy())
        #print(self.model.forward(self.to_variable(self.make_input(food,arr))).cpu().data.numpy())
        action = self.egreedy(action)
        return action
		
    def egreedy(self, action):
        e = 0.1 + (1 - 0.1) * np.exp(-0.00001 * self.iter)
        self.iter += 1
        if e > np.random.random():
            #if np.random.random() < 0.9:
             #   return self.rev_moves[tuple(self.last_move.tolist())]
            return np.random.randint(0,self.output_size)
        return action

    def add_mem(self, data):
        if len(self.mem) < self.maxsize:
            self.mem.append(data)
        else:
            self.mem.popleft()
            self.mem.append(data)  
        return
        
    def distance_from_food(self, food):
        #print(100/(1+np.sqrt(((self.location - np.array([food.cy, food.cx]))**2).sum(-1))))
        return np.sqrt(((self.location - np.array([food.cy, food.cx]))**2).sum(-1))
        
    def train(self, obs, action, food, arr):
        '''+ 100/(1+self.distance_from_food(food))'''
        #print(self.score)
        self.add_mem((obs,action,self.score+10/(1+self.distance_from_food(food)),self.make_input(food,arr),self.discount))
        self.score = 0
        if len(self.mem) > 1:
            batch_s = min(self.batch_size, len(self.mem))
            batch = random.sample(self.mem, batch_s)
            x_train = np.zeros((batch_s,self.input_size))
            y_train = np.zeros((batch_s,self.output_size))
            state = np.array([s[0] for s in batch])
            next_state = np.array([s[3] for s in batch])
            q = self.model.forward(self.to_variable(state)).cpu().data.numpy()
            q_next = self.target_model.forward(self.to_variable(next_state)).cpu().data.numpy()
            for m in range(0,batch_s):
                obv,action,reward,next_obv,dis = batch[m]
                target = q[m]
                target[action] = reward + (dis * np.amax(q_next[m]))
                x_train[m] = state[m]
                y_train[m] = target
            self.optimizer.zero_grad()
            y = self.to_variable(y_train)
            #print(self.to_variable(y_train))
            out = self.model.forward(self.to_variable(x_train, True)) 
            #print(out.cpu().data.numpy())
            error = self.loss_func(out, y)
            #print(error.cpu().data.numpy())
            error.backward()
            self.optimizer.step()
            if self.iter % 1000 == 0:
                self.target_model = copy.deepcopy(self.model)
            
    
    def to_variable(self, x, grad=False):
        x = Variable(torch.from_numpy(x).float(), requires_grad=grad)
        if _CUDA:
            x.cuda()
        return x
        
    def reset(self):
        z_obv = np.zeros(self.input_size)
        self.add_mem((self.obs,self.rev_moves[tuple(self.last_move.tolist())],0,z_obv,0))
        self.location = np.array([np.random.randint(0+4,size-4), np.random.randint(0+4,size-4)])
        self.last_eaten = 300
        self.current_move = np.array([0,1])
        self.score = 0
        self.total_score = 0
        self.collision = False
        
		
class Food:
    def __init__(self, size, food_size):
        self.x = np.arange(0, size)
        self.y = np.arange(0, size)
        self.cx = np.clip(np.random.choice(size),food_size+4,size-food_size-4)
        self.cy = np.clip(np.random.choice(size),food_size+4,size-food_size-4)
        #self.cx = 40
        #self.cy = 40
        self.radius = food_size
        self.location = np.array([np.random.choice(size), np.random.choice(size)])
        self.circle = (self.x[np.newaxis,:]-self.cx)**2 + (self.y[:,np.newaxis]-self.cy)**2 < self.radius**2
        self.age = 0
        
    def step(self):
        self.age += 1
        if self.age == 200:
            self.age = 0
            self.cx = np.clip(np.random.choice(size),self.radius+4,size-self.radius-4)
            self.cy = np.clip(np.random.choice(size),self.radius+4,size-self.radius-4)
            #self.cx = 40
            #self.cy = 40
            self.radius = food_size
            self.location = np.array([np.random.choice(size), np.random.choice(size)])
            self.circle = (self.x[np.newaxis,:]-self.cx)**2 + (self.y[:,np.newaxis]-self.cy)**2 < self.radius**2

    def is_food(self, ix):
        l1,l2 = ix
        return self.circle[l1,l2]
        
    def remove_food(self, ix):
        l1,l2 = ix
        self.circle[l1,l2] = False
        

class Environment:
	def __init__(self, size, num_ants, food_size):
		self.size = size
		self.num_ants = num_ants
		self.ants = set()
		self.saved_ants = set()
		self.food = Food(size, food_size)
		self.food_size = food_size
		self.arr = np.zeros([self.size]*2 + [3])
		self.round = 1
		self.countdown = 900
	
	def print_highest_score(self, step=None):
		highest_score = 0
		total_score = 0
		ant_id = -1
		for ant in self.saved_ants:
			total_score += ant.total_score
			if ant.total_score >= highest_score:
				highest_score = ant.total_score
				ant_id = ant.id
		if step is None:
			print("ROUND " + str(self.round))
		else:
			print("STEP " + str(step))
		print("Highest scoring ant was " + str(ant_id) + " with score: " + str(highest_score))
		print("Average score was: " + str(total_score/self.num_ants) + "\n")
		
	def step(self):
		for ant in self.ants:
			ant.collision = False
			l1,l2 = ant.location
			p1,p2 = ant.previous_location
			if self.arr[l1,l2,2] == 1 and not (ant.location == ant.previous_location).all():
				self.arr[p1,p2]= np.array([0,0,1]) #red
				ant.location = ant.previous_location
				ant.collision = True
		for ant in self.ants:
			if not ant.collision:
			    l1,l2 = ant.location
			    p1,p2 = ant.previous_location
			    self.arr[p1,p2]= 0
			    self.arr[l1,l2]= 1
		self.arr[self.food.circle] = [1.0,0.5,0.5]
		image = cv2.resize(self.arr, (1000, 1000))
		cv2.imshow('Color image', image)
		dead_ants = []
		for ant in self.ants:
			dead = ant.step(self.food, self.arr)
			if dead:
				dead_ants.append(ant)
		for dead_ant in dead_ants:
			l1,l2 = dead_ant.location  
			self.arr[l1,l2]= 0
			#self.ants.remove(dead_ant) #This permanently removes the ant
			dead_ant.reset() #This tells the ant that it has died, and resets it
		self.arr[self.food.circle] = 0
		self.food.step()
		if len(self.ants) < self.num_ants:
			self.countdown -= 1
		
	def start(self):
		'''use esc to see the results'''
		for i in range(self.num_ants):
			new_ant = Ant(size,i)
			self.ants.add(new_ant)
			self.saved_ants.add(new_ant)
		step = 0
		while True:
			step += 1
			if step % 1000 == 0:
				self.print_highest_score(step=step)
			if len(self.ants) == 0 or self.countdown == 0:
				done = False
				self.print_highest_score()
				self.reset()
				continue
			self.step()
			k = cv2.waitKey(1) & 0xFF
			if k == 27: 
				done = True
				break 
		self.print_highest_score()
		self.reset()
		return done
		cv2.destroyAllWindows()
			
	def reset(self):
		for ant in self.saved_ants:
			ant.reset()
			self.ants.add(ant)
		self.food = Food(self.size, self.food_size)
		self.arr = np.zeros([self.size]*2 + [3])
		self.round += 1
		self.countdown = 900
        
if __name__=="__main__":
	size = 80
	num_ants = 15
	food_size = 8
	env = Environment(size, num_ants, food_size)
	done = False
	while not done:
		done = env.start()
