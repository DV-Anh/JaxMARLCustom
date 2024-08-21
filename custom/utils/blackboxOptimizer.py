# import jax
# import jax.numpy as jnp
import numpy as np
import math
import itertools

class PermutationSolver(object):
    def __init__(self,fitness=lambda x:0,opt_dir=1,sol_shape=(1,1),value_pool=[1]):
        self.fitness,self.opt_dir,self.sol_shape,self.value_pool=fitness,opt_dir,sol_shape[:2],value_pool
        self.real_sol_length=sol_shape[0]*sol_shape[1]
        self.sol_length=max(len(value_pool),self.real_sol_length)
        self.value_pool+=[-1]*(self.sol_length-len(value_pool))
        poisson_mean=2
        self.poisson_p=np.array([(poisson_mean**k)*np.exp(-poisson_mean)/math.factorial(k) for k in range(self.sol_length-1)])
        self.poisson_p/=np.sum(self.poisson_p)
    def prep_sol(self,x):
        return np.reshape(x[:self.real_sol_length],self.sol_shape)
    def eval(self,x):
        return self.fitness(self.prep_sol(x))
    def random_solution(self):
        return np.random.permutation(self.value_pool)
    def scramble(self,x,k=2):
        idx=np.random.choice(self.sol_length,size=k,replace=False)
        idx_shuffle=np.copy(idx)
        np.random.shuffle(idx_shuffle)
        y=np.copy(x)
        for i,v in enumerate(idx):
            y[v]=x[idx_shuffle[i]]
        return y
    def fill_partial_seq(self,x,max_length):
        y,idx=[],0
        for i in range(max_length):
            if i not in x:
                y.append(i)
            else:
                y.append(x[idx])
                idx+=1
        return y
    def solve_ls(self,pop_size=1,neighborhood_size=2): # hill climbing
        neighborhood_size=min(neighborhood_size,self.sol_length)
        all_neighbors=np.array([self.fill_partial_seq(v,self.sol_length) for v in itertools.permutations(range(self.sol_length),neighborhood_size)])
        all_neighbors=np.unique(all_neighbors,axis=0)[1:]
        pop=[self.random_solution() for i in range(pop_size)]
        fit=np.array([self.eval(i) for i in pop])
        for i in range(pop_size):
            improved=True
            x,fx=pop[i],fit[i]
            while improved:
                improved=False
                np.random.shuffle(all_neighbors)
                for j in all_neighbors:
                    y=np.copy(x)
                    for k,v in enumerate(j):
                        y[k]=x[v]
                    if ~np.array_equal(x,y):
                        fy=self.eval(y)
                        if (fy-fx)*self.opt_dir>0:
                            x,fx,improved=y,fy,True
            pop[i],fit[i]=x,fx
        sel_idx=np.argmax(fit*self.opt_dir)
        return self.prep_sol(pop[sel_idx])
    def solve_rls(self,pop_size=1,budget=1000): # random local search
        scramble_strength=np.random.choice(np.arange(2,self.sol_length+1),size=budget-pop_size,replace=True,p=self.poisson_p)
        parent_idx=np.random.randint(0,pop_size,size=budget-pop_size)
        pop=[self.random_solution() for i in range(pop_size)]
        fit=np.array([self.eval(i) for i in pop])
        add_last,worst_idx=True,-1
        for i in range(budget-pop_size):
            parent_sel=parent_idx[i]
            y=self.scramble(pop[parent_sel],scramble_strength[i])
            while np.array_equal(pop[parent_sel],y):
                y=self.scramble(pop[parent_sel],scramble_strength[i])
            fy=self.eval(y)
            worst_idx=np.argmin(fit*self.opt_dir) if add_last else worst_idx
            add_last=(fy-fit[worst_idx])*self.opt_dir>=0
            if add_last:
                pop[worst_idx]=y
                fit[worst_idx]=fy
        sel_idx=np.argmax(fit*self.opt_dir)
        return self.prep_sol(pop[sel_idx])