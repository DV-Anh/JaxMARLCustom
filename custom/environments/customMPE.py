import math
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as lin
import chex
from typing import Tuple, Dict
from typing_extensions import override
from flax import struct
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State, AGENT_COLOUR, OBS_COLOUR
from jaxmarl.environments.mpe.default_params import DISCRETE_ACT, CONTINUOUS_ACT
from gymnax.environments.spaces import Box, Discrete

bound_low=-1.0
bound_hi=1.0
mission_rew=1
obj_type=['agent','obstacle','target']
def fill_list_repeat(l,n):
    L = l if isinstance(l,list) else [l]
    a=math.ceil(n/len(L))
    ol=L[:n] if a<1 else (L*a)[:n]
    return np.array(ol)

def init_obj_to_array(obj_list): # translate obj in dict to array (pos + vel)
    idx = {'agent':0,'obstacle':1,'target':2} # concatenate in this order
    pos, vel= [[],[],[]], [[],[],[]]
    for i in obj_list:
        index_i = idx[i['obj_type']]
        pos[index_i].append(i['p_pos'])
        vel[index_i].append(i['p_vel'])
    return np.array([*pos[0],*pos[1],*pos[2]],dtype=float), np.array([*vel[0],*vel[1],*vel[2]],dtype=float), {a:[len(pos[idx[a]])] for a in ['agent','obstacle','target']}

@struct.dataclass
class CustomMPEState(State):
    # p_pos: chex.Array  # [num_entities, [x, y]]
    # p_vel: chex.Array  # [n, [x, y]]
    # c: chex.Array  # communication state [num_agents, [dim_c]]
    # done: chex.Array  # bool [num_agents, ]
    # step: int  # current step
    rad: chex.Array = None # object radius
    entity_types: chex.Array = None
    tar_touch: chex.Array = None
    tar_touch_b: chex.Array = None
    obs_touch_b: chex.Array = None
    is_exist: chex.Array = None
    hist_pos: chex.Array = None
    mission_prog: int = 0
    mission_con: chex.Array = None
    collision_count: chex.Array = None
    hist_idx: int = 0
    prev_act: chex.Array = None
    pre_obs: chex.Array = None
    cur_obs: chex.Array = None
    #p_face: chex.Array = None
    map_fog_timer: chex.Array = None
    last_score_timer: int = 0
    valid_tar_p_dist: chex.Array = None
    task_list: chex.Array = None # coord - radius - seen flag
    task_cost_table: chex.Array = None
    task_queues: chex.Array = None
    task_no: int = 0
    task_cost_max: float = 0.0
    tar_resolve_idx: chex.Array = None
    is_task_changed: bool = False
    min_dist_to_furthest_tar: float = 0.0 # currently only relevant during testing
    
    def _to_dict(self,oo=False):
        if oo:
            a=[{'obj_type':obj_type[x],'id':i,'p_pos':self.p_pos[i].tolist(),'p_vel':self.p_vel[i].tolist(),'is_exist':self.is_exist.tolist()[i],'rad':self.rad.tolist()[i]} for i,x in enumerate(self.entity_types)]
            for i in range(len(a)):
                if obj_type[self.entity_types[i]]=='agent':
                    a[i]['mission_con']=int(self.mission_con[i])
                    a[i]['collision_count']=int(self.collision_count[i])
            a={'step':int(self.step),'objects':a,'mission_prog':int(self.mission_prog),'min_dist_to_furthest_tar':float(self.min_dist_to_furthest_tar)}
        else:
            a={'step':int(self.step),'p_pos':self.p_pos.tolist(),'p_vel':self.p_vel.tolist(),'tar_touch':self.tar_touch.tolist(),'is_exist':self.is_exist.tolist(),'rad':self.rad.tolist(),'mission_prog':int(self.mission_prog),'mission_con':self.mission_con.tolist(),'collision_count':self.collision_count.tolist(),'min_dist_to_furthest_tar':float(self.min_dist_to_furthest_tar)}
        return a
    def get_task_indices(self):
        return jnp.where(self.task_list[:,-1]>0)[0]

class CustomMPE(SimpleMPE):
    """
    Custom MPE based on simple_world_comm from jaxmarl.
    Reference: https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/mpe/simple_world_comm/simple_world_comm.py
    Note, currently only have discrete actions implemented.
    """

    def __init__(
        self,
        num_agents=[2],
        num_obs=[1],
        num_tar=[2],
        action_type=DISCRETE_ACT,
        rad_agents=[0.045],
        rad_obs=[0.2],
        rad_tar=[0.1],
        accel_agents=[4.0],
        maxspeed_agents=[1.3],
        tar_amount=[10],
        vision_rad=[0.5],
        expected_step_to_new_tar=jnp.inf,
        dimension=2,
        hist_pos_dur=30,
        dir_per_quad=1,
        damping=[0.3],
        map_fog_res=10, # number of point per axis for fog-of-war resolution
        map_fog_forget_time=100,
        param_perturb=0,
        training=True,
        central_controller=False,
        task_queue_length=2,
        tar_resolve_rad=[0,0.1,0.2],
        action_mode=2,
        reward_separate=False,
        max_steps=100,
        bounds=[[-1,-1],[1,1]],# (2,dimensions),
        move_semicontinuous=False,
        agent_disperse_coef=0.5, # the greater the value, the more agents try to disperse
        tar_update_fn=None,# Callable[[CustomMPE, CustomMPEState, chex.PRNGKey], CustomMPEState]
        init_p=None, # initial positions
        init_v=None, # initial velocities
    ):
        self.is_training=training
        self.is_cc=central_controller
        self.action_mode=action_mode
        self.reward_separate=reward_separate
        self.task_queue_length=task_queue_length
        self.bounds=jnp.array(bounds)
        self.agent_disperse_coef=agent_disperse_coef
        self.is_move_semicontinuous=move_semicontinuous
        self.init_p=init_p
        self.init_v=init_v
        self._updateTar = partial(_updateTarUniform,self) if tar_update_fn is None else partial(tar_update_fn,self)
        self.reward_min = -1 if self.reward_separate else -2
        
        # Fixed parameters
        dim_c = 4  # communication channel dimension

        # Number of entities in each entity class.
        num_agents = np.random.choice(num_agents) if isinstance(num_agents, list) else num_agents
        self.num_obs, self.num_tar = np.random.choice(num_obs) if isinstance(num_obs, list) else num_obs, np.random.choice(num_tar) if isinstance(num_tar, list) else num_tar
        num_landmarks = self.num_obs + self.num_tar
        self.task_queue_length_total=self.task_queue_length*num_agents
        self.tar_resolve_no=len(tar_resolve_rad)
        self.tar_resolve_rad=np.array([[0]+tar_resolve_rad]*num_agents)*np.random.uniform(1-param_perturb,1+param_perturb,(num_agents,self.tar_resolve_no+1))
        self.tar_resolve_rad=jnp.sort(self.tar_resolve_rad)

        # Entity names
        self.agents = ["agent_{}".format(i) for i in range(num_agents)]
        agents = self.agents

        landmarks = (
            ["landmark {}".format(i) for i in range(self.num_obs)]
            + ["tar {}".format(i) for i in range(self.num_tar)]
        )
        self.entity_types=[0]*num_agents+[1]*self.num_obs+[2]*self.num_tar

        # Action and observation spaces
        if action_type == DISCRETE_ACT:
            def get_acts_box(num_dir):
                sv=np.arange(-num_dir,num_dir+1)/num_dir
                def acts_box_recur(dim,arr,m):
                    if dim==dimension-1:
                        return np.vstack([np.append(arr,1.0),np.append(arr,-1.0)]) if m<1.0 else np.vstack([np.append(arr,x) for x in sv])
                    return np.vstack([acts_box_recur(dim+1,np.append(arr,a),max(m,abs(a))) for a in sv])
                acts_mat=np.vstack([np.full((dimension),0.0),acts_box_recur(0,[],0)])
                transform_mat=[]#np.eye(dimension)]
                for i in range(dimension-1):
                    for j in range(i+1,dimension):
                        new_transform_mat90p=np.eye(dimension)
                        new_transform_mat90p[[i,j]]=new_transform_mat90p[[j,i]]
                        new_transform_mat90m=np.copy(new_transform_mat90p)
                        new_transform_mat90p[j]=-new_transform_mat90p[j]
                        new_transform_mat90m[i]=-new_transform_mat90m[i]
                        new_transform_mat180=np.eye(dimension)
                        new_transform_mat180[[i,j]]=-new_transform_mat180[[i,j]]
                        transform_mat.append(new_transform_mat90p)
                        transform_mat.append(new_transform_mat90m)
                        transform_mat.append(new_transform_mat180)
                # transform_mat=np.stack(transform_mat)
                transform_perm=[]
                for tmat in transform_mat:
                    transformed_acts=np.matmul(tmat,acts_mat.T).T
                    perm=[]
                    for v in transformed_acts:
                        for j,u in enumerate(acts_mat):
                            if np.array_equal(u,v):
                                perm.append(j)
                                break
                    transform_perm.append(perm)
                return jnp.array(acts_mat),[lin.block_diag(x,x,x,x) for x in transform_mat],jnp.array(transform_perm),len(transform_mat)
            self.act_vec,self.transform_mat,self.transform_perm,self.transform_no=get_acts_box(dir_per_quad)
            self.act_move_no=len(self.act_vec)
            self.transform_dim=self.transform_mat[0].shape[0]
            if self.action_mode==0:# factorized + concurrent
                self.act_type_idx=[ # index list for action types
                    np.arange(self.act_move_no),
                    np.arange(self.act_move_no,self.act_move_no+self.tar_resolve_no)
                ]
                shared_act_space = Box(-jnp.inf,jnp.inf,(self.act_move_no+self.tar_resolve_no,)) if self.is_move_semicontinuous else Discrete(self.act_move_no+self.tar_resolve_no)
            elif self.action_mode==1:# factorized + sequential
                self.act_type_idx=[np.arange(self.act_move_no+self.tar_resolve_no)]
                shared_act_space = Box(-jnp.inf,jnp.inf,(self.act_move_no+self.tar_resolve_no,)) if self.is_move_semicontinuous else Discrete(self.act_move_no+self.tar_resolve_no)
                self.transform_perm=jnp.concatenate([self.transform_perm,jnp.tile(jnp.arange(self.act_move_no,self.act_move_no+self.tar_resolve_no),(self.transform_perm.shape[0],1))],axis=-1)
            else:# joint (concurrent)
                self.act_type_idx=[np.arange(self.act_move_no*self.tar_resolve_no)]
                shared_act_space = Box(-jnp.inf,jnp.inf,(self.act_move_no*self.tar_resolve_no,)) if self.is_move_semicontinuous else Discrete(self.act_move_no*self.tar_resolve_no)
                self.transform_perm=jnp.repeat(self.transform_perm,self.tar_resolve_no,axis=-1)
            if not self.is_move_semicontinuous:
                shared_act_space.shape=(len(self.act_type_idx),)
            action_spaces = {i: shared_act_space for i in agents}
            self.act_type_idx_lim=jnp.array([len(x)-1 for x in self.act_type_idx])
            self.act_type_idx_offset=jnp.concatenate([jnp.array([0]),jnp.cumsum(self.act_type_idx_lim+1)])[:-1]
        elif action_type == CONTINUOUS_ACT:
            action_spaces = {i: Box(0.0, 1.0, (dimension*2+1,)) for i in agents}
        else:
            raise NotImplementedError("Action type not implemented")
        
        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (dimension*4+self.tar_resolve_no*0+6,)) for i in self.agents}
        self.tar_resolve_onehot=jnp.eye(self.tar_resolve_no)
        colour = (
            [AGENT_COLOUR] * num_agents
            + [OBS_COLOUR] * self.num_obs
            + [(39, 39, 166)] * self.num_tar
        )

        # Parameters
        rad_agent=fill_list_repeat(rad_agents,num_agents)
        rad_obst=fill_list_repeat(rad_obs,self.num_obs)
        rad_targ=fill_list_repeat(rad_tar,self.num_tar)
        accela=fill_list_repeat(accel_agents,num_agents)
        maxspeed=fill_list_repeat(maxspeed_agents,num_agents)
        rad = jnp.concatenate(
            [
                rad_agent*np.random.uniform(1-param_perturb,1+param_perturb,(num_agents)),
                rad_obst*np.random.uniform(1-param_perturb,1+param_perturb,(self.num_obs)),
                rad_targ*np.random.uniform(1-param_perturb,1+param_perturb,(self.num_tar)),
            ]
        )
        #silent = jnp.insert(jnp.ones((num_agents - 1)), 0, 0).astype(jnp.int32)
        collide = jnp.concatenate(
            [
                jnp.full((num_agents + self.num_obs), True),
                jnp.full((self.num_tar), False)
            ]
        )
        accel = jnp.concatenate(
            [
                accela*np.random.uniform(1-param_perturb,1+param_perturb,(num_agents)),
            ]
        )
        max_speed = jnp.concatenate(
            [
                maxspeed*np.random.uniform(1-param_perturb,1+param_perturb,(num_agents)),
                jnp.full((num_landmarks), 0.0),
            ]
        )
        self.vision_rad=jnp.array(fill_list_repeat(vision_rad,num_agents)*np.random.uniform(1-param_perturb,1+param_perturb,(num_agents)))
        self.tar_amounts=fill_list_repeat(tar_amount,self.num_tar)
        self.coinflip=jnp.array([False,True])
        self.coinflip_bias=jnp.array([1-1/self.num_tar/expected_step_to_new_tar,1/self.num_tar/expected_step_to_new_tar])
        self.collision_check_idx=num_agents+num_obs
        self.agents_blind_vec=jnp.full((num_agents-1,dimension),0.0)
        self.landmarks_blind_vec=jnp.full((num_landmarks,dimension),0.0)
        self.hist_pos_dur=hist_pos_dur
        #weights=jnp.arange(hist_pos_dur,0,-1)
        weights=jnp.full((hist_pos_dur),1)
        self.w_norm=jnp.cumsum(weights)
        self.hist_avg_w=jnp.transpose(jnp.tile(weights,[dimension,1]))
        self.map_fog_forget_time=map_fog_forget_time
        grid_size=2.0/map_fog_res
        grid_notches=jnp.arange(-1,1+grid_size,grid_size)
        self.num_map_fog_grid=(map_fog_res+1)**dimension
        self.grid_cen=jnp.array(jnp.meshgrid(*[grid_notches for i in range(dimension)])).T.reshape(-1,dimension)
        self.grid_tar=jnp.array(jnp.meshgrid(*[jnp.arange(-1,1+0.1,0.1) for i in range(dimension)])).T.reshape(-1,dimension)
        self.infos_buffer_keys = [f'reward_{i}' for i in range(len(self.act_type_idx))]
        
        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            action_spaces=action_spaces,
            observation_spaces=observation_spaces,
            dim_c=dim_c,
            dim_p=dimension,
            colour=colour,
            rad=rad,
            #silent=silent,
            collide=collide,
            accel=accel,
            max_speed=max_speed,
            damping=np.random.choice(damping) if isinstance(damping, list) else damping,
            max_steps=max_steps,
        )
    def _to_dict(self):
        a=[{'obj_type':obj_type[x],'id':i,'rad':float(self.rad[i])} for i,x in enumerate(self.entity_types)]
        for i in range(len(a)):
            if obj_type[self.entity_types[i]]=='agent':
                a[i]['vision_rad']=float(self.vision_rad[i])
                a[i]['accel']=float(self.accel[i])
                a[i]['max_speed']=float(self.max_speed[i])
        a={'max_steps':self.max_steps,'objects':a,'damping':float(self.damping)}
        return a
    def reward_invariant_transform_obs(self,obs,axis):
        extra_dims=(None,)*(obs.ndim-1)
        obs_r,obs_i=obs[...,:self.transform_dim,None],obs[...,self.transform_dim:]
        return jnp.concatenate([jnp.concatenate([jnp.matmul(x[extra_dims],obs_r).squeeze(-1),obs_i],axis=-1) for x in self.transform_mat],axis=axis)
    def reward_invariant_transform_acts(self,acts,axis):
        act0,act1=acts[...,0,None],acts[...,1:len(self.act_type_idx)]
        extra_dims=(None,)*(act0.ndim-self.transform_perm[0].ndim)
        oacts=[jnp.concatenate([jnp.take_along_axis(self.transform_perm[idx][extra_dims],act0,axis=-1),act1],axis=-1) for idx in range(self.transform_no)]
        return jnp.concatenate(oacts,axis=axis)
    def get_valid_tar_p_dist(self,pos,rad):
        @partial(jax.vmap, in_axes=(0,0))
        def check_grid_tar_single(p,r):
            return jnp.linalg.norm((p-self.grid_tar),axis=1,ord=2)>r
        clearid=jnp.all(check_grid_tar_single(pos,rad),axis=0)
        return clearid/jnp.sum(clearid)
    @partial(jax.vmap, in_axes=(None,0,0))
    def _rollMat(self,ol,ne):
        row_no=ol.shape[0]
        to_append=jnp.atleast_2d(ne)
        roll_num=to_append.shape[0]
        a=jnp.append(ol[:(row_no-roll_num)],to_append,axis=0)
        return jnp.roll(a,1,axis=0)
    @partial(jax.vmap, in_axes=(None,0,None))
    def get_dist(self,a,b):
        return jnp.linalg.norm((a-b),axis=-1,ord=2)
    def get_task_queue_cost(self,state:CustomMPEState,x):
        a=[[z for z in y if (z<self.num_tar)&(z>=0)] for y in x] # self.num_tar = number of tasks
        # redundant_penalty=0 # check redundant tasks for each agent
        # for i in a:
        #     b=[False]*self.num_tar
        #     for j in i:
        #         redundant_penalty+=b[j]
        #         b[j]=True
        # coverage_penalty=state.task_no-jnp.sum(jnp.array(b)*state.task_list[:,-1])
        # coverage_leeway=jnp.clip(state.task_no-self.task_queue_length_total,0,None)
        # coverage_penalty=jnp.clip(coverage_penalty-coverage_leeway,0.0,None)
        # penalty=redundant_penalty+coverage_penalty
        b=[[(j if i==0 else (y[i-1]+self.num_agents),z) for i,z in enumerate(y)] for j,y in enumerate(a)]
        costs_move=jnp.array([jnp.sum(state.task_cost_table[tuple(zip(*y))])/self.max_speed[j]/self.dt for j,y in enumerate(b)])
        costs_op=jnp.array([jnp.sum(jax.tree_util.tree_map(lambda x:self.tar_amounts-state.tar_touch[x],jnp.array(y,dtype=jnp.int32))) for y in a])
        return jnp.max(costs_move+costs_op)#jax.lax.select(penalty>0,state.task_cost_max*self.task_queue_length_total+penalty,jnp.max(costs_move+costs_op))
    def clean_task_queue(self,state:CustomMPEState,x):
        task_flag=state.task_list[:,-1]>0
        @partial(jax.vmap, in_axes=(0))
        def clean_task_queue_single(a):
            a_flag=jax.tree_util.tree_map(lambda v:(v>=0)&(v<self.num_tar)&task_flag[v],a)
            b=jnp.select([a_flag],[a],-1)
            for i in range(self.task_queue_length-1):
                for j in range(i+1,self.task_queue_length):
                    x,y,z=b[i],b[j],a_flag[i]<a_flag[j]
                    b=b.at[i].set(jax.lax.select(z,y,x))
                    b=b.at[j].set(jax.lax.select(z,x,y))
            # missing_no=a.shape[0]-jnp.sum(a_flag)
            return b
        a=clean_task_queue_single(x)
        return a
    def get_min_dist_to_furthest_tar(self,state: CustomMPEState):
        tar_p=state.p_pos[-self.num_tar:]
        # get distance from agents to tars, dists to non-existent tars set to inf
        agent_to_tar=self.get_dist(jnp.atleast_2d(state.p_pos[:self.num_agents]),tar_p)
        +jax.lax.select(state.is_exist[-self.num_tar:],jnp.zeros((self.num_tar,)),jnp.full((self.num_tar,),-jnp.inf))[None]
        # check which agent is closest to each tar, then take max; not counting non-existent agents
        maxmin_dist=jnp.min(agent_to_tar,axis=-2)
        maxmin_dist=jax.lax.select(state.is_exist[(-self.num_tar):],maxmin_dist,jnp.full(maxmin_dist.shape,-jnp.inf))
        maxmin_dist=jnp.max(maxmin_dist,axis=-1)
        return maxmin_dist
    @partial(jax.vmap, in_axes=(None,0,0))
    def get_tank_vec(self,act,ori):#Only support 2D
        ma=jnp.append(ori[jnp.newaxis,:],(ori[1:0:-1]*jnp.array([-1,1]))[jnp.newaxis,:],axis=0)
        return jnp.matmul(act,ma)
    def get_tank_vec_single(self,act,ori):#Only support 2D
        return self.get_tank_vec(act[jnp.newaxis,:],ori[jnp.newaxis,:])
    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: CustomMPEState, actions: dict):
        u, c, t = self.set_actions(actions)
        t=jax.lax.select(t>=0,t,state.tar_resolve_idx)
        #u=self.get_tank_vec(u,state.p_face)
        if (
            c.shape[1] < self.dim_c
        ):  # This is due to the MPE code carrying around 0s for the communication channels
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1
            )

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        #vel_mag=jnp.linalg.norm(p_vel[:self.num_agents],axis=1,ord=2)
        #vel_z=jnp.transpose(jnp.tile(vel_mag==0,[self.dim_p,1]))
        #p_face=state.p_face*vel_z+p_vel[:self.num_agents]*(~vel_z)
        #vel_mag=jnp.transpose(jnp.tile(jnp.linalg.norm(p_face,axis=1,ord=2),[self.dim_p,1]))
        #p_face=p_face/vel_mag
        
        obs_touch_b=self.get_agent_obs_touch_flag(state)
        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step+1,
            tar_touch_b=self.get_agent_tar_touch_flag(state),# check if target resolve actions are effective on which targets
            obs_touch_b=obs_touch_b,
            collision_count=state.collision_count+jnp.sum(obs_touch_b,axis=1),
            prev_act=u,
            tar_resolve_idx=t,
            #p_face=p_face,
        )
        hist_pos=self._rollMat(state.hist_pos,state.p_pos[:self.num_agents])
        state=state.replace(hist_pos=hist_pos,hist_idx=jnp.min(jnp.array([state.hist_idx+1,self.hist_pos_dur])))
        
        @partial(jax.vmap, in_axes=(0,0,0,None,None))
        def _checkFog(fog_timer,ar,vis_rad,am,cap_timer):
            @partial(jax.vmap, in_axes=(None,0,None))
            def _checkFogSingle(seer,gridc,vis_rad):
                return jnp.linalg.norm(seer-gridc,ord=2)>vis_rad
            fog_unseen=_checkFogSingle(ar,am,vis_rad)
            fog_timer_o=fog_timer*fog_unseen+fog_unseen
            return jnp.clip(fog_timer_o,None,cap_timer)
        if self.is_training: # if training, also simulate non-target destinations via fog of war
            state=state.replace(map_fog_timer=_checkFog(state.map_fog_timer,state.p_pos[:self.num_agents],state.rad[:self.num_agents],self.grid_cen,self.map_fog_forget_time))
        else:
            state=state.replace(map_fog_timer=_checkFog(state.map_fog_timer,state.p_pos[:self.num_agents],self.vision_rad,self.grid_cen,self.map_fog_forget_time))
    
        if self.num_tar>0: # update targets if any
            state = self._updateTar(state,key)
        
        obs,obs_ar,other_blin=self.get_obs(state)
        state=state.replace(pre_obs=state.cur_obs) # update last-step observation
        state=state.replace(cur_obs=obs_ar) # update current observation, must be done again if modified externally
        if self.is_cc:
            task_flag_last=state.task_list[:,-1]>0
            task_flag=task_flag_last&state.is_exist[-self.num_tar:]
            task_flag=jnp.any(~other_blin[:,-self.num_tar:],axis=0)|task_flag
            is_task_changed=(task_flag_last!=task_flag).any()
            tar_p=state.p_pos[-self.num_tar:]
            task_list=jnp.hstack([tar_p,state.rad[-self.num_tar:][...,None],task_flag[...,None]])
            task_cost_table=self.get_dist(jnp.atleast_2d(jnp.vstack([state.p_pos[:self.num_agents],tar_p])),tar_p)
            state=state.replace(task_list=task_list,task_cost_table=task_cost_table,task_no=jnp.sum(task_flag),task_cost_max=task_cost_table.max())
            next_task_queues=self.clean_task_queue(state,state.task_queues)
            state=state.replace(task_queues=next_task_queues,is_task_changed=is_task_changed)
        reward,reward_vec=self.rewards(state)
        if self.is_training:
            info={'collision_count':state.collision_count,'mission_con':state.mission_con}|{f'reward_{j}':reward_vec[:,j] for j in range(len(self.act_type_idx))}
        else:
            info={'targets':obs_ar[...,(self.dim_p):(self.dim_p*2)],'avoid':obs_ar[...,(self.dim_p*2):(self.dim_p*3)]*jnp.abs(obs_ar[...,(self.dim_p*4+1),None])}
            state=state.replace(min_dist_to_furthest_tar=self.get_min_dist_to_furthest_tar(state))
        is_tar_resolved = (~jnp.any(state.is_exist[(-self.num_tar):]))
        done|=is_tar_resolved
        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})
        return obs, state, reward, dones, info
    @override
    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, CustomMPEState]:
        """Initialise with random positions"""
        key_a, key_l = jax.random.split(key)
        p_pos=jax.random.uniform(key_l, (self.num_entities, self.dim_p), minval=self.bounds[0], maxval=self.bounds[1]) if (self.init_p is None) else jnp.array(self.init_p)
        p_vel=jnp.zeros_like(p_pos) if (self.init_v is None) else jnp.array(self.init_v)
        hist_pos=jnp.full((self.num_agents,self.hist_pos_dur,self.dim_p),0.0)
        hist_pos=self._rollMat(hist_pos,p_pos[:self.num_agents])
        state = CustomMPEState(
            entity_types=self.entity_types,
            p_pos=p_pos,
            p_vel=p_vel,
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            tar_touch=jnp.full(self.num_tar,0),
            is_exist=jnp.full(self.num_entities,True),
            mission_prog=0,
            hist_pos=hist_pos,
            hist_idx=0,
            mission_con=jnp.full((self.num_agents),0),
            map_fog_timer=jnp.full((self.num_agents,self.num_map_fog_grid),self.map_fog_forget_time),
            collision_count=jnp.full((self.num_agents),0),
            prev_act=jnp.full((self.num_agents,self.dim_p),0.0),
            last_score_timer=0,
            tar_resolve_idx=jnp.full((self.num_agents),0),
            rad=self.rad,
            # valid_tar_p_dist=valid_tar_p_dist,
            #p_face=jnp.concatenate([jnp.full((self.num_agents,1),1.0),jnp.full((self.num_agents,1),0.0)],axis=1),
        )
        if not self.is_training:
            state=state.replace(min_dist_to_furthest_tar=self.get_min_dist_to_furthest_tar(state))
        obs_touch_b=self.get_agent_obs_touch_flag(state)
        state=state.replace(
            tar_touch_b=self.get_agent_tar_touch_flag(state),
            obs_touch_b=obs_touch_b,
            collision_count=state.collision_count+jnp.sum(obs_touch_b,axis=1),
        )
        obs,obs_ar,other_blin=self.get_obs(state,True)
        state=state.replace(cur_obs=obs_ar,pre_obs=obs_ar)
        if self.is_cc:
            tar_blin=other_blin[:,-self.num_tar:]
            tar_blin=jnp.any(~tar_blin,axis=0)
            tar_p=state.p_pos[-self.num_tar:]
            task_list=jnp.hstack([tar_p,self.rad[-self.num_tar:][...,None],tar_blin[...,None]])
            task_cost_table=self.get_dist(jnp.atleast_2d(jnp.vstack([state.p_pos[:self.num_agents],tar_p])),tar_p)
            state=state.replace(task_list=task_list,task_cost_table=task_cost_table,task_no=jnp.sum(tar_blin),task_cost_max=task_cost_table.max(),task_queues=jnp.full((self.num_agents,self.task_queue_length),-1))
        return obs, state
    def set_actions(self, actions: dict):
        """Extract actions for each agent from their action array."""
        return self.action_decoder(None, actions)

    def _decode_semi_continuous_action(
        self, a_idx: int, actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        @partial(jax.vmap, in_axes=[0])
        def tar_resolve_decoder(a):
            return jnp.zeros((self.dim_p,)),a%self.tar_resolve_no
        @partial(jax.vmap, in_axes=[0, 0])
        def joint_decoder(a_idx, a):
            resolve_w=a.reshape((self.act_move_no,self.tar_resolve_no))
            resolve_w_tar=resolve_w.sum(-2)
            resolve_w_tar-=resolve_w_tar.min()
            resolve_w_sum=resolve_w_tar.sum()
            t=jax.lax.select(resolve_w_sum>0,jnp.round(jnp.arange(self.tar_resolve_no)*resolve_w_tar/resolve_w_sum),-1)
            a=(resolve_w.sum(-1)[...,None]*self.act_vec).sum(-2)*self.accel[a_idx]*self.moveable[a_idx]
            return a,t
        @partial(jax.vmap, in_axes=[0, 0])
        def u_decoder_multi(a_idx, a):
            accel=(self.act_vec*a[self.act_type_idx[0],...,None]).sum(-2)*self.accel[a_idx]*self.moveable[a_idx]
            resolve_w=a[self.act_type_idx[1]]
            resolve_w-=resolve_w.min()
            resolve_w_sum=resolve_w.sum()
            t=jax.lax.select(resolve_w_sum>0,jnp.round(jnp.arange(self.tar_resolve_no)*resolve_w/resolve_w_sum),0)
            return accel,t

        gact = jnp.array([actions[a] for a in self.agents])
        c = jnp.zeros((self.num_agents,self.dim_c))
        if gact.ndim>1:
            if gact.shape[-1]>1:
                if gact.shape[-1]==(self.act_move_no+self.tar_resolve_no):
                    u, t = u_decoder_multi(self.agent_range, gact)
                elif gact.shape[-1]==(self.act_move_no*self.tar_resolve_no):
                    u, t = joint_decoder(self.agent_range, gact)
            else:
                u, t = tar_resolve_decoder(gact.squeeze(-1))
        else:
            u, t = tar_resolve_decoder(gact)
        return u, c, t # movement, communication, target resolve
    def _decode_discrete_action(
        self, a_idx: int, actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        @partial(jax.vmap, in_axes=[0, 0])
        def u_decoder(a_idx, a):
            m, t = a//self.tar_resolve_no, a%self.tar_resolve_no
            return self.act_vec[m] * self.accel[a_idx] * self.moveable[a_idx],t
        @partial(jax.vmap, in_axes=[0, 0])
        def u_decoder_multi(a_idx, a):
            # a=jnp.clip(a,None,self.act_type_idx_lim)
            return self.act_vec[a[0]] * self.accel[a_idx] * self.moveable[a_idx],a[1]

        gact = jnp.array([actions[a] for a in self.agents])
        c = jnp.zeros((self.num_agents,self.dim_c))
        if gact.ndim>1:
            if gact.shape[-1]>1:
                u, t = u_decoder_multi(self.agent_range, gact)
            else:
                u, t = u_decoder(self.agent_range, gact.squeeze(-1))
        else:
            u, t = u_decoder(self.agent_range, gact)
        # u, t = jax.lax.select(gact.ndim>1,u_decoder_multi(self.agent_range, gact),u_decoder(self.agent_range, gact))
        return u, c, t # movement, communication, target resolve

    def _decode_continuous_action(
        self, a_idx: int, actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        @partial(jax.vmap, in_axes=[0, 0])
        def u_decoder(a_idx, action):
            u = jnp.array([action[2] - action[1], action[4] - action[3]])
            return u * self.accel[a_idx] * self.moveable[a_idx]

        gact = jnp.array([actions[a] for a in self.agents])
        c = jnp.zeros((self.num_agents, self.dim_c))

        u_acts = gact
        u = u_decoder(self.agent_range, u_acts)

        return u, c
    @partial(jax.vmap, in_axes=(None,0, None, None))
    def _common_stats(self,aidx: int, state: CustomMPEState, initial_obs=False):
        """Values needed in all observations"""
        num_non_tar=self.num_agents+self.num_obs-1
        other_pos = (
            state.p_pos - state.p_pos[aidx]
        )  # All positions in agent reference frame

        # use jnp.roll to remove ego agent from arrays
        other_pos=jnp.roll(other_pos,shift=self.num_entities-aidx-1,axis=0)[:self.num_entities-1]
        other_vel=jnp.roll(state.p_vel,shift=self.num_entities-aidx-1,axis=0)[:self.num_entities-1]
        other_rad=jnp.roll(self.rad,shift=self.num_entities-aidx-1,axis=0)[:self.num_entities-1]
        other_exist=jnp.roll(state.is_exist,shift=self.num_entities-aidx-1,axis=0)[:self.num_entities-1]
        other_pos=jnp.roll(other_pos,shift=aidx,axis=0)
        other_vel=jnp.roll(other_vel,shift=aidx,axis=0)
        other_rad=jnp.roll(other_rad,shift=aidx,axis=0)
        other_exist=jnp.roll(other_exist,shift=aidx,axis=0)

        other_dist=jnp.linalg.norm(other_pos,axis=1,ord=2)-other_rad
        other_dist_mod=other_dist+jax.lax.select(other_exist,jnp.full((self.num_entities-1),0.0),jnp.full((self.num_entities-1),jnp.inf))
        other_blin=(other_dist_mod>self.vision_rad[aidx])
        tar_dist=other_dist_mod[-self.num_tar:]
        if self.num_entities>1:
            no_other=jnp.min(other_blin[:num_non_tar])
            no_agent=jnp.min(other_blin[:self.num_agents])
            other_avg=jax.lax.select(no_agent,jnp.full((self.dim_p),0.0),jnp.sum(other_pos[:self.num_agents]*(~other_blin[:self.num_agents,None]),axis=0)/jnp.sum(~other_blin[:self.num_agents]))
        else:
            no_other,near_other_p,near_other_vel,other_avg=True,jnp.full((self.dim_p),0.0),jnp.full((self.dim_p),0.0),jnp.full((self.dim_p),0.0)
        tar_min=jnp.argmin(tar_dist)
        no_tar=other_blin[num_non_tar+tar_min]
        
        # past_avg=jnp.sum(state.hist_pos[aidx]*self.hist_avg_w,axis=0)/self.w_norm[state.hist_idx]-state.p_pos[aidx]
        min_timer=jnp.min(state.map_fog_timer[aidx])
        grid_timer_norm=state.map_fog_timer[aidx]-min_timer
        grid_pos=(state.p_pos[aidx]-self.grid_cen)
        grid_norm=jnp.linalg.norm(grid_pos,axis=1,ord=2)
        grid_norm=jax.lax.select(grid_norm>0,grid_norm,jnp.full((self.num_map_fog_grid),1.0))
        grid_timer_norm/=grid_norm
        grid_timer_max=jnp.argmax(grid_timer_norm)
        # grid_timer_norm=jnp.transpose(jnp.tile(grid_timer_norm,[self.dim_p,1]))
        past_avg=grid_pos[grid_timer_max]
        
        other_avg_n=jnp.clip(jnp.linalg.norm(other_avg,ord=2),0.00001,None)
        other_avg_co=jnp.clip(self.vision_rad[aidx]+state.rad[aidx]-other_avg_n,0.0,None)
        tar_from_past=(1-self.agent_disperse_coef*(~no_other))*past_avg+self.agent_disperse_coef*(~no_other)*other_avg/other_avg_n*other_avg_co
        if self.is_cc&(~initial_obs):
            task_exist=(state.task_queues[aidx,0]>=0)#&(state.task_list[state.task_queues[aidx,0],-1]>0)
            near_tar=jax.lax.select(task_exist,state.task_list[state.task_queues[aidx,0],:self.dim_p]-state.p_pos[aidx],jax.lax.select(no_tar,-tar_from_past,other_pos[num_non_tar+tar_min]))
            near_tar_rad=jax.lax.select(task_exist,state.task_list[state.task_queues[aidx,0],self.dim_p],jax.lax.select(no_tar,0.0,other_rad[num_non_tar+tar_min]))
        else:
            near_tar=jax.lax.select(no_tar,-tar_from_past,other_pos[num_non_tar+tar_min])
            near_tar_rad=jax.lax.select(no_tar,0.0,other_rad[num_non_tar+tar_min])
        
        near_tar_n=jnp.linalg.norm(near_tar,ord=2)
        near_tar=jax.lax.select(near_tar_n>0,near_tar/near_tar_n*jnp.clip(near_tar_n-near_tar_rad,0,None),near_tar)
        #tar resolve distance check
        check_dist=jnp.clip(near_tar_n-near_tar_rad,0.0,None)
        real_tar_dist=jax.lax.select(no_tar,self.vision_rad[aidx],check_dist)
        # preferred_tar_resolve_idx=jax.lax.select((~no_tar)&(check_dist<self.tar_resolve_rad[aidx,-1])|jnp.any(state.tar_touch_b[aidx]),jnp.clip(jnp.argmax(check_dist<self.tar_resolve_rad[aidx])-1,0,None),0)
        #keep tar destination within bounds
        near_tar=jax.lax.select(no_tar,jnp.clip(near_tar+state.p_pos[aidx],self.bounds[0],self.bounds[1])-state.p_pos[aidx],near_tar)
        #prepare avoidance
        dummy_obstacle=-near_tar
        other_blin_flag=jnp.tile(other_blin[:num_non_tar],[self.dim_p,1]).T
        near_other_p=jax.lax.select(other_blin_flag,jnp.tile(dummy_obstacle,[num_non_tar,1]),other_pos[:num_non_tar])
        # near_other_vel=jax.lax.select(other_blin_flag,jnp.full((num_non_tar,self.dim_p),0.0),other_vel[:num_non_tar])
        near_other_rad=jax.lax.select(other_blin[:num_non_tar],jnp.full((num_non_tar),0.0),other_rad[:num_non_tar])
        near_tar_n,near_other_p_n=jnp.linalg.norm(near_tar,ord=2),jnp.linalg.norm(near_other_p,ord=2,axis=-1)
        near_other_rev=near_other_p-near_tar
        other_f,other_r=jnp.sum(near_tar*near_other_p,axis=-1),jnp.sum(near_tar*near_other_rev,axis=-1)
        other_col_r=near_other_rad+state.rad[aidx]*(~other_blin[:num_non_tar]) # avoidance distance = obstacle rad + ego rad
        # collision factor, >=1 means collided
        r_other=jax.lax.select((near_other_p_n>0),jnp.clip(other_col_r/near_other_p_n,None,1.0),jax.lax.select((other_col_r==0.0)|no_other,0.0,1.0))
        other_f_n=jax.lax.select(near_tar_n>0,other_f/near_tar_n,jnp.full(other_f.shape,0.0))
        other_f_n=jax.lax.select(near_other_p_n>0,other_f_n/near_other_p_n,jnp.full(other_f.shape,1.0))
        other_f_n=jnp.sqrt(1-r_other**2)-other_f_n
        c_other=jax.lax.select((other_f<=0)|(other_r>=0)|other_blin[:num_non_tar],jnp.full(r_other.shape,0.0),jnp.clip(other_f_n,None,0.0))
        # otv_f=jnp.sum(state.p_vel[aidx]*near_other_p,axis=-1)
        # vel=jnp.linalg.norm(state.p_vel[aidx],ord=2)
        # otv_f_n=jnp.sqrt(1-(otv_f/vel/near_other_p_n)**2)
        # otv_f_n=jnp.clip(otv_f_n/r_other-1,None,0.0)
        # v_other=jax.lax.select((other_r>=0)|other_blin[:num_non_tar],jnp.full(r_other.shape,0.0),jax.lax.select(jnp.isfinite(otv_f_n),otv_f_n,jnp.full(r_other.shape,-1.0)))
        # near_other_p=jax.lax.select(near_other_p_n>0,near_other_p/near_other_p_n*jnp.clip(near_other_p_n-near_other_rad,0,None),near_other_p)
        avoid_idx=jnp.argmin(c_other)
        avoid_idxv=jnp.argmax(r_other)
        c_other,near_other_p_focus,near_other_vel_focus,near_other_rad_focus,r_other_v,r_other_focus=c_other[avoid_idx],near_other_p[avoid_idx],near_other_p[avoid_idxv],near_other_rad[avoid_idx],r_other[avoid_idxv],r_other[avoid_idx]
        near_other_p_focus=jax.lax.select(c_other<0,near_other_p_focus,dummy_obstacle)
        near_other_vel_focus=jax.lax.select(r_other_v>0,near_other_vel_focus,dummy_obstacle)
        near_other_rad_focus=jax.lax.select(c_other<0,near_other_rad_focus,0.0)
        # r_other_focus=r_other_focus*(c_other<0)

        return jnp.concatenate(
                    [
                        #state.p_pos[aidx].flatten(),  # 2
                        (state.p_vel[aidx]).flatten(),  # 2
                        ((near_tar)).flatten(),  # 5, 2
                        ((near_other_p_focus)).flatten(),  # 5, 2
                        ((near_other_vel_focus)).flatten(),  # 5, 2
                        # jnp.array([near_other_rad_focus]),
                        # jnp.array([no_other]),
                        jnp.array([state.rad[aidx]]),
                        jnp.array([c_other]),
                        jnp.array([r_other_v]),
                        jnp.array([r_other_focus]),
                        jnp.array([real_tar_dist]),
                        jnp.array([near_tar_rad]),
                        # self.tar_resolve_onehot[preferred_tar_resolve_idx].flatten(),
                        # jnp.array([self.vision_rad[aidx]]),
                        #((past_avg)).flatten(),
                    ]
                ),other_blin
    def get_obs(self, state: CustomMPEState, initial_obs=False) -> Dict[str, chex.Array]:
        """Returns observations of all agents"""
        obs_ar,other_blin=self._common_stats(self.agent_range,state,initial_obs)
        return {a: obs_ar[i] for i, a in enumerate(self.agents)},obs_ar,other_blin
    @partial(jax.vmap, in_axes=(None,0,0))
    def _bound_rew(self,x,v):
        xa=jnp.abs(x)
        w = xa > 0.9
        m = xa < 1.0
        mr = (xa - 0.9) * 10
        br = jnp.min(jnp.array([jnp.exp(2*xa-2),9000]))

        return jax.lax.select(m, mr, br) * w
    def rewards(self, state: CustomMPEState) -> Dict[str, float]:
        """Computes rewards for all agents"""

        @partial(jax.vmap, in_axes=[0, None])
        def _reward(aidx: int, state: State):
            #return self.agent_reward(aidx, state)
            return self.agent_reward_heuristics(aidx, state)
        r,r_vec=_reward(self.agent_range, state)
        rr={agent: r[i] for i, agent in enumerate(self.agents)}
        return rr,r_vec
    def agent_reward_heuristics(self, aidx: int, state: CustomMPEState):
        return self.agent_reward_heuristics_from_obs(aidx, state, state.cur_obs[aidx])
    def agent_reward_heuristics_from_obs_sparse(self, aidx: int, state: CustomMPEState, ob):
        # sparse reward function
        rew=jax.lax.select(state.tar_touch_b[aidx].any(),1.0,jax.lax.select(state.tar_resolve_idx[aidx]>0,-1.0,0.0))
        ego_rad=ob[self.dim_p*4]
        tar_n=jnp.linalg.norm(ob[(self.dim_p):(self.dim_p*2)],ord=2)
        p_next_dist=ob[self.dim_p*4+4]
        tar_rad=ob[self.dim_p*4+5]
        rc=jax.lax.select(
            ((self.tar_resolve_rad[aidx,-1]>=p_next_dist) # outer radius over inner distance
            &(self.tar_resolve_rad[aidx,1]<=(p_next_dist+2*tar_rad))) # inner radius before outer distance
            |(tar_n<=(ego_rad+self.dt*self.max_speed[aidx])) # approach destination
            ,1.0,-0.5)
        rc=jax.lax.select(jnp.any(state.obs_touch_b[aidx])|(p_next_dist<ego_rad),-1.0,rc)
        total_rew=rc+rew
        return total_rew,(jnp.array([rc,rew]) if self.reward_separate else jnp.array([total_rew,total_rew])) if self.action_mode<1 else jnp.array([total_rew])
    def agent_reward_heuristics_from_obs(self, aidx: int, state: CustomMPEState, ob):
        # dense reward function
        #unit_div=self.vision_rad[aidx]
        # obp=state.pre_obs[aidx]
        ego_vel,ego_rad=ob[:self.dim_p],ob[self.dim_p*4]
        tar_vec=ob[(self.dim_p):(self.dim_p*2)]
        avoid_vec,avoid_vel=ob[(self.dim_p*2):(self.dim_p*3)],ob[(self.dim_p*3):(self.dim_p*4)]
        avoidc,avoidv,avoida=ob[(self.dim_p*4+1)],ob[(self.dim_p*4+2)],ob[(self.dim_p*4+3)]
        real_tar_dist=ob[self.dim_p*4+4]
        rew=jax.lax.select(state.tar_touch_b[aidx].any(),1.0,jax.lax.select(state.tar_resolve_idx[aidx]>0,-1.0,0.0))
        p_next,p_next_dist=tar_vec,jnp.linalg.norm(tar_vec,ord=2)
        co=p_next/p_next_dist
        ego_vel_norm=jnp.linalg.norm(ego_vel,ord=2)
        coef=jnp.sum(co*ego_vel)/self.max_speed[aidx]
        coef=jax.lax.select(jnp.isfinite(coef)&(p_next_dist>ego_rad),coef,0.0)
        rc=jnp.min(jnp.array([coef*(1+avoidc)+avoidc,1]))
        # rc-=jax.lax.select(p_next_dist>ego_rad,0.0,ego_vel_norm/self.max_speed[aidx])
        rc=jax.lax.select(jnp.any(state.obs_touch_b[aidx])|(real_tar_dist<=ego_rad),-1.0,rc)
        total_rew=rc+rew
        return total_rew,(jnp.array([rc,rew]) if self.reward_separate else jnp.array([total_rew,total_rew])) if self.action_mode<1 else jnp.array([total_rew])
    def get_agent_tar_touch_flag(self,state: CustomMPEState):
        @partial(jax.vmap, in_axes=(0,0, None,None,None))
        def get_agent_tar_touch_flag_single(aidx,a,t,r,exist):
            dist=jnp.linalg.norm((a-t),ord=2,axis=1)
            return (dist-r<self.tar_resolve_rad[aidx,state.tar_resolve_idx[aidx]+1])&(dist+r>=self.tar_resolve_rad[aidx,state.tar_resolve_idx[aidx]])&exist
        return get_agent_tar_touch_flag_single(self.agent_range,state.p_pos[:self.num_agents],state.p_pos[(-self.num_tar):],self.rad[(-self.num_tar):],state.is_exist[(-self.num_tar):])
    def get_agent_obs_touch_flag(self,state: CustomMPEState):
        fl=self._collision_batch(
            state.p_pos[:self.num_agents],
            state.rad[:self.num_agents],
            state.p_vel[:self.num_agents],
            state.p_pos[:(self.num_obs+self.num_agents)],
            state.rad[:(self.num_obs+self.num_agents)],
            state.p_vel[:(self.num_obs+self.num_agents)],
            state.is_exist[:(self.num_obs+self.num_agents)],
        )[0]
        for i in range(self.num_agents):
            fl=fl.at[i,i].set(False)
        return fl
    @partial(jax.vmap, in_axes=(None, None, None, None, 0, 0, 0, 0))
    def _collision(self, apos, arad, avel, opos, orad, ovel, fl):
        """Check collision between two entities."""
        deltas = opos - apos
        deltas_next = (opos+(ovel*self.dt))-(apos+(avel*self.dt))
        size = arad + orad
        pen=(~fl)*jnp.inf
        dist = jnp.sqrt(jnp.sum(deltas**2))+pen
        dist_next = jnp.sqrt(jnp.sum(deltas_next**2))+pen
        return fl&(dist<size), fl&(dist_next<size), dist, dist_next, deltas, deltas_next
    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None, None, None))
    def _collision_batch(self, apos, arad, avel, opos, orad, ovel, fl):
        """Check collision in batch."""
        return self._collision(apos, arad, avel, opos, orad, ovel, fl)

def _updateTarUniform(env: CustomMPE, s: CustomMPEState, key: chex.PRNGKey) -> CustomMPEState:
    ff=s.tar_touch+jnp.sum(s.tar_touch_b,axis=0)# target life decreases by number of touching resolve actions
    s0=ff<env.tar_amounts
    new_exhaust=jnp.sum(jnp.logical_xor(s0,s.is_exist[-(env.num_tar):None]))
    pr=s.mission_prog+new_exhaust
    key, key_f, key_p = jax.random.split(key,3)
    recur_flags=jax.random.choice(key_f,a=env.coinflip,shape=(env.num_tar,),replace=True,p=env.coinflip_bias)&(~s0)
    p=jax.random.uniform(key_p, (env.num_tar, env.dim_p), minval=env.bounds[0], maxval=env.bounds[1])
    pp=s.p_pos
    recur_p=jnp.transpose(jnp.tile(recur_flags,[env.dim_p,1]))
    ppp=(pp[-(env.num_tar):None]*(~recur_p))+(p*recur_p)
    pp=pp.at[-(env.num_tar):None].set(ppp)
    ff=ff*(~recur_flags)
    ss=s.is_exist&jnp.concatenate([jnp.full((env.num_obs+env.num_agents),True),s0])
    ss=ss|jnp.concatenate([jnp.full((env.num_obs+env.num_agents),True),recur_flags])
    s = s.replace(
        tar_touch=ff,
        is_exist=ss,
        p_pos=pp,
        last_score_timer=jax.lax.select(pr>s.mission_prog,0,s.last_score_timer+1),
        mission_prog=pr,
        mission_con=s.mission_con+jnp.any(s.tar_touch_b,axis=1),
    )
    return s
    