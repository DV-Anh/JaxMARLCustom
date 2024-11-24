import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
from typing import Optional
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class MPEVisualizer(object):
    def __init__(
        self,
        env,
        state_seq: list,
        reward_seq=None,
        act_seq=None,
        info_seq=None,
        frame_to_run=None,
        title_text=None,
    ):
        self.env = env

        self.interval = 100
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.act_seq = act_seq
        self.info_seq = info_seq
        self.frame_to_run=frame_to_run
        self.title_text=title_text
        
        self.comm_active = not np.all(self.env.silent)
        print('Comm active? ', self.comm_active)
        
        self.init_render()

    def animate(
        self,
        save_fname: Optional[str] = None,
        view: bool = True,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)

        if view:
            plt.show(block=True)

    def init_render(self):
        from matplotlib.patches import Circle, Rectangle, Arrow, Annulus
        state = self.state_seq[0]
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        
        ax_lim = 2
        self.ax.set_xlim([-ax_lim, ax_lim])
        self.ax.set_ylim([-ax_lim, ax_lim])
        
        self.entity_artists = []
        for i in range(self.env.num_entities):
            c = Circle(
                state.p_pos[i], state.rad[i], color=np.array(self.env.colour[i]) / 255, edgecolor=None, alpha=0.3
            )
            self.ax.add_patch(c)
            self.entity_artists.append(c)
        self.ax.add_patch(Rectangle((-1,-1),2,2,edgecolor=[0,0,0],fill=None))
        self.step_counter = self.ax.text(-1.95, 1.95, f"Step: {state.step}", va="top")
        
        if self.comm_active:
            self.comm_idx = np.where(self.env.silent == 0)[0]
            print('comm idx', self.comm_idx)
            self.comm_artists = []
            i = 0
            for idx in self.comm_idx:
                
                letter = ALPHABET[np.argmax(state.c[idx])]
                a = self.ax.text(-1.95, -1.95 + i*0.17, f"{self.env.agents[idx]} sends {letter}")
                
                self.comm_artists.append(a)
                i += 1
        self.dectection_artists=[]
        if hasattr(self.env,'vision_rad'):
            for i in range(self.env.num_agents):
                c=Circle(state.p_pos[i],self.env.vision_rad[i],edgecolor=[0,0,0],fill=None,alpha=0.3,linestyle='--')
                self.ax.add_patch(c)
                self.dectection_artists.append(c)
        self.act_artists=[]
        self.acte_artists=[]
        if self.act_seq is not None:
            for i in range(self.env.num_agents):
                c=Arrow(state.p_pos[i,0],state.p_pos[i,1],self.act_seq[0][i][0]*self.env.dt,self.act_seq[0][i][1]*self.env.dt,width=0.2,alpha=0.3,color=[0,0,0])
                self.ax.add_patch(c)
                self.act_artists.append(c)
            if self.act_seq[0][0].shape[0]>self.env.dim_p+1:
                for i in range(self.env.num_agents):
                    c=Annulus((state.p_pos[i,0],state.p_pos[i,1]),self.act_seq[0][i][self.env.dim_p+1],self.act_seq[0][i][self.env.dim_p+1]-self.act_seq[0][i][self.env.dim_p],alpha=0.3,color=[0,1,0])
                    self.ax.add_patch(c)
                    self.acte_artists.append(c)
        self.rew_artists=[]
        if self.reward_seq is not None:
            for i in range(self.env.num_agents):
                c=self.ax.text(state.p_pos[i,0],state.p_pos[i,1],"{:.3f}".format(self.reward_seq[0][i]),alpha=0.3)
                self.rew_artists.append(c)
        self.tar_artists=[]
        self.av_artists=[]
        if self.info_seq is not None:
            if 'targets' in self.info_seq[0].keys():
                for i in range(self.env.num_agents):
                    c=Arrow(state.p_pos[i,0],state.p_pos[i,1],self.info_seq[0]['targets'][i,0],self.info_seq[0]['targets'][i,1],width=0.2,alpha=0.3,color=[0,0,1])
                    self.ax.add_patch(c)
                    self.tar_artists.append(c)
            if 'avoid' in self.info_seq[0].keys():
                for i in range(self.env.num_agents):
                    c=Arrow(state.p_pos[i,0],state.p_pos[i,1],self.info_seq[0]['avoid'][i,0],self.info_seq[0]['avoid'][i,1],width=0.2,alpha=0.3,color=[1,0,0])
                    self.ax.add_patch(c)
                    self.av_artists.append(c)
                
    def update(self, frame):
        state = self.state_seq[frame]
        frame_back,frame_cap=max(frame-1,0),min(frame,len(self.act_seq)-1)
        for i, c in enumerate(self.entity_artists):
            if hasattr(state,'is_exist'):
                if state.is_exist[i]:
                    c.set_radius(state.rad[i])
                else:
                    c.set_radius(0)
            
            c.center = state.p_pos[i]
        if hasattr(self.env,'vision_rad'):
            for i in range(self.env.num_agents):
                self.dectection_artists[i].center=state.p_pos[i]
        if self.act_seq is not None:
            for i in range(self.env.num_agents):
                x,y=state.p_pos[i,0],state.p_pos[i,1]
                dx,dy=self.act_seq[frame_cap][i][0]*self.env.dt,self.act_seq[frame_cap][i][1]*self.env.dt
                self.act_artists[i]._patch_transform=(Affine2D().scale(np.hypot(dx,dy),0.2).rotate(np.arctan2(dy,dx)).translate(x,y).frozen())
        if len(self.acte_artists)>=self.env.num_agents:
            for i in range(self.env.num_agents):
                self.acte_artists[i].set_center((state.p_pos[i,0],state.p_pos[i,1]))
                self.acte_artists[i].set_radii(self.act_seq[frame_cap][i][self.env.dim_p+1])
                self.acte_artists[i].set_width(self.act_seq[frame_cap][i][self.env.dim_p+1]-self.act_seq[frame_cap][i][self.env.dim_p])
        if self.reward_seq is not None:
            for i in range(self.env.num_agents):
                self.rew_artists[i].set_text("{:.3f}".format(self.reward_seq[frame_back][i]))
                self.rew_artists[i].set_position(state.p_pos[i])
        if len(self.tar_artists)>=self.env.num_agents:
            for i in range(self.env.num_agents):
                x,y=state.p_pos[i,0],state.p_pos[i,1]
                dx,dy=self.info_seq[frame_back]['targets'][i,0],self.info_seq[frame_back]['targets'][i,1]
                self.tar_artists[i]._patch_transform=(Affine2D().scale(np.hypot(dx,dy),0.2).rotate(np.arctan2(dy,dx)).translate(x,y).frozen())
        if len(self.av_artists)>=self.env.num_agents:
            for i in range(self.env.num_agents):
                x,y=state.p_pos[i,0],state.p_pos[i,1]
                dx,dy=self.info_seq[frame_back]['avoid'][i,0],self.info_seq[frame_back]['avoid'][i,1]
                self.av_artists[i]._patch_transform=(Affine2D().scale(np.hypot(dx,dy),0.2).rotate(np.arctan2(dy,dx)).translate(x,y).frozen())
        title_t=self.title_text if self.title_text else ''
        if self.frame_to_run is not None:
            title_t+=f' - Run {self.frame_to_run[frame]}'
        self.ax.title.set_text(title_t)
        self.step_counter.set_text(f"Step: {state.step}")
        
        if self.comm_active:
            for i, a in enumerate(self.comm_artists):
                idx = self.comm_idx[i]
                letter = ALPHABET[np.argmax(state.c[idx])]
                a.set_text(f"{self.env.agents[idx]} sends {letter}")
        