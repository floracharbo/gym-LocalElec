#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:32:24 2020

@author: floracharbonnier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:47:57 2020

@author: floracharbonnier

"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy 
import datetime
from scipy.stats import gamma
from userdeftools import initialiseentries
import itertools
from collections import defaultdict

class LocalElecEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_agents = 1):
        # Define action and observation space
        self.done                          = 0
        self.n_agents                      = n_agents
        self.t                             = 0
        self.indicators, self.discrete, self.granularity = {}, {}, {}
        self.indicators['action']          = ['cons_flex_share','prod_to_sell_share','store_to_sell_share','prod_to_store_share','buy_to_store_share']
        self.default_actions               = [1, 0, 0, 1, 1]
        self.indicators['state']           = ['store','lds_f','lds_clus','gen_f','grdC','bat_clus','daytype', 'time']
        self.discrete['action']            = [0, 0, 0, 0, 0]
        self.discrete['state']             = [0, 0, 1, 0, 0, 1, 1, 0]
        self.nbins, nbins                  = [3 for _ in range(2)]
        self.bins_time                     = [0, 5, 8, 11, 14, 18, 20, 23]
        self.binno_time                    = [i for i in range(8)] + [0]
        self.nbins_time                    = 7
        nclustrip                          = 4
        ncluslds                           = 4
        self.granularity['state']          = np.array([nbins, nbins, ncluslds, nbins, nbins, nclustrip, 2, self.nbins_time])
        self.granularity['action']         = [nbins for _ in range(len(self.indicators['action']))]

        # initialise multipliers
        self.multipliers, self.combs, self.n, self.possible, self.space, self.index = {}, {}, {}, {}, {},{}
        for e in ['state','action']:
            self.multipliers[e] = []
            for i in range(len(self.granularity[e]) - 1):
                self.multipliers[e].append(np.prod(self.granularity[e][i + 1:]))
            self.multipliers[e].append(1)
            self.combs[e]          = self.listcombs_indextocomb(e)
            self.n[e]              = len(self.combs[e])
            self.possible[e]       = np.linspace(0, self.n[e]-1, num=self.n[e])
            self.space[e]          = spaces.Discrete(self.n[e])
            # self.possible_joint[e] = list(itertools.product(*[self.possible[e] for _ in range(self.n_agents)]))
            # self.index[e]          = {a: i for a,i in zip(self.possible_joint[e], range(0, self.n[e]))}
        
        self.date0                         = datetime.datetime(year = 2018, month = 6, day = 12, hour = 0)
        self.end_date                      = datetime.datetime(year = 2018, month = 11, day = 13, hour = 0)
        self.duration_days                 = (self.end_date - self.date0).days
                
        """
        A joint action is a tuple with each agent's actions.
        This property should be the list of all possible joint actions:
        """
        self.possible_joint_actions = list(itertools.product(*[self.possible['action'] for _ in range(self.n_agents)]))
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def loadnextday(self):
        day = {}
        nt  = 24
        dt  = self.labelday[self.idt]
        dtt = self.labelsdaytrans[self.idt0 * 2 + self.idt * 1]

        for a in range(self.n_agents):
            # factor for demand - differentiate between day types
            self.f[a]['lds'] = self.f[a]['lds'] + gamma.rvs(*list(self.fprm['lds'][dtt])) - self.fmean['lds'][dtt]
        
            #  factor for generation -  without differentiation between day types
            self.f[a]['gen']  = self.f[a]['gen'] + gamma.rvs(*self.fprm['gen']) - self.fmean['gen']
        
            for e in ['lds', 'gen']:
                self.f[a][e] = min(max(self.minf[e], self.f[a][e]), self.maxf[e])

            for e in ['lds', 'EV']:
                self.clus[a][e] = np.random.choice(range(self.nclus[e]), p = self.ptrans[e][dtt][self.clus[a][e]])        
        
        day['lds'] = [self.prof['lds'][dt][self.clus[a]['lds']][np.random.choice(np.arange(self.nprof['lds'][dt][self.clus[a]['lds']]))] * self.f[a]['lds'] for a in range(self.n_agents)]
        
        m_prod               = self.date.month
        while not self.nprof['gen'][m_prod - 1] > 0:
            m_prod +=1
            m_prod = 1 if m_prod == 12 else m_prod
        day['gen']      = [self.prof['gen'][m_prod - 1][np.random.choice(np.arange(self.nprof['gen'][m_prod - 1]))] * self.f[a]['gen'] for a in range(self.n_agents)]
        iEV             = [np.random.choice(np.arange(self.nprof['EV'][dt][self.clus[a]['EV']])) for a in range(self.n_agents)]
        day['lds_EV']   = [self.prof['EV']['cons'][dt][self.clus[a]['EV']][iEV[a]] for a in range(self.n_agents)]
        day['avail_EV'] = [self.prof['EV']['avail'][dt][self.clus[a]['EV']][iEV[a]] for a in range(self.n_agents)]
        
        # flexibility matrix: at which time step indicates how much energy is required with which time flexibility
        day['flex']    = [None] * self.n_agents
        for a in range(self.n_agents):
            day['flex'][a]    = [None] * nt
            for t in range(nt):
                day['flex'][a][t]                     = [0] * self.max_delay
                day['flex'][a][t][0]                  = (1 - self.share_flex) * day['lds'][a][t]
                day['flex'][a][t][self.max_delay - 1] = self.share_flex * day['lds'][a][t]
        
        for e in day.keys(): self.batch[a][e] = self.batch[a][e] + list(day[e][a])

                
    def update_flex(self, cons_flex):
        h                  = self.date.hour
        for a in range(self.n_agents):
            remaining_cons = cons_flex[a]
            # remove what has been consumed
            i_flex         = 1
            while remaining_cons > 0:
                delta_cons = min(self.batch[a]['flex'][h][i_flex], remaining_cons)
                self.batch[a]['flex'][h][i_flex] -= delta_cons
                remaining_cons                   -= delta_cons
                i_flex += 1
            
            # move what has not be consumed to one step more urgent
            for i_flex in range(self.max_delay - 1):
                self.batch[a]['flex'][h + 1][i_flex] += self.batch[a]['flex'][h][i_flex + 1]
                
    def get_reward(self, netp, netstore):
        grid      = sum(netp)
        store_out = sum([- netstore[a] if netstore[a] < 0 else 0 for a in range(self.n_agents)])
        netp2     = sum([netp[a] ** 2 for a in range(self.n_agents)])
        gc                       = self.grdC[self.t] * (grid + self.grdloss * grid ** 2)
        sc                       = self.batC * store_out
        dc                       = self.ntwC * netp2
        reward                   = - (gc + sc + dc)    
        
        return reward;
    
    def step(self, action, typereward = 'advantage', baseline = True):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if self.done == 1:
            return [self.state, self.reward, self.done]
        h = self.date.hour
        if h == 0:
            for a in range(self.n_agents):
                for e in self.batch_entries:
                    self.batch[a][e] = self.batch[a][e][24:] # remove day for the day we just finished
            self.loadnextday()
        
        # print(action)
        netps, netstores, actionvals = self.policy_to_rewardvar(action)
        
        # ----------- get rewards ---------------
        # total reward
        self.tot_reward = self.get_reward(netps['greedy'], netstores['greedy'])
        # advantages
        if typereward == 'advantage':
            advantages = []
            for a in range(self.n_agents):
                netpa, netstorea  = netps['greedy'], netstores['greedy']
                netpa[a]          = netps['default'][a]
                netstorea[a]      = netstores['default'][a]
                # def_reward.append(self.get_reward(netpa, netstorea))
                advantages.append(self.tot_reward - self.get_reward(netpa, netstorea))
            reward = advantages
        elif typereward == 'total':
            reward = self.tot_reward
        # dumb policy
        self.reward_default = self.get_reward(netps['default'], netstores['default']) if baseline else None
            
        # ----- update environment variables and state
        lds_flex               = [sum(self.batch[a]['flex'][h][1:]) for a in range(self.n_agents)]
        cons_flex              = [actionvals[a][0] * lds_flex[a] for a in range(self.n_agents)]
        self.update_flex(cons_flex)
        self.t                  += 1
        self.date                = self.date + datetime.timedelta(hours = 1)
        self.idt                 = 0 if self.date.weekday() < 5 else 1

        if self.date == self.end_date:
            self.done = True
            return None, self.done, reward, self.reward_default
        # else:
            # self.state_vals          = self.store, self.f['lds'], self.clus['lds'], self.f['gen'], self.grdC[self.t] , self.clus['EV'], self.idt, self.date.hour
        else:
            return self.get_state(), self.done, reward, self.reward_default 
    
    def policy_to_rewardvar(self, action):
        actionvals = []
        netps, netstores = [initialiseentries(['greedy','default'], typeobj = 'Nones', n = self.n_agents) for _ in range(2)]
        h = self.date.hour

        for a in range(self.n_agents):
            # action to reward
            actionavala  = self.index_to_val(self.listcombs_indextocomb('action', index = action[a]), typev = 'action')
            actionvals.append(actionavala)
            lds_flex               = sum(self.batch[a]['flex'][h][1:])
            lds_fixed              = self.batch[a]['flex'][h][0]
            prod, lds_EV, avail_EV = [self.batch[a][e][h] for e in ['gen', 'lds_EV', 'avail_EV']]
            store0                 = self.store
        
            # netp[a], netstore[a]         = self.policy_to_rewardvar(actionavala)
            # netp_def[a], netstore_def[a] = self.policy_to_rewardvar(self.default_actions)
            
            for actionvars, label in zip([actionavala, self.default_actions], ['greedy', 'default']):
                cons_flex_share, prod_to_sell_share, store_to_sell_share, prod_to_store_share, buy_to_store_share = actionvars

                # ---------------- first ensure consumption is met -------------------
                # 1 - store_to_consEV
                store_to_consEV         = min(lds_EV, store0)
                store2                  = store0 - store_to_consEV
                extra_cons_EV           = lds_EV - store_to_consEV
                
                # 2 - consumption
                cons_flex               = cons_flex_share * lds_flex
                tot_cons                = lds_flex + lds_fixed + extra_cons_EV
                
                # 3 - prod_to_cons
                prod_to_cons            = min(tot_cons, prod)
                net_prod3               = prod - prod_to_cons
                net_lds3                = tot_cons - prod_to_cons
        
                # 4 - store_to_cons
                store_to_cons           = float(min(store2, net_lds3)) * avail_EV
                store4                  = store2 - store_to_cons
                
                # 5 - buy_to_cons
                buy_to_cons             = net_lds3 - store_to_cons
                
                # ----- then see what to do with extra production / capacity -------
                # 6 - prod_to_store     
                net_cap6                = self.cap - store4
                prod_to_store_potential = min(net_cap6, net_prod3)
                prod_to_store           = prod_to_store_share * prod_to_store_potential * avail_EV
                net_prod6               = net_prod3 - prod_to_store
                store6                  = store4 + self.etach * prod_to_store 
                
                # 7 - prod_to_sell
                prod_to_sell            = prod_to_sell_share  * net_prod6
                prod_to_curt            = net_prod6 - prod_to_sell
                
                # 8 - store to sell
                store_to_sell           = store_to_sell_share * store6 * avail_EV
                store8                  = store6 - store_to_sell
                
                # 9 - buy_to_store
                net_cap9                = self.cap - store8
                buy_to_store            = buy_to_store_share * net_cap9 * avail_EV
                
                # get variables for costs
                netp                    = buy_to_store + buy_to_cons - store_to_sell - prod_to_sell
                netstore                = (prod_to_store + buy_to_store) - (store_to_sell + store_to_cons + store_to_consEV)
                self.store              = float(store0 + netstore)
                
                netps[label][a], netstores[label][a] = netp, netstore
                
        return netps, netstores, actionvals;

    def reset(self, bat, grd, lds, ntw, prm, gen):
        # initialise environment time
        self.date                       = self.date0
        self.t                          = 0 # hrs since start
        self.steps_beyond_done          = None
        self.done                       = False
        self.idt                        = 0 if self.date0.weekday() < 5 else 1
        self.idt0                       = 0 if (self.date0 - datetime.timedelta(days=1)).weekday() < 5 else 1
        
        # initialise parameters
        self.labelday                   = prm['labelday']
        self.labelsdaytrans             = prm['labelsdaytrans']
        self.share_flex, self.max_delay = lds['flex']
        self.cap                        = bat['cap']
        self.etach                      = bat['etach']
        self.grdC, self.batC, self.ntwC = grd['C'], bat['C'], ntw['C']
        self.grdloss                    = grd['R']/(grd['V']**2)
        self.minf, self.maxf, self.fprm, self.fmean, self.nclus, self.pclus, self.ptrans, self.nprof, self.prof = [{} for _ in range(9)]
        for obj, labobj in zip([lds, gen], ['lds','gen']):
            self.minf[labobj]           = np.min(obj['listfactors'])
            self.maxf[labobj]           = np.max(obj['listfactors'])
            self.fprm[labobj]           = obj['fprms']
            self.fmean[labobj]          = obj['fmean']
        for obj, labobj in zip([lds, bat], ['lds','EV']):
            self.nclus[labobj]          = obj['nclus']
            self.pclus[labobj]          = obj['pclus']
            self.ptrans[labobj]         = obj['ptrans']
        for obj, labobj in zip([lds, gen, bat], ['lds','gen','EV']):
            self.nprof[labobj]          = obj['nprof']
            self.prof[labobj]           = obj['prof']
                
        self.maxval = {}
        self.maxval['state']    = [bat['cap'], np.max(lds['listfactors']), 
                                    lds['nclus'], np.max(gen['listfactors']), 
                                    np.max(grd['C']), bat['nclus'],
                                    1, 24]
        self.maxval['action']   = [1 for _ in range(len(self.indicators['action']))]
        self.brackets           = self.init_brackets(gen, lds, bat)
        
        # initial state
        self.f, self.clus, self.store = {}, {}, {}
        for a in range(self.n_agents):
            self.f[a], self.clus[a] = {},{}
            for e in ['lds', 'gen']: self.f[a][e]    = 10
            for e in ['lds', 'EV']:  self.clus[a][e] = 0
            self.store             = bat['SoC0'] * bat['cap']    
        
        # initialise demand ahead (2 days)
        self.batch_entries = ['lds','gen', 'lds_EV', 'avail_EV', 'flex']
        self.batch = {}
        for a in range(self.n_agents):
            self.batch[a] = initialiseentries(self.batch_entries)
        for _ in range(2): self.loadnextday()
        
        return self.get_state()
    
    def render(self, mode='human', close=False):
        return 0
    
    def init_brackets(self, gen, lds, bat):
        brackets = {}
        for typev in ['state','action']:
            brackets[typev] = []
            for s in range(len(self.indicators[typev])):
                if self.discrete[typev][s] == 1:
                    brackets[typev].append([0])
                elif self.indicators[typev][s][-1] == 'f':
                    if self.indicators[typev][s][0:3] == 'gen': 
                        obj = gen
                    elif self.indicators[typev][s][0:3] == 'lds': 
                        obj = lds
                    elif self.indicators[typev][s][0:3] == 'bat':
                        obj = bat
                    listf = obj['listfactors']
                    brackets[typev].append( [np.percentile(listf, 1/self.nbins * 100 * i) for i in range(self.nbins)] )
                elif self.indicators[typev][s] == 'time':
                    brackets[typev].append(self.bins_time)
                else:  
                    brackets[typev].append( [self.maxval[typev][s]/self.granularity[typev][s] * i for i in range(self.granularity[typev][s])] )
        return brackets;    
    
    def get_state(self):
        index=[]
        for a in range(self.n_agents):
            state_vals  = [self.store , self.f[a]['lds'], self.clus[a]['lds'], self.f[a]['gen'], self.grdC[self.t], self.clus[a]['EV'], self.idt, self.date.hour]
            indexes     = []
            for s in range(len(state_vals)):
                if self.discrete['state'][s] == 1:
                    indexes.append(state_vals[s])
                else:
                    try:
                        indexes.append([i for i in range(len(self.brackets['state'][s])) if state_vals[s] >= self.brackets['state'][s][i]][-1] )
                    except Exception as ex:
                        print(ex.args)
                        print(s)
                        print(self.brackets['state'][s])
                        print(state_vals[s])
                        return 0
                        
                    if self.indicators['state'][s] == 'time':
                        indexes[-1] = self.binno_time[indexes[-1]]
            
            index.append( sum([a*b for a,b in zip(indexes, self.multipliers['state'])]) )
        
        return index;
    
    def listcombs_indextocomb(self, typev, index = None):
        granularity = self.granularity[typev]
        indicators  = np.zeros(len(granularity))
        ncombs      = 1
        for x in granularity: 
            ncombs  = ncombs * x
        combs = copy.deepcopy(indicators)
        
        nind        = len(indicators) 
        ind         = nind - 1
        allcombs    = 0
        count       = 0
        loop    = 1
        while loop:
            count +=1
            if indicators[ind] < granularity[ind] - 1:
                indicators[ind] += 1
            else:
                addhigher = 0
                while addhigher == 0:
                    indicators[ind] = 0
                    ind = ind - 1
                    if indicators[ind] < granularity[ind] - 1:
                        indicators[ind] += 1
                        addhigher = 1
                        ind = nind - 1
            maxind = sum(1 for i in range(nind) if indicators[i] == granularity[i] - 1)
            if maxind == nind:
                allcombs = 1
            illegal = 0
            if typev == 'action':
                if (indicators[4] > 0 and indicators[1] > 0) or (indicators[4] > 0 and indicators[2] > 0):
                    illegal = 1
                elif indicators[1] == granularity[1] and indicators[3] > 0:
                    illegal = 1
            if illegal == 0: combs = np.vstack((combs, indicators))
            # print(typev)
            # print(index)
            loop = (allcombs == 0 and count < 1e6) if index is None else len(combs) < index + 1
            
            obj = combs if index is None else combs[-1]
        return obj
    
    def index_to_val(self, index, typev = 'state'):
        val  = []
        for s in range(len(index)):
            if self.discrete[typev][s] == 1:
                val.append(index[s])
            else:
                brackets_s = self.brackets[typev][s] + [self.maxval[typev][s]]
                val.append((brackets_s[int(index[s])] + brackets_s[int(index[s] + 1)])/2)
        return val;     