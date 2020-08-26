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
# from gym import error, spaces, utils
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy 
import datetime
from scipy.stats import gamma
from userdeftools import initialiseentries
import itertools
# from collections import defaultdict
# import sys

class LocalElecEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_agents = 1):
        # Define action and observation space
        self.done                          = 0
        self.n_agents                      = n_agents
        self.t                             = 0
        self.indicators, self.discrete, self.granularity = {}, {}, {}
        self.indicators['action']          = ['cons_flex_share','store']
        self.default_actions               = [1, 2]
        # self.indicators['state']           = ['store','lds_f','lds_clus','gen_f','grdC','bat_clus','daytype', 'time']
        self.indicators['state']           = ['store','lds_f','lds_clus','gen_f','bat_clus','daytype', 'time']
        self.discrete['action']            = [1, 1]
        self.discrete['state']             = [0, 0, 1, 0, 1, 1, 0]
        self.nbins, nbins                  = [3 for _ in range(2)]
        self.bins_time                     = [0, 5, 8, 11, 14, 18, 20, 23]
        self.binno_time                    = [i for i in range(8)] + [0]
        self.nbins_time                    = 7
        nclustrip                          = 4
        ncluslds                           = 4
        self.granularity['state']          = np.array([nbins, nbins, ncluslds, nbins, nclustrip, 2, self.nbins_time])
        self.granularity['action']         = [2, 3]

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
        self.end_date                      = datetime.datetime(year = 2018, month = 7, day = 12, hour = 0)
        # self.end_date                      = datetime.datetime(year = 2018, month = 11, day = 13, hour = 0)
        self.duration_days                 = (self.end_date - self.date0).days
                
        """
        A joint action is a tuple with each agent's actions.
        This property should be the list of all possible joint actions:
        """
        self.possible_joint_actions = list(itertools.product(*[self.possible['action'] for _ in range(self.n_agents)]))
        self.batchfile = 'batch'

        self.envseed = self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def loadnextday(self):
        if not self.presaved:
            day = {}
            nt  = 24
            dt  = self.labelday[self.idt]
            dtt = self.labelsdaytrans[self.idt0 * 2 + self.idt * 1]
    
            for a in range(self.n_agents):
                # save fs and cluss as being at the start of the episode, before update = previous day
                for e in ['lds','gen']: self.fs[a][e].append(self.f[a][e])
                for e in ['lds','EV']: self.cluss[a][e].append(self.clus[a][e])

                # factor for demand - differentiate between day types
                df               = gamma.ppf(self.np_random.rand(), *list(self.fprm['lds'][dtt]))
                self.f[a]['lds'] = self.f[a]['lds'] +  df - self.fmean['lds'][dtt]
                
                #  factor for generation -  without differentiation between day types
                df = gamma.ppf(self.np_random.rand(), * self.fprm['gen'])
                self.f[a]['gen']  = self.f[a]['gen'] + df - self.fmean['gen']
            
                for e in ['lds', 'gen']:
                    self.f[a][e] = min(max(self.minf[e], self.f[a][e]), self.maxf[e])
    
                for e in ['lds', 'EV']:
                    ps = self.ptrans[e][dtt][self.clus[a][e]]
                    cump = [sum(ps[0:i]) for i in range(1,len(ps))] + [1]
                    rdn = self.np_random.rand()
                    self.clus[a][e] = [c > rdn for c in cump].index(True)

            iproflds = self.np_random.choice(np.arange(self.nprof['lds'][dt][self.clus[a]['lds']]))
            day['lds'] = [self.prof['lds'][dt][self.clus[a]['lds']][iproflds] * self.f[a]['lds'] for a in range(self.n_agents)]
            
            m_prod               = self.date.month
            while not self.nprof['gen'][m_prod - 1] > 0:
                m_prod +=1
                m_prod = 1 if m_prod == 12 else m_prod
            iprofgen = self.np_random.choice(np.arange(self.nprof['gen'][m_prod - 1]))
            day['gen']      = [self.prof['gen'][m_prod - 1][iprofgen] * self.f[a]['gen'] for a in range(self.n_agents)]
            iEV             = [self.np_random.choice(np.arange(self.nprof['EV'][dt][self.clus[a]['EV']])) for a in range(self.n_agents)]
            day['lds_EV']   = [self.prof['EV']['cons'][dt][self.clus[a]['EV']][iEV[a]] for a in range(self.n_agents)]
            day['avail_EV'] = [self.prof['EV']['avail'][dt][self.clus[a]['EV']][iEV[a]] for a in range(self.n_agents)]
            
            # flexibility matrix: at which time step indicates how much energy is required with which time flexibility
            day['flex']    = [None] * self.n_agents
            for a in range(self.n_agents):
                day['flex'][a]    = [None] * nt
                for t in range(nt):
                    day['flex'][a][t]                     = [0] * (self.max_delay + 1)
                    day['flex'][a][t][0]                  = (1 - self.share_flex) * day['lds'][a][t]
                    day['flex'][a][t][self.max_delay]     = self.share_flex * day['lds'][a][t]
            
            for e in day.keys(): self.batch[a][e] = self.batch[a][e] + list(day[e][a])
            
        else:
            for a in range(self.n_agents):
                for e in self.batch_entries: 
                    batchae          = np.load(self.batchfile + str(a) + e + '.npy', mmap_mode='c')
                    batchaed         = list(batchae[self.dloaded*24:(self.dloaded+1)*24].copy())
                    self.batch[a][e] = self.batch[a][e] + batchaed   
                    if len(self.batch[a][e]) < 48 and self.date > self.date0:
                        print('len(self.batch[a][e]) < 48')
                        print('a = {}, e = {}, date = {}'.format(a,e,self.date))
                        print('np.shape(self.batch[a][e]) = {}'.format(np.shape(self.batch[a][e])))
                for e in ['lds','gen']: 
                    self.f[a][e] = self.fs[a][e][(self.date - self.date0).days]
                for e in ['lds','EV']: 
                    self.clus[a][e] = self.cluss[a][e][(self.date - self.date0).days]
                
            self.dloaded += 1 

    def update_flex(self, cons_flex, opts = None):
        if opts is None:
            h          = self.date.hour      
            n_agents   = self.n_agents
            batch_flex = [self.batch[a]['flex'] for a in range(n_agents)]
            max_delay  = self.max_delay
        else:
            h, batch_flex, max_delay, n_agents = opts

        for a in range(n_agents):            
            remaining_cons = cons_flex[a]
            # remove what has been consumed
            i_flex         = 1
            
            for i_flex in range(1, max_delay + 1):
                delta_cons                = min(batch_flex[a][h][i_flex], remaining_cons)
                remaining_cons           -= delta_cons
                batch_flex[a][h][i_flex] -= delta_cons
            
            if remaining_cons > 1e-3:
                print("cons_flex[a] = {}, remaining_cons = {}".format(cons_flex[a], remaining_cons) )
                return None
            
            # print('h = {}'.format(h))
            # print('before update batch_flex[a][h] = {}'.format(batch_flex[a][h]))
            # print('before update batch_flex[a][h+1] = {}'.format(batch_flex[a][h+1]))
            # move what has not be consumed to one step more urgent
            for i_flex in range(max_delay):
                batch_flex[a][h + 1][i_flex] += batch_flex[a][h][i_flex + 1]
            # print('after update batch_flex[a][h+1] = {}'.format(batch_flex[a][h+1]))

                
        if opts is None:
            self.batch_flex = batch_flex
        else:
            return batch_flex

    def get_reward(self, netp, store_out, other_inputs = None):
        n_agents, grdCt, batC, ntwC, grdloss = [self.n_agents, self.grdC[self.t], self.batC, self.ntwC, self.grdloss] if other_inputs is None else other_inputs

        grid      = sum(netp) # negative netp is selling, positive buying 
        netp2     = sum([netp[a] ** 2 for a in range(n_agents)])
        gc        = grdCt * (grid + grdloss * grid ** 2)
        sc        = batC * sum(store_out[a] for a in range(n_agents))
        dc        = ntwC * netp2
        reward    = - (gc + sc + dc)    
        return reward;
    
    def step(self, action):
        
        if self.done == 1:
            return [self.state, self.reward, self.done]
        h = self.date.hour
        if h == 0:
            for a in range(self.n_agents):
                for e in self.batch_entries:
                    self.batch[a][e] = self.batch[a][e][24:] # remove day for the day we just finished
            self.loadnextday()
        
        netps, store_outs, actionvals = self.policy_to_rewardvar(action)
        
        tot_reward = self.get_reward(netps['greedy'], store_outs['greedy']) if self.typereward in ['advantage','total'] else self.get_reward(netps['default'], store_outs['default'])

        # ----------- get rewards ---------------
        if self.typereward == 'advantage':
            advantages = []
            for a in range(self.n_agents):
                netpa, store_outa  = netps['greedy'].copy(), store_outs['greedy'].copy()
                netpa[a]          = netps['default'][a]
                store_outa[a]      = store_outs['default'][a]
                agentabasereward = self.get_reward(netpa, store_outa)
                advantages.append(tot_reward - agentabasereward)
            reward = advantages
                
        elif self.typereward == 'total':
            reward = tot_reward
        elif self.typereward == 'baseline':
            reward = tot_reward
                
                # ----- update environment variables and state
        lds_flex               = [sum(self.batch[a]['flex'][h][1:]) for a in range(self.n_agents)]
        if self.typereward == 'baseline':
            cons_flex = [self.default_actions[0] * lds_flex[a] for a in range(self.n_agents)]
        else:
            cons_flex  = [actionvals[a][0] * lds_flex[a] for a in range(self.n_agents)]
        self.update_flex(cons_flex)
        self.t                  += 1
        self.date                = self.date + datetime.timedelta(hours = 1)
        self.idt                 = 0 if self.date.weekday() < 5 else 1

        if self.date == self.end_date:
            self.done = True
            return None, self.done, reward, tot_reward
        else:
            return self.get_state(), self.done, reward, tot_reward
        
    def policy_to_rewardvar(self, action):
        actionvals = []            
        netps, store_outs = [initialiseentries(['greedy','default'], typeobj = 'Nones', n = self.n_agents) for _ in range(2)]
        h = self.date.hour
        for a in range(self.n_agents):

            lds_flex               = sum(self.batch[a]['flex'][h][1:])        
            
            if self.typereward == 'total':
                labels = ['greedy']
                actionavala  = self.index_to_val(self.listcombs_indextocomb('action', index = action[a]), typev = 'action')
                actions  = [actionavala]
            elif self.typereward == 'advantage':
                labels = ['greedy', 'default']
                actionavala  = self.index_to_val(self.listcombs_indextocomb('action', index = action[a]), typev = 'action')
                actions  = [actionavala, self.default_actions]
            elif self.typereward == 'baseline':
                labels = ['default']
                actionavala = self.default_actions
                actions = [self.default_actions]
            actionvals.append(actionavala)
            
            for actionvars, label in zip(actions, labels):
                cons_flex_share, alpha_store = actionvars
                if self.date == self.end_date:
                    cons_flex_share = 1
                    alpha_store     = 1
                prod0, l_EV0, avail_EV       = copy.deepcopy([self.batch[a][e][h] for e in ['gen', 'lds_EV', 'avail_EV']])
                lds_fixed                    = self.batch[a]['flex'][h][0]
                lds_flex_                    = lds_flex
                totload                      = lds_fixed + cons_flex_share * lds_flex_
                load, prod, l_EV, store, imp = totload.copy(), prod0.copy(), l_EV0.copy(), self.store[a], 0
                mincharge                    = self.cap * max(self.SoCmin, self.baseld)
                charge, decharge             = 0, 0
                losscharge                   = 0
                
                #  ---------------- meet consumption --------------------
                
                #  1 - EV load
                addstore = max(0, l_EV + mincharge - store)
                
                # add from prod
                prod_to_store = min(prod, addstore/self.etach)
                prod         -= prod_to_store
                charge       += prod_to_store * self.etach
                store        += prod_to_store * self.etach
                addstore     -= prod_to_store * self.etach
                losscharge   += prod_to_store * (1 - self.etach)
                
                # buy
                buy_to_store = addstore/self.etach
                imp         += buy_to_store
                charge      += buy_to_store * self.etach
                store       += buy_to_store * self.etach
                addstore    -= buy_to_store * self.etach
                losscharge  += buy_to_store * (1 - self.etach)
                
                # consume
                decharge += l_EV
                store    -= l_EV
                if store +1e-2 < mincharge:
                    print('after lEV, store < mincharge')
                    print('store_0 = {}'.format(self.store[a]))
                    print('prod_to_store = {}'.format(prod_to_store))
                    print('buy_to_store = {}'.format(buy_to_store))
                    print('l_EV = {}'.format(l_EV))
                    print('store = {}'.format(store))
                    print('date = {}'.format(self.date))
                
                # 2 - rest of load
                # from prod
                prod_to_cons         = min(load, prod)
                prod                -= prod_to_cons
                load                -= prod_to_cons
                
                # from store
                store_to_cons        = float(min(store - mincharge, load, max(0,(self.dmax + charge - decharge)))) * avail_EV
                store               -= store_to_cons
                load                -= store_to_cons
                decharge            += store_to_cons
                
                # buy
                imp                 += load
                load                 = 0
                
                # ----- then see what to do with extra production / capacity -------
                if alpha_store == 1: # store until capacity and sell extra prod
                    # first store prod as it is free
                    prod_to_store            = min(prod, (self.cap - store)/self.etach, max(0,(self.cmax + decharge - charge)/self.etach))
                    prod                    -= prod_to_store
                    store                   += prod_to_store * self.etach
                    charge                  += prod_to_store * self.etach
                    losscharge              += prod_to_store * (1 - self.etach)
                    
                    # buy to fill in rest
                    buy_to_store = min((self.cap - store)/self.etach, max(0, (self.cmax + decharge - charge)/self.etach))
                    imp        += buy_to_store
                    store      += buy_to_store * self.etach
                    charge     += buy_to_store * self.etach
                    losscharge += buy_to_store * (1 - self.etach)
                    
                    # sell prod that does not fit in store
                    prod_to_sell             = prod
                    prod                    -= prod_to_sell
                    imp                     -= prod_to_sell
                    
                if alpha_store == 2: # sell
                    # sell prod
                    prod_to_sell             = prod
                    prod                    -= prod_to_sell
                    imp                     -= prod_to_sell
                    
                    # then also sell what i can sell in the store
                    to_sell = min(max(0, store - mincharge), max(0, self.dmax + charge - decharge))
                    imp      -= to_sell
                    store    -= to_sell 
                    decharge += to_sell
                
                # alpha_store 0 is do nothing and curtail
                curt = prod

                # get variables for costs
                netp                    = imp
                store_out               = decharge
                
                # check constraints
                # battery balanace
                if abs(store - self.store[a] - ( charge - decharge ) ) > 1e-2:
                    print('abs(store - self.store - ( charge - losscharge - decharge )) = {}, self.date = {}'.format(abs(store - self.store[a] - ( charge - losscharge - decharge )), self.date))
                # prosumer balance
                if abs(imp - totload - charge - losscharge + decharge - l_EV0 + prod0 - curt) > 1e-2:
                    print('abs(- ( charge + losscharge ) - totload + (prod0 - curt) + imp ) = {} , self.date = {}'.format(abs(- ( charge + losscharge ) - totload + (prod0 - curt) + imp ), self.date))
                if store +1e-2 < mincharge:
                    print("store < mincharge, self.date ={}".format(self.date))
                if totload < 0:
                    print("totload < 0, self.date ={}".format(self.date))
                if curt < 0 :
                    print("curt < 0, self.date ={}".format(self.date))
                    
                netps[label][a], store_outs[label][a] = netp, store_out
                
                if self.typereward == 'baseline' or self.typereward =='total' or (self.typereward == 'advantage' and label == 'greedy'):
                    self.store[a]              = store
                
        return netps, store_outs, actionvals;

    def reset(self, bat, grd, lds, ntw, prm, gen, presave, typereward, nonamefile = None, loaddata = False):
        # initialise environment time
        self.date                       = self.date0
        self.t                          = 0 # hrs since start
        self.steps_beyond_done          = None
        self.done                       = False
        self.idt                        = 0 if self.date0.weekday() < 5 else 1
        self.idt0                       = 0 if (self.date0 - datetime.timedelta(days=1)).weekday() < 5 else 1
        
        # data management
        self.presaved = True if loaddata else False
        presave       = 0 if loaddata else presave
        self.typereward = typereward
        self.dloaded = 0
        self.batchfile = self.batchfile if nonamefile is None else self.batchfile + '_' + str(nonamefile)
        self.fs, self.cluss = [initialiseentries(range(self.n_agents)) for _ in range(2)]
        for a in range(self.n_agents):
            self.fs[a], self.cluss[a] = [initialiseentries(['EV','gen','lds']) for _ in range(2)]
            if self.presaved:
                for e in ['lds','gen']: 
                    self.fs[a][e] = np.load(self.batchfile + '_fs' + str(a) + e + '.npy')
                for e in ['lds','EV']: 
                    self.cluss[a][e] = np.load(self.batchfile  + '_cluss' + str(a) + e+ '.npy')
                    
        # initialise parameters
        self.labelday, self.labelsdaytrans = [prm[e] for e in ['labelday','labelsdaytrans']]
        self.share_flex, self.max_delay    = lds['flex']
        self.cap, self.etach, self.SoCmin, self.baseld, self.dmax, self.cmax = [bat[e] for e in ['cap','etach','SoCmin','baseld','dmax','cmax']]
        self.grdC, self.batC, self.ntwC    = grd['C'], bat['C'], ntw['C']
        self.grdloss                       = grd['R']/(grd['V']**2)
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
                                    np.max(grd['C']), bat['nclus'], 1, 24]
        self.maxval['action']   = [1,2]
        self.brackets           = self.init_brackets(gen, lds, bat)
        
        # initial state
        self.f, self.clus, self.store = {}, {}, {}
        for a in range(self.n_agents):
            self.f[a], self.clus[a] = {},{}
            for e in ['lds', 'gen']: self.f[a][e]    = 10
            for e in ['lds', 'EV']:  self.clus[a][e] = 0
            self.store[a] = bat['SoC0'] * bat['cap']    
        
        # initialise demand ahead (2 days)
        self.batch_entries = ['lds','gen', 'lds_EV', 'avail_EV', 'flex']
        self.batch = {}
        for a in range(self.n_agents): self.batch[a] = initialiseentries(self.batch_entries)
        if presave:
            dateload = self.date0
            while dateload < self.end_date + datetime.timedelta(days = 2):
                self.loadnextday()
                dateload += datetime.timedelta(days = 1)
            for a in range(self.n_agents):
                for e in self.batch_entries: 
                    np.save(self.batchfile + str(a) + e, self.batch[a][e])
                for e in ['lds','gen']: np.save(self.batchfile + '_fs' + str(a) + e, self.fs[a][e])
                for e in ['lds','EV']: np.save(self.batchfile  + '_cluss' + str(a) + e, self.cluss[a][e])
                
            self.presaved = True
            self.batch[a] = initialiseentries(self.batch_entries)
            
        for _ in range(2): self.loadnextday()
        return self.get_state()
    
    def render(self, mode = 'human', close = False):
        return 0
    
    def init_brackets(self, gen, lds, bat):
        brackets = {}
        for typev in ['state','action']:
            brackets[typev] = []
            for s in range(len(self.indicators[typev])):
                indstr=self.indicators[typev][s]
                if self.discrete[typev][s] == 1:
                    brackets[typev].append([0])
                elif indstr[-1] == 'f':
                    if indstr[0:3] == 'gen': 
                        obj = gen
                    elif indstr[0:3] == 'lds': 
                        obj = lds
                    elif indstr[0:3] == 'bat':
                        obj = bat
                    listf = obj['listfactors']
                    brackets[typev].append( [np.percentile(listf, 1/self.nbins * 100 * i) for i in range(self.nbins)] )
                elif indstr == 'time':
                    brackets[typev].append(self.bins_time)
                elif indstr == 'cons_flex_share':
                    brackets[typev].append([0,1])
                elif indstr == 'netstore':
                    brackets[typev].append([0,1,2])
                else:  
                    brackets[typev].append( [self.maxval[typev][s]/self.granularity[typev][s] * i for i in range(self.granularity[typev][s])] )
        return brackets;    
    
    def get_state(self, other = None, type_ = 'state'):
        if type_ == 'state':
            if other is not None:
                store, f_lds, clus_lds, f_gen, clus_EV, idt, hour, discrete, brackets, n_agents, indicators, multipliers, binno_time = other
            else:
                # store, grdC_t, idt, hour, discrete, brackets, n_agents, indicators, multipliers, binno_time = self.store, self.grdC[self.t], self.idt, self.date.hour, self.discrete[type_], self.brackets[type_], self.n_agents, self.indicators[type_], self.multipliers[type_], self.binno_time
                store, idt, hour, discrete, brackets, n_agents, indicators, multipliers, binno_time = self.store, self.idt, self.date.hour, self.discrete[type_], self.brackets[type_], self.n_agents, self.indicators[type_], self.multipliers[type_], self.binno_time
                f_lds, clus_lds, f_gen, clus_EV = [[obj[a][label] for a in range(self.n_agents)] for obj, label in zip([self.f, self.clus, self.f, self.clus], ['lds','lds','gen','EV'])]
        elif type_ == 'action':
            if other is not None:
                cons_flex_share, store_action, discrete, brackets, n_agents, indicators, multipliers = other
        index = []
        for a in range(n_agents):
            if type_ == 'state':
                vals  = [store[a] , f_lds[a], clus_lds[a], f_gen[a], clus_EV[a], idt, hour]
            elif type_ == 'action':
                vals = cons_flex_share[a], store_action[a]
            indexes     = []
            for v in range(len(vals)):
                if discrete[v] == 1:
                    indexes.append(vals[v])
                        
                else:
                    try:
                        indexes.append([i for i in range(len(brackets[v])) if vals[v] >= brackets[v][i]][-1] )
                    except Exception as ex:
                        print(ex.args)
                        print('v = {}'.format(v))
                        print('valsv] = {}'.format(vals[v]))
                        print("self.brackets[{}][v] = {}".format(type_, brackets[v]))
                    if indicators[v] == 'time':
                        indexes[-1] = binno_time[indexes[-1]]
            
            index.append( sum([a*b for a,b in zip(indexes, multipliers)]) )
        
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
            # if typev == 'action':
            #     if (indicators[4] > 0 and indicators[1] > 0) or (indicators[4] > 0 and indicators[2] > 0):
            #         illegal = 1
            #     elif indicators[1] == granularity[1] and indicators[3] > 0:
            #         illegal = 1
            if illegal == 0: combs = np.vstack((combs, indicators))
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