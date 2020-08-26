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
from userdeftools import initialiseentries, haslength, seeded_rand, seededchoice
import pandas as pd
import seaborn as sns
from six import integer_types

class LocalElecEnv(gym.Env):
    metadata = {'render.modes': ['human']}
 
# =============================================================================
# # initialisation / data interface
# =============================================================================
    def __init__(self):
        self.batchfile0       = 'batch'
        self.envseed          = self.seed()
        self.random_seeds_use = {}
        
    def seed(self, seed = None):
        if seed is not None and not isinstance(seed, integer_types): seed = int(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _myinit(self, learn_prm, syst, state_space):
        self.syst      = syst
        self.n_agents  = syst['ntw']['n']

        # initialise parameters        
        self.minf, self.maxf, self.fprm, self.fmean, self.nclus, self.pclus, self.ptrans, self.nprof, self.prof = [{} for _ in range(9)]
        for obj, labobj in zip([syst['lds'], syst['gen']], ['lds','gen']):
            self.minf[labobj]              = np.min(obj['listfactors'])
            self.maxf[labobj]              = np.max(obj['listfactors'])
            self.fprm[labobj]              = obj['fprms']
            self.fmean[labobj]             = obj['fmean']
        self.minf['EV'] = min(min(syst['bat']['bracket_fs'][dtt]) for dtt in syst['prm']['labelsdaytrans'])
        self.maxf['EV'] = max(max(syst['bat']['bracket_fs'][dtt]) for dtt in syst['prm']['labelsdaytrans'])
        
        for obj, labobj in zip([syst['lds'], syst['bat']], ['lds','EV']):
            self.nclus[labobj]             = obj['nclus']
            self.pclus[labobj]             = obj['pclus']
            self.ptrans[labobj]            = obj['ptrans']
        for obj, labobj in zip([syst['lds'], syst['gen'], syst['bat']], ['lds','gen','EV']):
            self.nprof[labobj]             = obj['nprof']
            self.prof[labobj]              = obj['prof']
        
        self.alpha_manager = Alpha_manager(syst, learn_prm)
        self.utils         = Utilities(self, learn_prm)
        self.utils.new_state_space(state_space)
        self.space         = self.utils.get_spaces(learn_prm['typeenv'])
        self.observation_space, self.action_space = [self.space[s] for s in ['state','action']]
        self.duration_days = (syst['prm']['date_end'] - syst['prm']['date0']).days
    
    def loadnextday(self, envseed = None):
        if not self.presaved:
            day = {}
            nt  = 24
            dt  = self.syst['prm']['labelsday'][self.idt]
            dtt = self.syst['prm']['labelsdaytrans'][self.idt0 * 2 + self.idt * 1]
            fEV_new_interval = np.zeros((self.n_agents,))
            for a in range(self.n_agents):
                # save fs and cluss as being at the start of the episode, before update = previous day
                for e in ['lds','gen','EV']: self.fs[a][e].append(self.f[a][e])
                for e in ['lds','EV']: self.cluss[a][e].append(self.clus[a][e])

                # factor for demand - differentiate between day types
                # if envseed is None:
                rand = self.np_random.rand()
                # else:
                    # rand, self.random_seeds_use[envseed] = seeded_rand(envseed, self.random_seeds_use[envseed])
                df               = gamma.ppf(rand, *list(self.fprm['lds'][dtt]))
                self.f[a]['lds'] = self.f[a]['lds'] +  df - self.fmean['lds'][dtt]
                
                #  factor for generation -  without differentiation between day types
                # if envseed is None:
                rand = self.np_random.rand()
                # else:
                    # rand, self.random_seeds_use[envseed] = seeded_rand(envseed, self.random_seeds_use[envseed])
                df = gamma.ppf(rand, * self.fprm['gen'])
                self.f[a]['gen']  = self.f[a]['gen'] + df - self.fmean['gen']
                
                # factor for EV consumption
                current_interval = [i for i in range(self.syst['bat']['intervals_fprob'] - 1) if self.syst['bat']['bracket_fs'][dtt][i] <= self.f[a]['EV']][-1]
                # if envseed is None:
                fEV_new_interval[a] = self.np_random.choice(range(self.syst['bat']['intervals_fprob'] - 1), p = self.syst['bat']['f_prob'][dtt][current_interval])
                # else:
                    # new_interval, self.random_seeds_use[envseed] = seededchoice(envseed, range(self.syst['bat']['intervals_fprob'] - 1), pd = self.syst['bat']['f_prob'][dtt][current_interval], number_use = self.random_seeds_use[envseed])
                self.f[a]['EV']  = self.syst['bat']['mid_fs'][dtt][int(fEV_new_interval[a])]
                for e in ['lds', 'gen' ]:
                    self.f[a][e]  = min(max(self.minf[e], self.f[a][e]), self.maxf[e])
    
                for e in ['lds', 'EV']:
                    ps = self.ptrans[e][dtt][self.clus[a][e]]
                    cump = [sum(ps[0:i]) for i in range(1,len(ps))] + [1]
                    # if envseed is None:
                    rdn = self.np_random.rand()
                    # else:
                        # rdn, self.random_seeds_use[envseed] = seeded_rand(envseed, self.random_seeds_use[envseed])
                    self.clus[a][e]  = [c > rdn for c in cump].index(True)
            # if envseed is None:
            iproflds   = self.np_random.choice(np.arange(self.nprof['lds'][dt][self.clus[a]['lds']]))
            # else:
                # iproflds, self.random_seeds_use[envseed] = seededchoice(envseed, np.arange(self.nprof['lds'][dt][self.clus[a]['lds']]), number_use = self.random_seeds_use[envseed])
            day['lds'] = [self.prof['lds'][dt][self.clus[a]['lds']][iproflds] * self.f[a]['lds'] for a in range(self.n_agents)]
            
            m_prod               = self.date.month
            while not self.nprof['gen'][m_prod - 1] > 0:
                m_prod +=1
                m_prod = 1 if m_prod == 12 else m_prod
            # if envseed is None:
            iprofgen        = self.np_random.choice(np.arange(self.nprof['gen'][m_prod - 1]))
            iEV             = [self.np_random.choice(np.arange(self.nprof['EV'][dt][self.clus[a]['EV']])) for a in range(self.n_agents)]
                
            day['gen']      = [[g * self.f[a]['gen'] for g in self.prof['gen'][m_prod - 1][iprofgen]] for a in range(self.n_agents)]
            day['lds_EV']   = [[x * self.f[a]['EV'] for x in self.prof['EV']['cons'][dt][self.clus[a]['EV']][iEV[a]]] for a in range(self.n_agents)]
            for a in range(self.n_agents):
                it = 0
                while np.max(day['lds_EV'][a]) > self.syst['bat']['cap'] and it < 100:
                    if fEV_new_interval[a] > 0:
                        fEV_new_interval[a] -= 1
                        self.f[a]['EV']  = self.syst['bat']['mid_fs'][dtt][int(fEV_new_interval[a])]
                        day['lds_EV']   = [[x * self.f[a]['EV'] for x in self.prof['EV']['cons'][dt][self.clus[a]['EV']][iEV[a]]] for a in range(self.n_agents)]
                    else:
                        iEV[a] = self.np_random.choice(np.arange(self.nprof['EV'][dt][self.clus[a]['EV']]))
                    it += 1
                    
            day['avail_EV'] = [self.prof['EV']['avail'][dt][self.clus[a]['EV']][iEV[a]] for a in range(self.n_agents)]
            
            # flexibility matrix: at which time step indicates how much energy is required with which time flexibility
            day['flex']    = [None] * self.n_agents
            for a in range(self.n_agents):
                day['flex'][a]    = [None] * nt
                for t in range(nt):
                    day['flex'][a][t]                     = [0] * (self.syst['lds']['max_delay'] + 1)
                    day['flex'][a][t][0]                  = (1 - self.syst['lds']['share_flex']) * day['lds'][a][t]
                    day['flex'][a][t][self.syst['lds']['max_delay']]     = self.syst['lds']['share_flex'] * day['lds'][a][t]
            
            for e in day.keys(): self.batch[a][e] = self.batch[a][e] + list(day[e][a])
            
        else:
            for a in range(self.n_agents):
                for e in self.batch_entries: 
                    batchae          = np.load(self.batch_file + '_a' + str(a) + '_' + e + '.npy', mmap_mode = 'c')
                    batchaed         = list(batchae[self.dloaded*24:(self.dloaded+1)*24].copy())
                    self.batch[a][e] = self.batch[a][e] + batchaed   
                for e in ['lds','gen','EV']: 
                    self.f[a][e] = self.fs[a][e][(self.date - self.syst['prm']['date0']).days]
                for e in ['lds','EV']: 
                    self.clus[a][e] = self.cluss[a][e][(self.date - self.syst['prm']['date0']).days]
                
            self.dloaded += 1 
            
    def reset(self, presave = True, nonamefile = None, seed = None, loaddata = False, presaved = None):
        if presaved is not None:
            self.presaved = presaved
        if seed is not None:
            self.envseed = self.seed(seed)
            self.random_seeds_use[seed] = 0
            
        # initialise environment time
        self.date              = self.syst['prm']['date0']
        self.t                 = 0 # hrs since start
        self.steps_beyond_done = None
        self.done              = False
        self.idt               = 0 if self.syst['prm']['date0'].weekday() < 5 else 1
        self.idt0              = 0 if (self.syst['prm']['date0'] - datetime.timedelta(days=1)).weekday() < 5 else 1

        # data management
        self.presaved       = True if loaddata else False
        if loaddata: presave = 0
        self.dloaded        = 0
        
        self.batch_file      = self.batchfile0 if nonamefile is None else self.batchfile0 + '_' + str(nonamefile)
        self.fs, self.cluss = [initialiseentries(range(self.n_agents)) for _ in range(2)]
        for a in range(self.n_agents):
            self.fs[a], self.cluss[a] = [initialiseentries(['EV','gen','lds']) for _ in range(2)]
            if self.presaved:
                for e in ['lds','gen', 'EV']: 
                    self.fs[a][e] = np.load(self.batch_file + '_fs_a' + str(a) + '_' + e + '.npy')
                for e in ['lds','EV']: 
                    self.cluss[a][e] = np.load(self.batch_file  + '_cluss_a' + str(a) + '_' + e + '.npy')
        
        # initial state
        self.f, self.clus, self.store = [{} for _ in range(3)]
        for a in range(self.n_agents):
            self.f[a], self.clus[a]= [{} for _ in range(2)]
            for e in ['lds', 'gen', 'EV']: 
                self.f[a][e]     = self.syst['prm']['f0'][e]
            for e in ['lds', 'EV']:  
                self.clus[a][e]  = self.syst['prm']['clus0'][e]
            self.store[a] = self.syst['bat']['store0']
        
        # initialise demand ahead (2 days)
        self.batch_entries = ['lds','gen', 'lds_EV', 'avail_EV', 'flex']
        self.batch = {}
        for a in range(self.n_agents): self.batch[a] = initialiseentries(self.batch_entries)
        if presave:
            dateload = self.syst['prm']['date0']
            while dateload < self.syst['prm']['date_end'] + datetime.timedelta(days = 2):
                self.loadnextday(seed)
                dateload += datetime.timedelta(days = 1)
            for a in range(self.n_agents):
                for e in self.batch_entries: 
                    np.save(self.batch_file + '_a' + str(a) + '_' + e, self.batch[a][e])
                for e in ['lds','gen','EV']: np.save(self.batch_file + '_fs_a' + str(a) + '_' + e, self.fs[a][e])
                for e in ['lds','EV']: np.save(self.batch_file  + '_cluss_a' + str(a) + '_' + e, self.cluss[a][e])
                self.batch[a] = initialiseentries(self.batch_entries)
            self.presaved = True
            self.dloaded  = 0
            
        for _ in range(2): 
            self.loadnextday(seed)
        
        return self.batch_file, self.batch

    def update_flex(self, cons_flex, opts = None):
        """ step updates """
        if opts is None:
            h          = self.date.hour      
            n_agents   = self.n_agents
            batch_flex = [self.batch[a]['flex'] for a in range(n_agents)]
        else:
            h, batch_flex_, self.syst['lds']['max_delay'], n_agents = opts
            batch_flex = copy.deepcopy(batch_flex_)

        for a in range(n_agents):            
            remaining_cons  = cons_flex[a]
            remaining_cons0 = remaining_cons
            batch_flexah0   = batch_flex[a][h].copy()
            # remove what has been consumed
            i_flex          = 1
            
            for i_flex in range(1, self.syst['lds']['max_delay'] + 1):
                delta_cons                = min(batch_flex[a][h][i_flex], remaining_cons)
                remaining_cons           -= delta_cons
                batch_flex[a][h][i_flex] -= delta_cons
            
            if remaining_cons > 1e-3:
                print("cons_flex[a] = {}, remaining_cons = {}".format(cons_flex[a], remaining_cons) )
                print('h = {}, len(batch_flex[a]) = {}, remaining_cons0 = {}'.format(h, len(batch_flex[a]), remaining_cons0 ))
                print('batch_flexah0 = {}'.format(batch_flexah0))
                return None
            
            # move what has not be consumed to one step more urgent
            for i_flex in range(self.syst['lds']['max_delay']):
                batch_flex[a][h + 1][i_flex] += batch_flex[a][h][i_flex + 1]
                
        if opts is None:
            for a in range(n_agents):
                self.batch[a]['flex'] = batch_flex[a]
        else:
            return batch_flex

    def get_reward(self, netp, store_out, other_inputs = None):
        n_agents, grdCt, batC, ntwC, grdloss = [self.n_agents, self.syst['grd']['C'][self.t], self.syst['bat']['C'], self.syst['ntw']['C'], self.syst['grd']['loss']] if other_inputs is None else other_inputs
        grid      = sum(netp) # negative netp is selling, positive buying 
        netp2     = sum([netp[a] ** 2 for a in range(n_agents)])
        gc        = grdCt * (grid + grdloss * grid ** 2)
        sc        = batC * sum(store_out[a] for a in range(n_agents))
        dc        = ntwC * netp2
        reward    = - (gc + sc + dc)    
        return reward;
        
    def step(self, action, implement = True, record = False, envseed = None):
        h = self.date.hour
        
        # copy variables that are edited during the step
        store0, batch0, date0, t0, idt0, done0  = self.store.copy(), self.batch.copy(), self.t, self.date, self.idt, self.done
        
        # update batch if needed
        daynumber = (self.date - self.syst['prm']['date0']).days
        if h == 1 and self.t > 1 and self.dloaded < daynumber + 2 == 0:
            for a in range(self.n_agents):
                for e in self.batch_entries:
                    self.batch[a][e] = self.batch[a][e][24:] # remove day for the day we just finished
            self.loadnextday(envseed)
            
        alpha_actions, netps, store_outs, cons_flex, _, bool_flex, bool_penalty, constraint_ok = self.policy_to_rewardvar(action)
        reward                                                                                 = self.get_reward(netps, store_outs)
    
        if reward > 400:
            print('store0 = {}, self.store = {}, self.batch[0][lds_EV] = {}, lds = {}, gen = {}, date0 = {}, t0 = {}, idt0 = {}'.format(store0, self.store, self.batch[0]['lds_EV'][h], self.batch[0]['lds'][h], self.batch[0]['gen'][h], date0, t0, idt0))
            print('alpha_actions = {}, netps = {}, store_outs = {}, cons_flex = {}, reward= {}'.format(alpha_actions, netps, store_outs, cons_flex, reward))
            
        # ----- update environment variables and state
        self.update_flex(cons_flex)
        self.t    += 1
        self.date += datetime.timedelta(hours = 1)
        self.idt   = 0 if self.date.weekday() < 5 else 1

        if self.date == self.syst['prm']['date_end']:
            self.done = True
        
        next_state    = self.utils.get_space_indexes(done = self.done, all_vals = self.get_state_vals()) if not self.done else [None for a in range(self.n_agents)]
        done          = self.done
        if not implement: # resume to initial variables
            self.store, self.batch, self.t, self.date, self.idt0, self.done = store0, batch0, date0, t0, idt0, done0

        if record:
            ldflex  = [sum(self.batch[a]['flex'][h][1:]) for a in range(self.n_agents)]
            ldfixed = [self.batch[a]['flex'][h][0] for a in range(self.n_agents)]
            record_output = netps, store_outs, alpha_actions, reward, store0, ldflex, ldfixed
            return next_state, self.done, reward, bool_flex, bool_penalty, record_output
        else:
            return next_state, done, reward, bool_flex, bool_penalty, constraint_ok
    
    def update_fixed_cons(self, prod, store, load, l_EV, avail_EV, charge, losscharge, netp, discharge, store_out_other):
        """ update ﻿variables after non flexible consumption is met """
        etach = self.syst['bat']['etach']
        mincharge = self.syst['bat']['store0'] if self.date == self.syst['prm']['date_end'] else self.syst['bat']['mincharge']
        #  1 - EV load
        if avail_EV == 0 and l_EV > store:
            addstore = l_EV - store
            bool_penalty = 1
        else:
            addstore = max(0, l_EV + mincharge - store) if avail_EV else 0
            bool_penalty  = 0
        
        # add from prod
        prod_to_store = min(prod, addstore/etach, (self.syst['bat']['cmax'] - charge)/etach) 
        prod         -= prod_to_store
        charge       += prod_to_store * etach
        store        += prod_to_store * etach
        addstore     -= prod_to_store * etach
        losscharge   += prod_to_store * (1 - etach)
        
        # buy
        buy_to_store = min(addstore/etach, (self.syst['bat']['cmax'] - charge)/etach) 
        netp        += buy_to_store
        charge      += buy_to_store * etach
        store       += buy_to_store * etach
        addstore    -= buy_to_store * etach
        losscharge  += buy_to_store * (1 - etach)
        
        # consume
        discharge += l_EV
        store     -= l_EV

        # 2 - rest of load
        # from prod
        prod_to_cons         = min(load, prod)
        prod                -= prod_to_cons
        load                -= prod_to_cons
        
        # from store
        store_to_cons        = float(max(0,min(store - mincharge, load, max(0,(self.syst['bat']['dmax'] + charge - discharge))))) * avail_EV
        store               -= store_to_cons
        load                -= store_to_cons
        discharge            += store_to_cons
        store_out_other      += (store_to_cons)
        
        # buy
        netp                += load
        load                 = 0
        
        return prod, store, charge, losscharge, netp, discharge, store_out_other, bool_penalty
    
    def policy_to_rewardvar(self, action, other_input = None, update = True):
        if other_input is None:
            n_agents, date    = self.n_agents, self.date
            h                 = date.hour
            lds_flexs         = [sum(self.batch[a]['flex'][h][1:]) for a in range(n_agents)]
            lds_fixeds        = [self.batch[a]['flex'][h][0] for a in range(n_agents)]
            prods, l_EVs, avail_EVs = [[copy.deepcopy(self.batch[a][e][h]) for a in range(n_agents)] for e in ['gen', 'lds_EV', 'avail_EV']]
            store0s           = [self.store[a] for a in range(n_agents)]
            alpha_actions     = self.utils.index_to_val([action[a] for a in range(n_agents)], typev = 'action')
        else:
            n_agents, date, lds_flexs, lds_fixeds, prods, l_EVs, avail_EVs, store0s, alpha_actions = other_input
            update = False
            h      = date.hour
            
        cons_flex, end_stores = [[[] for _ in range(n_agents)] for a in range(2)]
        netps, store_outs = [[] for _ in range(2)]
        
        for a in range(n_agents):
            lds_flex      = lds_flexs[a]     
            alpha_action  = alpha_actions[a]
            
            if date == self.syst['prm']['date_end']: alpha_action = 1
            prod0, l_EV, avail_EV      = prods[a], l_EVs[a], avail_EVs[a]
            
            lds_fixed                  = lds_fixeds[a]
            store0                     = store0s[a]
            if date + datetime.timedelta(hours = 1) == self.syst['prm']['date_end']:
                lds_fixed += max(0, self.syst['bat']['SoC0'] * self.syst['bat']['cap'] - store0)
                mincharge = self.syst['bat']['SoC0'] * self.syst['bat']['cap']
            else:
                mincharge = self.syst['bat']['mincharge']
            charge, discharge, netp, losscharge, store_out_other  = [0 for _ in range(5)]
            
            #  ---------------- meet consumption --------------------

            prod, store, charge, losscharge, netp, discharge, store_out_other, bool_penalty = self.update_fixed_cons(prod0, store0, lds_fixed, l_EV, avail_EV, charge, losscharge, netp, discharge, store_out_other)
            constraint_ok = True
            if store < - 1e-3 :
                constraint_ok = False
            # ----- then see what to do with extra production / capacity -------
            store, netp, losscharge, charge, discharge, store_out_other, lds_flex_cons, bool_flex = self.alpha_manager.update_vars_from_alpha(alpha_action, prod, store, charge, discharge, store_out_other, netp, losscharge, lds_flex, avail_EV)

            if store < - 1e-3 :
                constraint_ok = False
            totcons = lds_fixed + lds_flex_cons
            curt    = 0
            
            # prosumer balance
            if abs(netp - (losscharge + charge)  + (discharge - l_EV) + prod0 - curt - totcons) > 1e-3:
                print('prosumer energy balance sum = {}'.format(netp - (losscharge + charge)  + (discharge - l_EV) + prod0 - curt - totcons))
                constraint_ok = False
            # battery balance
            if abs(discharge - (store_out_other + l_EV)) > 1e-3: 
                print('battery discharge sum = {}'.format(discharge - (store_out_other + l_EV)))
                constraint_ok = False
            if abs(losscharge - ((charge + losscharge) * (1 - self.syst['bat']['etach'])) ) > 1e-3: 
                print('sum loss charge = {}'.format(losscharge - ((charge + losscharge) * (1 - self.syst['bat']['etach'])) ))
                constraint_ok = False
            if abs(self.syst['bat']['etach'] * (charge + losscharge) - discharge - (store - store0)) > 1e-3: 
                print('battery energy balance sum = {}'.format(self.syst['bat']['etach'] * (charge + losscharge) - discharge - (store - store0)))
                constraint_ok = False
            if date == self.syst['prm']['date0'] and store0 != self.syst['bat']['store0']: 
                print('store init not store0')
                constraint_ok = False
            if date == self.syst['prm']['date_end'] and abs(store - self.syst['bat']['store0']) > 1e-2: 
                print('store end not store0')
                constraint_ok = False
            
            # positivity constraint
            if charge > avail_EV * self.syst['prm']['M'] and bool_penalty == 0: 
                print('charge but EV not available and no bool_penalty')
                constraint_ok = False
            if store_out_other > avail_EV * self.syst['prm']['M']: 
                print('discharge (else than EV cons) but EV not available')
                constraint_ok = False
            if store > self.syst['bat']['cap'] + 1e-3: 
                print('store larger than cap')
                constraint_ok = False
            if store < (self.syst['bat']['SoCmin'] - 1e-3) * self.syst['bat']['cap'] * avail_EV  and bool_penalty == 0: 
                print('store smaller than SoCmin and no bool_penalty, store = {}, availEV = {}, charge = {}, cmax = {}'.format(store, avail_EV, charge, self.syst['bat']['cmax']))
                constraint_ok = False
            if store > -1e-3 and store < self.syst['bat']['baseld'] * self.syst['bat']['cap'] * avail_EV - 1e-3: 
                print('store smaller than base load')
                constraint_ok = False
            if charge > self.syst['bat']['cmax'] + 1e-3: 
                print(f"charge {charge} > cmax {self.syst['bat']['cmax']}")
                constraint_ok = False
            if discharge > self.syst['bat']['dmax'] + 1e-3: 
                print(f"discharge {discharge} > dmax {self.syst['bat']['dmax']}")
                constraint_ok = False
            if totcons < 0: 
                print(f'negative totcons {totcons}')
                constraint_ok = False
            if store < - 1e-3: 
                print(f'negative store {store}')
                constraint_ok = False
            if charge < 0: 
                print(f'negative charge {charge}')
                constraint_ok = False
            if losscharge < 0: 
                print(f'negative losscharge {losscharge}')
                constraint_ok = False
            if discharge < 0: 
                print(f'negative discharge {discharge}')
                constraint_ok = False
            if store_out_other < - 1e-3: 
                print('negative store_out_other = {}'.format(store_out_other))
                constraint_ok = False
            if curt < 0: 
                print('negative curt')
                constraint_ok = False
            
            # get variables for costs
            netps.append(netp)
            store_outs.append(discharge)
            
            if update:
                self.store[a] = store
            end_stores[a] = store
            cons_flex[a]  = lds_flex_cons
            
        return alpha_actions, netps, store_outs, cons_flex, end_stores, bool_flex, bool_penalty, constraint_ok
    
    def get_state_vals(self, inds = None):
        """ get values (before translation into index) 
        corresponding to array of indicators inds inputted
        do not input alpha 
        if obtaining state from outside environment, input inds and info
        """
        
        inds = inds if inds is not None else self.utils.indicators['state']
        vals = []
        for ind in inds:
            if ind == None:
                val = None
            elif ind == 'hour':
                val = self.date.hour
            elif ind == 'store0':
                val = self.store 
            elif ind == 'grdC':
                val = self.syst['grd']['C'][self.t] 
            elif ind == 'daytype':
                val = self.idt
            elif len(ind) > 9 and (ind[-9:-5] == 'fact' or ind[-9:-5] == 'clus'): 
                # scaling factors / profile clusters for the whole day
                module = ind.split('_')[0] # EV, lds or gen
                if ind.split('_')[-1] == 'prev':
                    prev_data = self.fs if ind[-9:-5] == 'fact' else self.cluss
                    val       = [prev_data[a][module][-1] for a in range(self.n_agents)]
                else: # step
                    step_data = self.f if ind[-9:-5] == 'fact' else self.clus
                    val       = [step_data[a][module] for a in range(self.n_agents)]                
                
            else:               # select current or previous hour - step or prev
                h   = self.date.hour if ind[-4:] == 'step' else self.date.hour - 1
                if len(ind) > 8 and ind[0:8] == 'EV_avail':
                    val = [self.batch[a]['avail_EV'][h] for a in range(self.n_agents)]
                
                elif ind[0:3] == 'lds':
                    val = [np.sum(self.batch[a]['flex'][h]) for a in range(self.n_agents)]
                else:
                    # remaining are gen_prod_step / prev and EV_cons_step / prev
                    batch_type = 'gen' if ind[0:3] == 'gen' else 'lds_EV'
                    val = [self.batch[a][batch_type][h] for a in range(self.n_agents)]
                    
            vals.append(val)
            return vals
    
class Utilities():
    def __init__(self, env, learn_prm):
        self.name     = 'utilities'
        self.n_agents = env.n_agents
        self.list_factors = {}
        for e, obj in zip(['gen','lds','bat'], [env.syst['gen'], env.syst['lds'], env.syst['bat']]): 
            self.list_factors[e] = obj['listfactors']
        
        # info on state and action spaces
        maxEVcons, max_normcons_hour, max_normprod_hour = [-1 for _ in range(3)]
        for dt in ['wd','we']:
            for c in range(env.nclus['EV']):
                if np.max(env.prof['EV']['cons'][dt][c]) > maxEVcons:
                    maxEVcons = np.max(env.prof['EV']['cons'][dt][c])
            for c in range(env.nclus['lds']):
                if np.max(env.prof['lds'][dt][c]) > max_normcons_hour:
                    max_normcons_hour = np.max(env.prof['lds'][dt][c]) # max_normcons_hour = 0.357
            for m in range(12):
                if len(env.prof['gen'][m]) > 0 and np.max(env.prof['gen'][m])  > max_normprod_hour:
                    max_normprod_hour = np.max(env.prof['gen'][m])     # max_normprod_hour = 0.368

        syst = env.syst
        columns     = ['name',          'min',             'max',                                 'n',                         'discrete']
        
        info        = [['none',          None,               None,                                1,                           0],
                       ['hour',          0,                  24,                                  syst['prm']['n_hour'],          0],
                       ['store0',        0,                  syst['bat']['cap'],                     learn_prm['n_other_states'],    0],
                       ['grdC',          min(syst['grd']['C']), max(syst['grd']['C']),                  learn_prm['n_other_states'],    0],
                       ['daytype',       0,                  1,                                   2,                           1],
                       ['EV_avail_step', 0,                  1,                                   2,                           1],
                       ['EV_avail_prev', 0,                  1,                                   2,                           1],
                       # clusters - for whole day
                       ['lds_clus_step', 0,                  env.nclus['lds'] - 1,                env.nclus['lds'],           1],
                       ['lds_clus_prev', 0,                  env.nclus['lds'] - 1,                env.nclus['lds'],           1],
                       ['EV_clus_step',  0,                  env.nclus['EV'] - 1,                 env.nclus['EV'],            1],
                       ['EV_clus_prev',  0,                  env.nclus['EV'] - 1,                 env.nclus['EV'],            1],
                       # scaling factors - for whole day
                       ['lds_fact_step', env.minf['lds'],    env.maxf['lds'],                     learn_prm['n_other_states'],    0],
                       ['lds_fact_prev', env.minf['lds'],    env.maxf['lds'],                     learn_prm['n_other_states'],    0],
                       ['gen_fact_step', env.minf['gen'],    env.maxf['gen'],                     learn_prm['n_other_states'],    0],
                       ['gen_fact_prev', env.minf['gen'],    env.maxf['gen'],                     learn_prm['n_other_states'],    0],
                       ['EV_fact_step',  env.minf['EV'],     env.maxf['EV'],                      learn_prm['n_other_states'],    0],
                       ['EV_fact_prev',  env.minf['EV'],     env.maxf['EV'],                      learn_prm['n_other_states'],    0],
                       # absolute value at time step / hour
                       ['lds_cons_step', 0,                  max_normcons_hour * env.maxf['lds'], learn_prm['n_other_states'],    0],
                       ['lds_cons_prev', 0,                  max_normcons_hour * env.maxf['lds'], learn_prm['n_other_states'],    0],
                       ['gen_prod_step', 0,                  max_normprod_hour * env.maxf['gen'], learn_prm['n_other_states'],    0],
                       ['gen_prod_prev', 0,                  max_normprod_hour * env.maxf['gen'], learn_prm['n_other_states'],    0],
                       ['EV_cons_step',  0,                  maxEVcons,                           learn_prm['n_other_states'],    0],
                       ['EV_cons_prev',  0,                  maxEVcons,                           learn_prm['n_other_states'],    0],
                       
                       # action
                       ['alpha',         0,                  1,                                   learn_prm['n_action'],          0]]
         
        self.space_info    = pd.DataFrame(info, columns = columns)
        
    def new_state_space(self, state_space):
        # initialiase info on current inidicators for state and action spaces
        [self.indicators, self.granularity, self.maxval, self.minval, self.multipliers, 
         self.global_multipliers, self.n, self.discrete, self.possible] = [{} for _ in range(9)]
        for s, ind in zip(['state','action'], [state_space, ['alpha']]): # s looping through state and action spaces
            self.indicators[s]    = ind
            name_indicators       = ['none'] if ind == [None] else ind
            ind_idx               = [self.space_info['name'] == ind for ind in name_indicators]
            subtable              = [self.space_info.loc[i] for i in ind_idx]
            self.granularity[s], self.minval[s], self.maxval[s], self.discrete[s] = [[row[field].values.item() for row in subtable] for field in ['n','min','max', 'discrete']]
            self.n[s]             = np.prod(self.granularity[s])
            # initialise multipliers
            self.multipliers[s]   = self.granularity_to_multipliers(self.granularity[s])
            self.possible[s]      = np.linspace(0, self.n[s] - 1, num = self.n[s])                
            # initialise global multipliers for going to agent states and actions to global states and actions
            self.global_multipliers[s] = []
            for i in range(self.n_agents - 1): self.global_multipliers[s].append(np.prod(self.n[s]))
            self.global_multipliers[s].append(1)
            
        self.brackets         = self.init_brackets() # need to first define indicators to define brackets            
        
    def granularity_to_multipliers(self, granularity):
        multipliers = []
        for i in range(len(granularity) - 1): multipliers.append(np.prod(granularity[i + 1:]))
        multipliers.append(1)
        
        return multipliers

    def global_to_indiv_index(self, typev, global_ind, multipliers = None):
        if multipliers is None:
            if typev == 'global_state':
                granularity = [self.n['state'] for _ in range(self.n_agents)]
                multipliers = self.granularity_to_multipliers(granularity)
            
            elif typev == 'global_action':
                granularity = [self.n['action'] for _ in range(self.n_agents)]
                multipliers = self.granularity_to_multipliers(granularity)
            else:
                multipliers = self.multipliers[typev]
        n = len(multipliers) 
        indexes = [[] for _ in range(len(multipliers))]
        remaining = global_ind
        for i in range(n):
            indexes[i] = int((remaining - remaining % multipliers[i])/multipliers[i])
            remaining -= indexes[i] * multipliers[i]
            
        return indexes

    def indiv_to_global_index(self, type_indicator, indexes = None, multipliers = None, done = False):
        if indexes is None and type_indicator == 'state':
            if done: 
                indexes = [None for a in range(self.n_agents)]
            else:
                indexes     = self.get_space_indexes(done = done, all_vals = self.get_state_vals()) 
        elif indexes is None and type_indicator == 'action':
            print('need to input action indexes for ind  iv_to_global_index')
        if multipliers is None:
            multipliers = self.global_multipliers[type_indicator]
        
        to_sum, some_None = [], 0
        for a,b in zip(indexes, multipliers):
            if a is not None:
                to_sum.append(a*b)
            else:
                some_None = 1
        if some_None:
            global_index = None
        else:
            global_index = sum(to_sum)
        
        return global_index    

    def index_to_val(self, index, typev = 'state'):
        val  = []
        for s in range(len(index)):
            if self.discrete[typev][s] == 1:
                val.append(index[s])
            else:
                brackets_s = self.brackets[typev][s] + [self.maxval[typev][s]]
                val.append((brackets_s[int(index[s])] + brackets_s[int(index[s] + 1)])/2)
                
        return val    
    
    def get_space_indexes(self, done = False, all_vals = None, info = None, type_ = 'state'):
        """ return an array of indexes of current environment state or action for each agent
            inputs: 
                all_vals :  = env.get_state_vals() / or values directly for action or if values inputted are not that of the current environment
                info     : optional - input for action or if not testing current environment
                type_    : 'action' or 'state' - default 'state'
        """
        # if info  not inputted, first obtain them
        if info is not None:
            discrete, brackets, n_agents, indicators, multipliers = info
        else:
            [discrete, brackets, n_agents, indicators, multipliers] = [self.discrete[type_], 
                        self.brackets[type_], self.n_agents, self.indicators[type_], self.multipliers[type_]]
            
        if type_ == 'state':
            if done: # if the sequence is over, return None
                return [ None for a in range(self.n_agents) ]
            if self.indicators['state'] == [ None ]: # if the state space is None, return 0
                return [ 0 for a in range(self.n_agents) ]
            if all_vals is None: 
                print('input all_vals = self.get_state_vals() from env')
                
        elif type_ == 'action': # for action alpha, input values
            alpha_action = all_vals
        
        # translate values into indexes
        index = [] # one global index per agent
        for a in range(n_agents):
            if type_ == 'state':
                vals_a = []
                for ind, val in zip(indicators, all_vals):
                    # correct negative values
                    if ind == 'gen_prod_prev' and val[a] < 0:
                        print('gen = {}, correct to 0'.format(val[a]))
                        val[a] = 0
                    # take agent specific value if relevant
                    val_a = val[a] if haslength(val) else val
                    vals_a.append(val_a)
            elif type_ == 'action':
                vals_a = [alpha_action[a]]
                
            indexes     = [] # one index per value - for current agent
            for v in range(len(vals_a)):
                if discrete[v] == 1:
                    indexes.append(int(vals_a[v]))
                else:    
                    # correct if value is smaller than smallest bracket
                    if vals_a[v] < brackets[v][0]:
                        if abs(vals_a[v] - brackets[v][0]) > 1e-2: 
                            print('ind set to smallest value but really too small')
                        vals_a[v] = brackets[v][0]
                    indexes.append([i for i in range(len(brackets[v])) if vals_a[v] >= brackets[v][i]][-1] )
                        
                    
            index_a = sum([a * b for a, b in zip(indexes, multipliers)]) # global index for all values of current agent a
            
            if index_a >= self.n[type_]:
                print('index larger than total size of space, a = {a}, type_ = {type_}')
                return 0
            else:
                index.append(index_a)
            
        return index

    def render(self, mode = 'human', close = False):
        return 0
    
    def init_brackets(self):
        brackets = {}
        for typev in ['state','action']:
            if typev == 'state' and self.indicators['state'] == [None]:
                brackets[typev] = None
            else:
                brackets[typev] = []
                for s in range(len(self.indicators[typev])):
                    indstr = self.indicators[typev][s]
                    if self.discrete[typev][s] == 1:
                        brackets[typev].append([0])
                    elif indstr[-1] == 'f':
                        listf = self.list_factors[indstr[0:3]]
                        nbins = self.granularity['state'][s]
                        brackets[typev].append( [np.percentile(listf, 1/nbins * 100 * i) for i in range(nbins)] )

                    else:  
                        brackets[typev].append( [self.minval[typev][s] + (self.maxval[typev][s] - self.minval[typev][s])/self.granularity[typev][s] * i for i in range(self.granularity[typev][s])] )
        return brackets

     
    def listcombs_indextocomb(self, typev, index = None):
        if typev == 'state' and self.indicators['state'] == [None]:
            return [0]
        
        if typev == 'global_state':
            granularity = [self.n['state'] for _ in range(self.n_agents)]
        elif typev == 'global_action':
            granularity = [self.n['action'] for _ in range(self.n_agents)]
        else:
            granularity = self.granularity[typev]
        
        indicators  = np.zeros(len(granularity))
        ncombs      = 1
        for x in granularity: 
            ncombs  = ncombs * x
        combs       = copy.deepcopy(indicators)
        
        nind        = len(indicators) 
        ind         = nind - 1
        allcombs    = 0
        count       = 0
        loop        = 1
                    
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
                        ind       = nind - 1
            maxind = sum(1 for i in range(nind) if indicators[i] == granularity[i] - 1)
            if maxind == nind:
                allcombs = 1
            combs = np.vstack((combs, indicators))
            loop  = (allcombs == 0 and count < 1e6) if index is None else len(combs) < index + 1
            
            obj = combs if index is None else combs[-1]
        return obj
    
    
    def get_spaces(self, typeenv):
        space    = {}
        for s in ['state','action']: # s looping through state and action spaces
            # initialise space - discrete or continuous
            space[s] = spaces.Discrete(self.n[s]) if typeenv == 'discrete' else spaces.Box([0 for i in range(self.n[s])],[self.maxval[s][i] for i in range(self.n[s])])
        
        """
        A joint action is a tuple with each agent's actions.
        This property should be the list of all possible joint actions:
        """
        # self.possible_joint_actions = list(itertools.product(*[self.possible['action'] for _ in range(self.n_agents)]))
        
        return space

class Alpha_manager():
    """
    k  - line coefficients per interval
    xs - intervals
    dp - positive import, negative export (after fixed consumption)
    ds - change in storage
    l  - losses
    fl - flexible load consumed
    """
    
    def __init__(self, syst, learn_prm, plotting = False):
        self.name     = 'alpha manager'
        self.entries  = ['dp','ds','fl','l']
        self.plotting = learn_prm['plotting_alpha']
        self.server   = learn_prm['server']
        self.n_agents = syst['ntw']['n']
        self.bat = {}
        for e in ['cmax', 'dmax', 'mincharge', 'etach','cap','store0']: self.bat[e] = syst['bat'][e] 
        if plotting:
            self.labels  = [r'$\Delta$p',r'$\Delta$s', 'Flexible consumption', 'Losses']
            self.zorders = [1,3,2,0]
            self.colors  = [sns.xkcd_rgb['black'],sns.xkcd_rgb['cerulean'], sns.xkcd_rgb['greenish'], sns.xkcd_rgb['light red']]
        
    def initial_processing(self, discharge, charge, store, l_flex, g_net, EV_avail):
        # translate inputs into more relevant quantities
        cmax_eff                = max(0, self.bat['cmax'] - charge)          # remaining maximum charge
        dmax_eff                = max(0, self.bat['dmax'] - discharge)       # remaining maximum discharge
        s_avail                 = (store - self.mincharge) * EV_avail # storage available above minimum charge
        self.s_avail_dis        = min(s_avail, dmax_eff) * EV_avail          # storage available above minimum charge that can be discharged
        C_eff                   = (min(self.bat['cap'] - self.mincharge, s_avail + cmax_eff)) * EV_avail # Effective capacity in my storage - how much can be added above minimum level
        self.C_avail            = min((C_eff - s_avail) * EV_avail, cmax_eff)     # how much can I charge it by rel. to current level
        C_avail_after_prod      = (self.C_avail - self.bat['etach'] * g_net) * EV_avail       # how much can I charge it by rel. to current level after I have added my production
        
        # get helper variables and reference coefficients / intervals
        self.k, self.xs = {}, {} 
        A               = - (self.s_avail_dis + g_net) # helper variables
        B               = C_avail_after_prod/self.bat['etach'] if C_avail_after_prod/self.bat['etach'] > 0 else - (self.bat['etach'] * g_net - self.C_avail)/self.bat['etach']
        self.k['dp']    = [[l_flex + B - A, A]]   # reference line - dp
        self.xs['dp']   = [0, 1]  
        
    def obtain_coefficients(self, g_net, l_flex):
        C               = 0 if self.C_avail > self.bat['etach'] * g_net else - (self.bat['etach'] * g_net - self.C_avail)/(self.bat['etach'])
        D               = min(self.bat['etach'] * g_net, self.C_avail) # helper variables
        
        alpha           = {} # intermediate points between intervals
        alpha['cons']   = - (g_net + self.k['dp'][0][1])/self.k['dp'][0][0]
        alpha['prod']   = (- g_net + l_flex - self.k['dp'][0][1])/self.k['dp'][0][0]
        alpha['import'] = (l_flex + C - self.k['dp'][0][1])/self.k['dp'][0][0]
        
        xs_ = [0]
        for e in ['ds','l','fl']: self.k[e] = []
        
        if alpha['cons'] > 0: # between 0 and alpha_cons - playing with initial storage
            xs_.append(alpha['cons'])
            self.k['ds'].append([self.s_avail_dis/alpha['cons'], - self.s_avail_dis])
            self.k['l'].append([0, 0])
            self.k['fl'].append([0, 0])
        
        if alpha['prod'] > alpha['cons']: # up to alpha_prod - playing with flexible load consumption
            xs_.append(alpha['prod'])
            self.k['ds'].append([0,0])
            self.k['l'].append([0,0])
            afl2 = l_flex/(alpha['prod'] - alpha['cons'])
            self.k['fl'].append([afl2, -afl2 * alpha['cons']])
            
        if alpha['import'] > alpha['prod']: # up to alpha_import - playing with PV genertion output
            xs_.append(alpha['import'])
            ads3 = D/(alpha['import'] - max(0,alpha['prod'])) # helper variables
            al3  = (1 - self.bat['etach']) * (D/self.bat['etach']) / (alpha['import'] - max(0,alpha['prod']))
            self.k['ds'].append([ads3, - ads3 * max(0,alpha['prod'])])
            self.k['l'].append([al3, - al3 * max(0,alpha['prod'])])
            self.k['fl'].append([0,l_flex])
        
        if alpha['import'] < 1: # up to 1 - playing with additional imports
            xs_.append(1)
            ads4 = (self.C_avail - D)/(1 - alpha['import']) # helper variables
            al4  = ((1 - self.bat['etach'])/self.bat['etach'])  / (alpha['import'] - 1) * (D - self.C_avail)
            self.k['ds'].append([ads4, D - ads4 * alpha['import']])
            self.k['l'].append([al4, self.C_avail / self.bat['etach'] * (1 - self.bat['etach']) -  al4])
            self.k['fl'].append([0,l_flex])
        self.xs['ds'], self.xs['l'], self.xs['fl'] = [xs_ for _ in range(3)]
        
        self.alpha, self.xs_ = alpha, xs_
        
    def update_vars_from_alpha(self, alpha_action, g_net, store, charge, discharge, store_out_other, netp, losscharge, l_flex, EV_avail, last_step = False):
        self.mincharge = self.bat['store0'] if last_step else self.bat['mincharge']

        if g_net + EV_avail + l_flex == 0: # there is not flexibility
            bool_flex       = 0 # boolean for whether or not we have flexibility
            l_flex_consumed = 0 # no flexible load was consumed
        else:    
            bool_flex       = 1
            self.initial_processing(discharge, charge, store, l_flex, g_net, EV_avail)
            self.obtain_coefficients(g_net, l_flex)
            
            # updata variables for given alpha_action
            ik     = [i for i in range(len(self.xs_) - 1) if alpha_action >= self.xs_[i]][-1] # obtain the interval in which alpha lies
            res    = {} # resulting values (for dp, ds, fl, l)
            for e in self.entries:
                ik_    = 0 if e == 'dp' else ik
                res[e] = self.k[e][ik_][0] * alpha_action + self.k[e][ik_][1] # use coefficients to obtain value
            
            # update variables
            store           += res['ds']
            dch              = res['ds'] if res['ds'] > 0 else 0
            ddis             = - res['ds'] if res['ds'] < 0 else 0
            charge          += dch
            discharge       += ddis
            store_out_other += ddis                
            losscharge      += res['l']
            netp            += res['dp']
            l_flex_consumed  = res['fl']
            
            # check for errors 
            self.error = False
            # with EV availability
            if EV_avail == 0 and res['ds'] > 0:
                print('in update alpha action dch > 0, EVavail = 0')
                if not self.server: 
                    import playsound
                    playsound()
                self.error = True
                
            # with energy balance
            if abs( (res['dp'] + ddis - dch - res['l'] - res['fl']) + g_net ) > 1e-2:
                print('g_net = {}'.format(g_net))
                print('ik = {}'.format(ik))
                print("alpha['cons'] = {}, alpha['prod'] = {}, alpha['import'] = {}".format(self.alpha['cons'], self.alpha['prod'], self.alpha['import']))
                for e in self.entries:
                    print('e = {}, res[e] = {}'.format(e, res[e]))
                print('dp+ddis - dch - dloss - dcons = {} '.format(res['dp'] + ddis - dch - res['l'] - res['fl']))
                print('dp = {}, ddis = {}, dch = {}, dloss = {}, dcons = {}'.format(res['dp'], ddis, dch, res['l'], res['fl']))    
                print('alpha_action = {}'.format(alpha_action))
                self.error = True
            
            if not self.error and self.plotting: self.plot_graph_alpha()
            
        return store, netp, losscharge, charge, discharge, store_out_other, l_flex_consumed, bool_flex
    
    def plot_graph_alpha(self):
        sns.set_palette("bright")
        import matplotlib.pyplot as plt
        ymin = 0
        ymax = 0
        plt.figure()
        for i in range(len(self.k['l'])):
            e          = self.entries[i]
            wd         = 3 if e == 'dp' else 1.5
            col, zo, l = self.colors[i], self.zorders[i], self.labels[i]
            n          = len(self.k[e])
            ys         = [self.k[e][i][0] * self.xs[e][i] + self.k[e][i][1] for i in range(n)]
            ys.append(self.k[e][-1][0] * self.xs[e][-1] + self.k[e][-1][1])
            plt.plot(self.xs[e], ys, label = l, linewidth = wd, color = col, zorder = zo)
            if min(ys) < ymin: ymin = min(ys)
            if max(ys) > ymax: ymax = max(ys)
        
        alpha = self.alpha
        for e in alpha.keys():
            plt.vlines(x = alpha[e], ymin = ymin + (ymax - ymin) * 0.1, ymax = ymax, linestyle = '--', linewidth=1)
            plt.text(alpha[e], ymin, r'$\alpha_{{%s}}$' % e, ha ='center', va = 'center',  fontsize = 15)
        plt.xlabel('Action variable '+ r'$\alpha$ [-]')
        plt.ylabel('[kWh]')
        props = dict(boxstyle = 'round', facecolor = [0.9, 0.9, 0.9], alpha = 0.7)
        plt.text((0 + alpha['cons'])/2, ymax * 0.9, "(1) Initial \nstorage", fontsize = 10, bbox = props, ha='center', va = 'center')
        plt.text((alpha['prod'] + alpha['cons'])/2, ymax * 0.9, "(2) Flexible \nload", fontsize = 10, bbox = props, ha = 'center', va = 'center')
        plt.text((alpha['prod'] + alpha['import'])/2, ymax * 0.9, '(3) PV \nGeneration', fontsize = 10, bbox = props, ha = 'center', va = 'center')
        plt.text((1 + alpha['import'])/2, ymax * 0.9, '(4) Import \n ', fontsize = 10, bbox = props, ha = 'center', va = 'center')
        plt.legend(fontsize = 10, ncol = 4, loc = 'upper left', bbox_to_anchor = (-0.03, 1.15))
        plt.grid()
    
    def dp_to_alpha(self, discharge, charge, store, l_flex, g_net, dp, EV_avail, last_step = None):
        if last_step is not None: self.mincharge = self.bat['store0'] if last_step else self.bat['mincharge']

        if g_net + EV_avail + l_flex == 0: # there is not flexibility
            bool_flex       = 0 # boolean for whether or not we have flexibility
            alpha_action    = None
        else:    
            bool_flex    = 1
            self.initial_processing(discharge, charge, store, l_flex, g_net, EV_avail)
            alpha_action = (dp - self.k['dp'][0][1])/self.k['dp'][0][0]
        
        return bool_flex, alpha_action
    
    def analyse_alphaneg(self, res, batch, inputs, updatecons_inputs, updatecons_outputs, dem_a_t):
        # unpack inputs
        prod0, store0, fixed_cons_prob_a, EVcons, EVavail                      = updatecons_inputs[0:5] # before fixed cons
        prod2, store2, charge2, losscharge2, imp2, discharge2, store_out_other = updatecons_outputs     # after fixed cons
        flex_load_prob_a, dp, t                                                = inputs
        
        a, b     = 0, 0
        t        = int(t)
        
        # according to batch
        # flex cons
        batch_flex     = [batch[a]['flex'] for a in range(self.n_agents)]
        lds_flex_batch = [sum(batch_flex[a][t][1:]) for a in range(self.n_agents)]
        
        # fixed cons
        fixed_cons_batch = [batch_flex[a][t][0] for a in range(self.n_agents)]
        
        # check this is the same information in problem
        lds_flex_prob   = self.syst['ntw']['dem'][1, b, t] # l = 1 for flex
        fixed_cons_prob = self.syst['ntw']['dem'][0, b, t] # l = 0 for fixed
        
        dmax_eff        = max(0, self.bat['dmax'] + charge2 - discharge2)
        s_avail         = store2 - self.mincharge
        s_avail_dis     = min(s_avail, dmax_eff)
        mindp           = - ( s_avail_dis + prod2 )
        alpha_action    = self.dp_to_alpha(discharge2, charge2, store2, flex_load_prob_a, prod2, dp, EVavail)
        alpha_action0   = 0
        store3, netp3, losscharge3, charge3, discharge3, store_out_other3, l_flex_consumed3, bool_flex = self.update_vars_from_alpha(alpha_action0, prod2, store2, charge2, discharge2, store_out_other, imp2, losscharge2, lds_flex_prob, EVavail) # after alpha = 0
        print('bool_flex = {}'.format(bool_flex))
        print('EVavail = {}'.format(EVavail))
        print('--- to meet fixed cons and EV cons ---')
        print('fixed cons {} + EV load {} = {}'.format(fixed_cons_prob_a, EVcons, fixed_cons_prob_a + EVcons ))
        print('ds (store0 - store2) {} + dg (prod0 - prod2) {} + imp2 {} + losscharge2 {}= {}'.format(store0 - store2, prod0 - prod2, imp2, losscharge2, store0 - store2+ prod0 - prod2 + imp2+ losscharge2))
        print('--- then existing flexibility is ---')
        print('remaining prod prod2 {}, remaining storage (store2-mincharge){}, l_flex = {}'.format(prod2, store2 - self.mincharge, flex_load_prob_a))
        print('----- for alpha_action = 0 we would have the additional changes -----')
        print('ds (store3-store2) = {}, dp (imp3-imp2) = {}, dloss (loss3-loss2) {}, lflexcons={}'.format(store3-store2, netp3-imp2,losscharge3-losscharge2, l_flex_consumed3))
        print("res['netp'] - imp2 = {}".format(res['netp'][a,t] - imp2))
        print('----thus in the step the totals are...---')
        print('ds (store3-store0) {}, netp (imp3){}, losscharge3 {}, totcons {}'.format(store3-store0, netp3, losscharge3, l_flex_consumed3 +fixed_cons_prob ))
        print('---- in the problem netp {} is accounted for by ---'.format(res['netp'][a][t]))
        print("store_in {}, losscharge {}, store_out_other {}, ntw['gen'] {}, curt {}, totcons{}, SUM {}".format(res['store_in'][a][t], res['losscharge'][a][t], res['store_out_other'][a][t], self.syst['ntw']['gen'][a][t], res['curt'][a][t],  res['totcons'][a][t] ,- res['store_in'][a][t] - res['losscharge'][a][t] + res['store_out_other'][a][t] + self.syst['ntw']['gen'][a][t] - res['curt'][a][t] - res['totcons'][a][t]  ))
    
        
            
            