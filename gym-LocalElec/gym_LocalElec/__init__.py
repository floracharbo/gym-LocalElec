#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:29:24 2020

@author: floracharbonnier
"""

from gym.envs.registration import register

register(
    id='LocalElec-v0',
    entry_point='gym_LocalElec.envs:LocalElecEnv',
)
