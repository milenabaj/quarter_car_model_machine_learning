#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:18:38 2020

@author: milena
"""


import logging
l = logging.getLogger('Main')
a = logging.getLogger('A')

l.info('main')
a.info('a')
 # Get the file handler from the root logger and add it to the module logger
for h in list(l.handlers):
    #if isinstance(h,logging.FileHandler):
    a.addHandler(h)