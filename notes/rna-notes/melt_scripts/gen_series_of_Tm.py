#!/usr/bin/env python

#line =  'GCGCGCGCGCGCGCGCGCGCGCGC'
line = 'AUAUAUAUAUAUAUAUAUAUAUAUAU'
import os
for i in range(5,21):
  print i 
  command = './Rmelttemp.py %s | tail -n 1 '  % line[0:i]
  os.system(command)
  
