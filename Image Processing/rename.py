# -*- coding: utf-8 -*-
import os

path = ""
for dirpath, dirname, filename in os.walk('.'):
    i = 0
    for item in os.listdir(dirpath):
        if '.jpg' in item:
            newname = str(i) * 5 + '.jpg'
            os.renames(dirpath + '/' + item, dirpath + '/' + newname)
            i = i+1

for dirpath, dirname, filename in os.walk('.'):
    i = 0
    for item in os.listdir(dirpath):
        if '.jpg' in item:
            newname = str(i) + '.jpg'
            os.renames(dirpath + '/' + item, dirpath + '/' + newname)
            i = i+1