#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:24:45 2023

@author: mroitegui
"""

import pandas as pd
import re
formula=pd.read_csv("/work/mroitegui/Superconductors/rellenotabla.csv")
superconductors_list=formula['material'].tolist()
index=index=0
for superconductor in superconductors_list:                   
            prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
           
            for x in prueba:                
                        if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                                x=x+'1'
                        elementos=re.sub('[^a-zA-Z]','',x)                        
                        cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                        formula.loc[[index],[elementos]]=cantidad_de_atomosstr
            index=index+1
formula.to_csv("/work/mroitegui/Superconductors/rellenotabla.csv")                          
                        