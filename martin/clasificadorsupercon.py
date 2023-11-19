#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:16:23 2022

@author: mroitegui
"""

import numpy as np
import pandas as pd
import re

formula=pd.read_csv("data/12340_all_pred.csv")
superconductors_list=formula['DOPPED'].tolist()
# working_file=pd.read_csv('/work/mroitegui/Superconductors/data/train.csv')

superpuro=[]
cupratos=[]
oxidoytrio=[]
oxidobario=[]
boruromagnesio=[]
ironbase=[]
aleaciones=[]
for superconductor in superconductors_list:
    prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes 
    lista_elementos=[]
    for x in prueba:
        elementos=re.sub('[^a-zA-Z]','',x) 
        lista_elementos.append(elementos)
        # print(lista_elementos)
    if len(lista_elementos) == 1:
        superpuro.append(superconductor)
    elif 'Cu'  in lista_elementos and "O" in lista_elementos:
        cupratos.append(superconductor)        
    # elif 'Fe' in lista_elementos:
    #     ironbase.append(superconductor)
    # elif 'Y' and 'O' in lista_elementos:
    #      oxidoytrio.append(superconductor)
    # elif 'Ba' and 'O' in lista_elementos:
    #     oxidobario.append(superconductor)
    # elif 'B' and 'Mg' in lista_elementos:
    #     boruromagnesio.append(superconductor)
    else:
        aleaciones.append(superconductor)    
         
        
        
            




'''
formula=pd.read_csv("/work/mroitegui/Superconductors/data/unique_m.csv")
electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
superconductors_list=formula['material'].tolist()
electroness_dict=dict(zip(electrones['Elemento'],electrones['electroness']))
electronesp_dict=dict(zip(electrones['Elemento'],electrones['electronesp']))
densidadslist=[]
densidadplist=[]

for superconductor in superconductors_list:
        electronesstotal=0
        electronesptotal=0
        atomostotal=0    
        prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
        for x in prueba:               
                    if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                            x=x+'1'
                    elementos=re.sub('[^a-zA-Z]','',x)
                    numeroelectroness=(electroness_dict[elementos])
                    numeroelectronesp=(electronesp_dict[elementos])
                    cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)               
                    cantidad_de_atomos=float(cantidad_de_atomosstr)
                    cantidadelectroness=cantidad_de_atomos*numeroelectroness
                    cantidadelectronesp=cantidad_de_atomos*numeroelectronesp
                    electronesstotal+=cantidadelectroness
                    electronesptotal+=cantidadelectronesp
                    atomostotal+=cantidad_de_atomos
        densidads=(electronesstotal)/atomostotal   
        densidadslist.append(densidads)
        densidadp=(electronesptotal)/atomostotal   
        densidadplist.append(densidadp)
        
density_feature0 = np.array([densidadplist,densidadslist])
density_feature = np.rot90(density_feature0)
'''