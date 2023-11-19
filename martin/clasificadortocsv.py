#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:49:21 2022

@author: mroitegui
"""


import pandas as pd
import re

formula=pd.read_csv("/work/mroitegui/Superconductors/data/12340_all_pred.csv")
superconductors_list=formula['DOPPED'].tolist()
superconductorsdict=dict(zip(formula['DOPPED'], formula['TC']))
# working_file=pd.read_csv('/work/mroitegui/Superconductors/data/train.csv')
# working_file.head()

superpuro=[]
cupratos=[]
oxidoytrio=[]
oxidobario=[]
boruromagnesio=[]
ironbase=[]
aleaciones=[]
puroname =[]
cupratoname=[]
oxidoytrioname=[]
oxidobarioname=[]
boruromagnesioname=[]
ironbasename=[]
aleacionesname=[]
lowT=[]
lowTname=[]
highTname=[]
highT=[]
for superconductor in superconductors_list:
    prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes 
    lista_elementos=[]
   
    for x in prueba:
        elementos=re.sub('[^a-zA-Z]','',x) 
        lista_elementos.append(elementos)
        
    # if len(lista_elementos) == 1:
    #     superpuro.append(superconductors_list.index(superconductor))
    #     puroname.append(superconductor)
    if 'Cu'  in lista_elementos and "O" in lista_elementos:
        cupratos.append(superconductors_list.index(superconductor)) 
        cupratoname.append(superconductor)       
        if superconductorsdict[superconductor] <= 40:
                
                lowT.append(superconductors_list.index(superconductor))
                lowTname.append(superconductor) 
        else:
                highT.append(superconductors_list.index(superconductor))
                highTname.append(superconductor)
                
    elif 'Fe' in lista_elementos:
        ironbase.append(superconductors_list.index(superconductor))
        ironbasename.append(superconductor)
    # elif 'Y' and 'O' in lista_elementos:
    #       oxidoytrio.append(superconductors_list.index(superconductor))
    #       oxidoytrioname.append(superconductor)
    # # elif 'Ba' and 'O' in lista_elementos:
    # #     oxidobario.append(superconductors_list.index(superconductor))
    # #     oxidobarioname.append(superconductor)
    # # elif 'B' and 'Mg' in lista_elementos:
    # #     boruromagnesio.append(superconductors_list.index(superconductor))
    # #     boruromagnesioname.append(superconductor)
    # else:
    #     aleaciones.append(superconductors_list.index(superconductor)) 
    #     aleacionesname.append(superconductor)
# for i in superconductorsdict:
#     if superconductorsdict[i] <= 40:
#         lowT.append(superconductors_list.index(i))
#         lowTname.append(i) 
cupratoscsv = formula.loc[cupratos] 
cupratoscsv['DOPPED']=cupratoname
cupratoscsv.to_csv('cupratos.csv', encoding='utf-8', index=False)

ironbasecsv = formula.loc[ironbase] 
ironbasecsv['DOPPED']=ironbasename
ironbasecsv.to_csv('ironbase.csv', encoding='utf-8', index=False)

lowTcsv = formula.loc[lowT] 
lowTcsv['DOPPED']=lowTname
lowTcsv.to_csv('lowT.csv', encoding='utf-8', index=False)

highTcsv = formula.loc[highT] 
highTcsv['DOPPED']=highTname
highTcsv.to_csv('highT.csv', encoding='utf-8', index=False)

   



     
        