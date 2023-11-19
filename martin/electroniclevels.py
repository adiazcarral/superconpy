#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:15:01 2022

@author: mroitegui
"""
# import numpy as np
import pandas as pd
import re
import numpy as np
formula=pd.read_csv("/work/mroitegui/Superconductors/data/unique_m.csv")
superconductors_list=formula['material'].tolist()
#declare nobel gases
noble_gases = {'He': '1s2',
'Ne':  '1s2  2s2 2p6',
'Ar': ' 1s2  2s2 2p6 3s2 3p6',
'Kr': '1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6', 
'Xe' :'1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6',
'Rn' :'1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p6'}

electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elementswithelectronstotal.csv") 
e_config=dict(zip( electrones['Elemento'], electrones['Conf_electronica'] ))
e_conf=pd.DataFrame({'element':e_config.keys(),'config':e_config.values()})
e_conf['config']=e_conf['config'].replace(noble_gases, regex=True) 
element=electrones['Elemento'].tolist()
matrixvalues=[]
mydict={}
# mydict = np.ndarray(shape=(7,4,118))
for e in e_conf['config']:
 
    
    # print(Elemento)
    elementconf={
    'n1':[],
    'n2':[],
    'n3':[],
    'n4':[],
    'n5':[],
    'n6':[],
    'n7':[]}
    a=e.split()  
        
    for x in a:      
            lel=x[2:]
            n=x[0]
            if n == '1' :          
                elementconf['n1'].append(lel)
            elif n=='2':           
                elementconf['n2'].append(lel)
            elif n=='3':          
                elementconf['n3'].append(lel)
            elif n=='4':           
                elementconf['n4'].append(lel)
            elif n=='5':      
                elementconf['n5'].append(lel)
            elif n=='6':    
                elementconf['n6'].append(lel)
            elif n=='7':          
                elementconf['n7'].append(lel)              
    y=np.array(list(elementconf.items()),dtype=object)                 
    
        
    matrixvalues.append(y)
mydict = dict(zip(element, matrixvalues))
        
    
#         # y=y.reshape(7,4)
#         # print(y)
#         # y=y.rename(columns={0:'S',1:'P',2:'D',3:'F'})
# mydict=mydict.reshape(7,4)       
#         # mydict[:,:,z] = y
        # z=z+1
matriztotal=[]
for superconductor in superconductors_list:
        # print(superconductor)
        atomostotal=0 
        prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus componentes
        scmatrix= []        
        for x in prueba:
                    if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                            x=x+'1'
                    elementos=re.sub('[^a-zA-Z]','',x)
                    cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                    cantidad_de_atomos=float(cantidad_de_atomosstr)
                    atomostotal+=cantidad_de_atomos
                    
                    # elementmatrix=mydict[elementos].to_numpy()
        #             elementmatrixtot=mydict[elementos]
        #             elementmatrix1=elementmatrixtot[:,1]
                    
        #             elementmatrix2=elementmatrix1*cantidad_de_atomos
                    
        # #             scmatrix.append(elementmatrix)
        # #             atomostotal+=cantidad_de_atomos
        # # # print(scmatrix)            
        # summatrix=(sum(scmatrix))
        # # print(summatrix)
        # # print(atomostotal)
        # levelmatrix=summatrix/atomostotal 
        # # print(levelmatrix)
        # matriztotal.append(levelmatrix)
        # # print(matriztotal)
 
        


    

 

