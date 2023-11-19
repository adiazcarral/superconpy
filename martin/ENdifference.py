#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:19:50 2022

@author: mroitegui
"""
# import numpy as np
# def ENdifference():   
import pandas as pd
import re
import numpy as np

def electrodifference():

    formula=pd.read_csv("/work/angel/Superconductivity/datasets/archive/unique_m.csv")
    superconductors_list=formula['material'].tolist()
    nonmetal = ['C','N','O','S','P','Se','He','Ne','Ar','Kr','Xe','Rn','F','Cl','Br','I','At','H']
    electronegativitycsv =pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
    electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elementswithelectronstotal.csv") 
    electronegativity_dict = dict(zip(electronegativitycsv['Symbol'],electronegativitycsv['Electronegativity']))
    element=electrones['Elemento'].tolist()
    EN=[]
    metal=[]
    for elemento in element:
        if elemento not in nonmetal:
            metal.append(elemento)
    for superconductor in superconductors_list:                   
            atomostotal=0
            electronegatividadmetal=0
            electronegatividadnm=0
            prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
            for x in prueba:
                        if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                                x=x+'1'
                        elementos=re.sub('[^a-zA-Z]','',x)
                        if elementos in metal:
                            electronegatividadmetal1=(electronegativity_dict[elementos])
                            cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                            cantidad_de_atomos=float(cantidad_de_atomosstr)
                            atomostotal+=cantidad_de_atomos    
                            cantidadelectronegatividad=cantidad_de_atomos*electronegatividadmetal1 
                            electronegatividadmetal+=cantidadelectronegatividad
                        else:    
                            electronegatividadnm1=(electronegativity_dict[elementos])
                            cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                            cantidad_de_atomos=float(cantidad_de_atomosstr)
                            cantidadelectronegatividadnm=cantidad_de_atomos*electronegatividadnm1                                                     
                            atomostotal+=cantidad_de_atomos                  
                            electronegatividadnm+=cantidadelectronegatividadnm      
            electronegatividadsup=electronegatividadnm-electronegatividadmetal
            EN.append(electronegatividadsup)
    EN=np.array(EN)
    EN = EN.reshape(-1,1)
    return (EN)    

def mval():

    formula=pd.read_csv("/work/angel/Superconductivity/datasets/archive/unique_m.csv")
    superconductors_list=formula['material'].tolist()      
    electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elementswithelectronstotal.csv") 
    val_dict = dict(zip(electrones['Elemento'],electrones['val']))   
    mval=[]    
    for superconductor in superconductors_list:                   
            atomostotal=0
            valelectrons=0
            prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
            for x in prueba:
                        if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                                x=x+'1'
                        elementos=re.sub('[^a-zA-Z]','',x)                   
                        valelec=(val_dict[elementos])
                        cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                        cantidad_de_atomos=float(cantidad_de_atomosstr)
                        atomostotal+=cantidad_de_atomos    
                        cantidadval=cantidad_de_atomos*valelec
                        valelectrons+=cantidadval                                                                  
            mval1=valelectrons/atomostotal
            mval.append(mval1)
    mval=np.array(mval)
    mval = mval.reshape(-1,1)
    return (mval)    
