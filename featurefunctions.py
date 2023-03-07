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

electronegativitycsv =pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
noble_gases = {'He': '1s2',
 'Ne':  '1s2  2s2 2p6',
 'Ar': ' 1s2  2s2 2p6 3s2 3p6',
 'Kr': '1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6', 
 'Xe' :'1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6',
 'Rn' :'1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p6'}    

def electrodifference(formula,electrones,superconductors_list):

   
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

def mval(formula,electrones,superconductors_list):

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

def mrad(formyula,electrones,superconductors_list):

    
    rad_dict = dict(zip(electrones['Elemento'],electrones['AtomicRadius']))   
    mrad=[]    
    for superconductor in superconductors_list:                   
            atomostotal=0
            radtot=0
            prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
            for x in prueba:
                        if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                                x=x+'1'
                        elementos=re.sub('[^a-zA-Z]','',x)                   
                        rad=(rad_dict[elementos])
                        cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                        cantidad_de_atomos=float(cantidad_de_atomosstr)
                        atomostotal+=cantidad_de_atomos    
                        cantidadrad=cantidad_de_atomos*rad
                        radtot+=cantidadrad                                                                 
            mrad1=radtot/atomostotal
            mrad.append(mrad1)
    mrad=np.array(mrad)
    mrad = mrad.reshape(-1,1)
    return (mrad)   
def mfie(formula,electrones,superconductors_list):

     
    fie_dict = dict(zip(electrones['Elemento'],electrones['FIE']))   
    mfie=[]    
    for superconductor in superconductors_list:                   
            atomostotal=0
            fietot=0
            prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
            for x in prueba:
                        if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                                x=x+'1'
                        elementos=re.sub('[^a-zA-Z]','',x)                   
                        fie=(fie_dict[elementos])
                        cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                        cantidad_de_atomos=float(cantidad_de_atomosstr)
                        atomostotal+=cantidad_de_atomos    
                        cantidadfie=cantidad_de_atomos*fie
                        fietot+=cantidadfie                                                                 
            mfie1=fietot/atomostotal
            mfie.append(mfie1)
    mfie=np.array(mfie)
    mfie = mfie.reshape(-1,1)
    return (mfie)    

def mtc(formula,electrones,superconductors_list):

    
    tc_dict = dict(zip(electrones['Elemento'],electrones['TC']))   
    mtc=[]    
    for superconductor in superconductors_list:                   
            atomostotal=0
            tctot=0
            prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
            for x in prueba:
                        if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                                x=x+'1'
                        elementos=re.sub('[^a-zA-Z]','',x)                   
                        tc=(tc_dict[elementos])
                        cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                        cantidad_de_atomos=float(cantidad_de_atomosstr)
                        atomostotal+=cantidad_de_atomos    
                        cantidadtc=cantidad_de_atomos*tc
                        tctot+=cantidadtc                                                                 
            mtc1=tctot/atomostotal
            mtc.append(mtc1)
    mtc=np.array(mtc)
    mtc = mtc.reshape(-1,1)
    return (mtc)    
 
def mec(formula,electrones,superconductors_list):

        
        ec_dict = dict(zip(electrones['Elemento'],electrones['EC']))   
        mec=[]    
        for superconductor in superconductors_list:                   
                atomostotal=0
                ectot=0
                prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
                for x in prueba:
                            if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                                    x=x+'1'
                            elementos=re.sub('[^a-zA-Z]','',x)                   
                            ec=float(ec_dict[elementos])
                          
                            cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                            cantidad_de_atomos=float(cantidad_de_atomosstr)
                            atomostotal+=cantidad_de_atomos    
                            cantidadec=cantidad_de_atomos*ec
                            ectot+=cantidadec                                                                 
                mec1=ectot/atomostotal
                mec.append(mec1)
        mec=np.array(mec)
        mec = mec.reshape(-1,1)
        return (mec)    
    
    
def eletronicdensity(formula, electrones,superconductors_list): 
    
    
     electroness_dict=dict(zip(electrones['Elemento'],electrones['s']))
     electronesp_dict=dict(zip(electrones['Elemento'],electrones['p']))
     electronesd_dict=dict(zip(electrones['Elemento'],electrones['d']))
     electronesf_dict=dict(zip(electrones['Elemento'],electrones['f']))
     densidadslist=[]
     densidadplist=[]
     densidaddlist=[]
     densidadflist=[]
     electronesslist=[]
     electronesplist=[]
     electronesdlist=[]
     electronesflist=[]
     for superconductor in superconductors_list:
             electronesstotal=0
             electronesptotal=0
             electronesdtotal=0
             electronesftotal=0        
             atomostotal=0        
             prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
             for x in prueba:
                         if x.isalpha():#Suma un uno al final del elemento en ca21285so de que el numero de atomos no aparezca en la tabla 
                                 x=x+'1'
                         elementos=re.sub('[^a-zA-Z]','',x)
                         numeroelectroness=(electroness_dict[elementos])
                         numeroelectronesp=(electronesp_dict[elementos])
                         numeroelectronesd=(electronesd_dict[elementos])
                         numeroelectronesf=(electronesf_dict[elementos])
                         cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                         cantidad_de_atomos=float(cantidad_de_atomosstr)            
                         cantidadelectroness=cantidad_de_atomos*numeroelectroness
                         cantidadelectronesp=cantidad_de_atomos*numeroelectronesp
                         cantidadelectronesd=cantidad_de_atomos*numeroelectronesd
                         cantidadelectronesf=cantidad_de_atomos*numeroelectronesf                   
                         electronesstotal+=cantidadelectroness
                         electronesptotal+=cantidadelectronesp
                         electronesdtotal+=cantidadelectronesd
                         electronesftotal+=cantidadelectronesf                                 
                         atomostotal+=cantidad_de_atomos        
             densidads=(electronesstotal)/atomostotal
             densidadslist.append(densidads)
             densidadp=(electronesptotal)/atomostotal
             densidadplist.append(densidadp)
             densidadd=(electronesdtotal)/atomostotal
             densidaddlist.append(densidadd)
             densidadf=(electronesftotal)/atomostotal
             densidadflist.append(densidadf)
             electronesslist.append(numeroelectroness)
             electronesplist.append(numeroelectroness)
             electronesdlist.append(numeroelectroness)
             electronesflist.append(numeroelectroness)
     densidadslist = np.array(densidadslist)
     densidadslist = densidadslist.reshape(-1,1)
     densidadplist = np.array(densidadplist)
     densidadplist = densidadplist.reshape(-1,1)
     densidaddlist = np.array(densidaddlist)
     densidaddlist = densidaddlist.reshape(-1,1)
     densidadflist = np.array(densidadflist)
     densidadflist = densidadflist.reshape(-1,1)
     return(densidadslist,densidadplist,densidaddlist,densidadflist)
         
def mend(formula,electrones,superconductors_list):
    mend_dict=dict(zip(electrones['elementmend'],electrones['mendnumber']))
    mendeleievlist=[]
    for superconductor in superconductors_list:           
            atomostotal=0
            mendtotal=0           
            prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
            for x in prueba:
                        if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                                x=x+'1'
                        elementos=re.sub('[^a-zA-Z]','',x)                       
                        numeromend=(mend_dict[elementos])                      
                        cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                        cantidad_de_atomos=float(cantidad_de_atomosstr)                                            
                        cantidadmend=cantidad_de_atomos*numeromend                       
                        atomostotal+=cantidad_de_atomos
                        mendtotal+=cantidadmend                                           
            avmend=mendtotal/atomostotal
            mendeleievlist.append(avmend)             
    mendeleievlist = np.array(mendeleievlist)
    mendeleievlist=mendeleievlist.reshape(-1,1)
    return(mendeleievlist)

def masa(formula,electrones,superconductors_list):
    
    masa_dict=dict(zip(electrones['Elemento'],electrones['Masa']))  
    masalist=[]   
    for superconductor in superconductors_list:         
            atomostotal=0           
            masa=0
            prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
            for x in prueba:
                        if x.isalpha():#Suma un uno al final del elemento en ca21285so de que el numero de atomos no aparezca en la tabla 
                                x=x+'1'
                        elementos=re.sub('[^a-zA-Z]','',x)                      
                        numeromasa=float(masa_dict[elementos])                
                        cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                        cantidad_de_atomos=float(cantidad_de_atomosstr)
                              
                        masatotal=cantidad_de_atomos*numeromasa
                        masa+=masatotal
                        atomostotal+=cantidad_de_atomos                       
            masalist.append(masa)          
    masalist=np.array(masalist)
    masalist=masalist.reshape(-1,1)    
    return(masalist)


 

def compute_vecs(formula, electrones,superconductors_list):
   
    e_config=dict(zip( electrones['Elemento'], electrones['Conf_electronica'] ))
    e_conf=pd.DataFrame({'element':e_config.keys(),'config':e_config.values()})
    e_conf['config']=e_conf['config'].replace(noble_gases, regex=True) 
    element=electrones['Elemento'].tolist()
    matrixvalues=[]
    
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
    levels = np.zeros(shape=(117,7,4))  
    vecss = np.zeros(shape=(np.size(superconductors_list),7,4))
    vecs = np.zeros(shape=(np.size(superconductors_list),28))
    # print(matrixvalues)
    for i in range(117):
        for j in range(7):
            size = np.array(np.shape(matrixvalues[i][j][1]))[0]
            for k in range(size):
                levels[i,j,k] = float(matrixvalues[i][j][1][k])
                    
    count = 0
    for superconductor in superconductors_list:
        # print(superconductor)
        atomostotal=0 
        prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus componentes
        sumorbs = 0
        for x in prueba:
            if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                x=x+'1'
            elementos=re.sub('[^a-zA-Z]','',x)
            # print(elementos)
            cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
            cantidad_de_atomos=float(cantidad_de_atomosstr)
            # print(cantidad_de_atomos)                   
            sumorbs += levels[element.index(elementos)]*cantidad_de_atomos
            # print(elementos)
            # print(sumorbs)
            # print(cantidad_de_atomos)
            atomostotal+=cantidad_de_atomos
        # print(atomostotal)
        # print(sumorbs)
        vecss[count] = sumorbs/atomostotal
        vecs[count] = vecss[count].ravel()
        count += 1
    return vecs

def EN_old(formula,electrones,superconductors_list):   
    electronegativity_dict = dict(
        zip(electronegativitycsv["Symbol"], electronegativitycsv["Electronegativity"]))    
    electronegativity_list = []   
    for superconductor in superconductors_list:     
        electronegatividadtotal = 0
        atomostotal = 0        
        prueba = re.findall(
            "([A-Z][^A-Z]*)", superconductor
        )  # dividimos el superconductor en sus compoentes
        for x in prueba:
            if (
                x.isalpha()
            ):  # Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla
                x = x + "1"
            elementos = re.sub("[^a-zA-Z]", "", x)           
            electronegatividad = electronegativity_dict[elementos]
            cantidad_de_atomosstr = re.sub("[a-zA-Z]", "", x)
            cantidad_de_atomos = float(cantidad_de_atomosstr)           
            cantidadelectronegatividad = cantidad_de_atomos * electronegatividad          
            atomostotal += cantidad_de_atomos        
            electronegatividadtotal += cantidadelectronegatividad        
        mean_electronegativity = electronegatividadtotal / atomostotal
        electronegativity_list.append(mean_electronegativity)          
    electronegativity_list = np.array(electronegativity_list)
    electronegativity_list = electronegativity_list.reshape(-1, 1)
    return(electronegativity_list)
def shap_mean(values,n):
    i=0
    suma=0
    for value in values:
        suma+=value[:n]
        i=i+1
    mean=suma/i
    return(mean)    