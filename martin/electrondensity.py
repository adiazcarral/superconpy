import numpy as np
import pandas as pd
import re

formula=pd.read_csv("/work/mroitegui/Superconductors/data/unique_m.csv")
# electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elementswithelectronstotal.csv")
electronegativitycsv =pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")

superconductors_list=formula['material'].tolist()

electroness_dict=dict(zip(electrones['Elemento'],electrones['s']))
electronesp_dict=dict(zip(electrones['Elemento'],electrones['p']))
electronesd_dict=dict(zip(electrones['Elemento'],electrones['d']))
electronesf_dict=dict(zip(electrones['Elemento'],electrones['f']))
structure_dict=dict(zip(electrones['Elemento'],electrones['Structurenumber']))
electronegativity_dict = dict(zip(electronegativitycsv['Symbol'],electronegativitycsv['Electronegativity']))
mend_dict=dict(zip(electrones['elementmend'],electrones['mendnumber']))
densidadslist=[]
densidadplist=[]
densidaddlist=[]
densidadflist=[]
electronesslist=[]
electronesplist=[]
electronesdlist=[]
electronesflist=[]
electronegativity_list=[]
mendeleievlist=[]
structure_list=[]
entropy_atvec=np.ndarray(shape=(21263,1))
psd_atvec=np.ndarray(shape=(21263,1))
sumatvec=np.ndarray(shape=(21263,1))


for superconductor in superconductors_list:
        electronesstotal=0
        electronesptotal=0
        electronesdtotal=0
        electronesftotal=0
        atomostotal=0 
        mendtotal=0
        structure=0
        electronegatividadtotal=0
        prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
        for x in prueba:
                    if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                            x=x+'1'
                    elementos=re.sub('[^a-zA-Z]','',x)
                    numeroelectroness=(electroness_dict[elementos])
                    numeroelectronesp=(electronesp_dict[elementos])
                    numeroelectronesd=(electronesd_dict[elementos])
                    numeroelectronesf=(electronesf_dict[elementos])
                    numeromend=(mend_dict[elementos])
                    numerostruc=(structure_dict[elementos])
                    electronegatividad=(electronegativity_dict[elementos])
                    cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                    cantidad_de_atomos=float(cantidad_de_atomosstr)
                    cantidadelectroness=cantidad_de_atomos*numeroelectroness
                    cantidadelectronegatividad=cantidad_de_atomos*electronegatividad
                    cantidadelectronesp=cantidad_de_atomos*numeroelectronesp
                    cantidadelectronesd=cantidad_de_atomos*numeroelectronesd
                    cantidadelectronesf=cantidad_de_atomos*numeroelectronesf
                    cantidadmend=cantidad_de_atomos*numeromend
                    electronesstotal+=cantidadelectroness
                    electronesptotal+=cantidadelectronesp
                    electronesdtotal+=cantidadelectronesd
                    electronesftotal+=cantidadelectronesf
                    atomostotal+=cantidad_de_atomos
                    mendtotal+=cantidadmend
                    electronegatividadtotal+=cantidadelectronegatividad
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
        mean_electronegativity=electronegatividadtotal/atomostotal
        electronegativity_list.append(mean_electronegativity)
        avmend=mendtotal/atomostotal
        mendeleievlist.append(avmend)

densidadslist = np.array(densidadslist)
densidadslist = densidadslist.reshape(-1,1)
densidadplist = np.array(densidadplist)
densidadplist = densidadplist.reshape(-1,1)
densidaddlist = np.array(densidaddlist)
densidaddlist = densidaddlist.reshape(-1,1)
densidadflist = np.array(densidadflist)
densidadflist = densidadflist.reshape(-1,1)
electronesslist = np.array(electronesslist)
electronesslist = electronesslist.reshape(-1,1)
electronesplist = np.array(electronesplist)
electronesplist = electronesplist.reshape(-1,1)
electronesdlist = np.array(electronesdlist)
electronesdlist = electronesdlist.reshape(-1,1)
electronesflist = np.array(electronesflist)
electronesflist = electronesflist.reshape(-1,1)
electronegativity_list =np.array(electronegativity_list)
electronegativity_list=electronegativity_list.reshape(-1,1)
mendeleievlist = np.array(mendeleievlist)
mendeleievlist=mendeleievlist.reshape(-1,1)

#density_feature0 = np.array([densidadplist,densidadslist])
#densidad
       






                #print(feature)
           
                    #print(numeroelectroness)
                #for elementos in electrones:
                 #   electroneslist=
                #print(elementos)
        #print('prueba:')
        #print(prueba)
        #sololetras=[''.join(x for x in prueba if x.isalpha())]
        #lista=[]
        #re.findall('[a-z]',i)
        #sololetras.append(i) 
       
        #print(no_integers)
#for x in prueba:
        #sololetras=re.findall('(A-Z][^A-Z]*/w+)',x)
        #print(sololetras)
       
      

