# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:17:33 2021

@author: Souha
"""

import time
import random
import numpy as np
from math import *
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Partie Commune : 

     
def DistancedeManhattan(X,Y,Z,U,V,W,x,y,z,u,v,w):
    a=abs(X-x)
    b=abs(Y-y)
    c=abs(Z-z)
    d=abs(U-u)
    e=abs(V-v)
    f=abs(W-w)
    g=a+b+c+d+e+f
    return g


def DistanceEclidienne(X,Y,Z,U,V,W,x,y,z,u,v,w):
    a=(X-x)**2
    b=(Y-y)**2
    c=(Z-z)**2
    d=(U-u)**2
    e=(V-v)**2
    f=(W-w)**2
    g=sqrt(a+b+c+d+e+f)
    return g


#%% Partie 1 : Etudie et construction du TrainSet et du TestSet avec seulement le fichier Data 
    
def LireCSV():
    df=pd.read_csv("D:/ESILV/Projet IA/Projet KNN/data.csv",sep=',',names=['X','Y','Z','U','V','W','Classe'], header=None)
    return df

# On va diviser selon un nombre alétoire compris entre 15% et 25% le dataset
def Pourcentage():
    a=random.uniform(0,1)
    while a<0.75 or a>0.85:
        a=random.uniform(0,1)
    return a

 #On retourne le nouveau dataset qu'on va étudier
def PartitionEnListe(df,a):
    b=round(a*len(df))
    TrainSet=df.iloc[:b]
    TestSet=df.iloc[b:len(df)]
    return TrainSet,TestSet,b

def Comparer(df,LC,Pos,k):
    x=0
    compteur=0
    l=0
    for j in range(len(Pos)):
        for i in range(k):
            if(df["Classe"][LC[l][i]]==df["Classe"][Pos[j]]):
                x+=1
            if(x>=round(k/2)):
                compteur+=1
                break
        x=0
        l+=1
    pourcentage=((compteur/len(Pos))*100)
    return pourcentage

def Algo():
    debut = time.time()
    df=LireCSV()
    a=Pourcentage()
    TrainSet,TestSet,b=PartitionEnListe(df,a)
    LC=[]
    Pos=[]
    beta=[]
    k = 3
    print(b)
    print(len(TestSet["X"]))
    for i in range(b,len(df["X"])):
        beta = []
        print(6)
        for j in range(len(TrainSet["X"])):
            X=TrainSet["X"][j]
            Y=TrainSet["Y"][j]
            Z=TrainSet["Z"][j]
            U=TrainSet["U"][j]
            V=TrainSet["V"][j]
            W=TrainSet["W"][j]
            x=TestSet["X"][i]
            y=TestSet["Y"][i]
            z=TestSet["Z"][i]
            u=TestSet["U"][i]
            v=TestSet["V"][i]
            w=TestSet["W"][i]
            beta.append([DistancedeManhattan(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j])
            #beta.append([DistanceEclidienne(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j]) 
        beta = sorted(beta, key = lambda x : x[0])
        best=[beta[x][-1] for x in range(k)]
        print("le meilleur triplet pos newdf",best," la valeur de pos de newdf2 associé :",beta[b-i][1])
        LC.append(best)
        Pos.append(beta[b-i][1])
    print(len(Pos))
    pourcentage=Comparer(df, LC,Pos,k)
    print("Le pourcentage de réussite est : ",pourcentage)
    fin = time.time()
    print("Temps :",fin-debut,"secondes")
    

def AlgoK():
    debut = time.time()
    df=LireCSV()
    a=Pourcentage()
    TrainSet,TestSet,b=PartitionEnListe(df,a)
    LC=[]
    Pos=[]
    beta=[]
    LesK=[]
    LesPourcentages=[]
    for k in range(2,21):
        LC=[]
        Pos=[]
        for i in range(b,len(df["X"])):
            beta = []
            for j in range(len(TrainSet["X"])):
                 X=TrainSet["X"][j]
                 Y=TrainSet["Y"][j]
                 Z=TrainSet["Z"][j]
                 U=TrainSet["U"][j]
                 V=TrainSet["V"][j]
                 W=TrainSet["W"][j]
                 x=TestSet["X"][i]
                 y=TestSet["Y"][i]
                 z=TestSet["Z"][i]
                 u=TestSet["U"][i]
                 v=TestSet["V"][i]
                 w=TestSet["W"][i]
                 beta.append([DistancedeManhattan(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j])
                #beta.append([DistanceEclidienne(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j]) 
            beta = sorted(beta, key = lambda x : x[0])
            best=[beta[x][-1] for x in range(k)]
            #print("le meilleur triplet pos newdf",best," la valeur de pos de newdf2 associé :",beta[b-i][1])
            LC.append(best)
            Pos.append(beta[b-i][1])
        pourcentage=Comparer(df, LC,Pos,k)
        print("Le pourcentage de réussite est : ",pourcentage,"et le k correspondant est :",k)
        LesK.append(k)
        LesPourcentages.append(pourcentage)
    print("\n")
    print("Le K qui permet d'avoir la valeur maximale est :",LesK[LesPourcentages.index(max(LesPourcentages))])
    Graphique(LesK,LesPourcentages)
    fin = time.time()
    print("Temps :",fin-debut,"secondes")
    
def Graphique(x,y):
    plt.title("Graphique des pourcentages de réussite en fonction de la valeur de K")
    plt.xlabel('K')
    plt.ylabel('Pourcentage') 
    plt.plot(x,y)
    plt.scatter(x,y,c = 'red')
    plt.show()
    
    
    
    
#%% Partie 2 : Etudie et construction du TrainSet et du TestSet avec les fichiers Data et preTest

def LireCSV_2():
    TrainSet=pd.read_csv("D:/ESILV/Projet IA/Projet KNN/data.csv",sep=',',names=['X','Y','Z','U','V','W','Classe'], header=None)
    TestSet=pd.read_csv("D:/ESILV/Projet IA/Projet KNN/preTest.csv",sep=',',names=['X','Y','Z','U','V','W','Classe'], header=None)
    return TrainSet,TestSet

def Comparer_2(TrainSet,TestSet,LC,Pos,k):
    x=0
    compteur=0
    l=0
    for j in range(len(Pos)):
        for i in range(k):
            if(TrainSet["Classe"][LC[l][i]]==TestSet["Classe"][Pos[j]]):
                x+=1
            if(x>=round(k/2)):
                compteur+=1
                break
        x=0
        l+=1
    pourcentage=((compteur/len(Pos))*100)
    return pourcentage     


def Algo2():
    debut = time.time()
    TrainSet,TestSet=LireCSV_2()
    LC=[]
    Pos=[]
    beta=[]
    k = 5
    for i in range(len(TestSet["X"])):
        beta = []
        for j in range(len(TrainSet["X"])):
            X=TrainSet["X"][j]
            Y=TrainSet["Y"][j]
            Z=TrainSet["Z"][j]
            U=TrainSet["U"][j]
            V=TrainSet["V"][j]
            W=TrainSet["W"][j]
            x=TestSet["X"][i]
            y=TestSet["Y"][i]
            z=TestSet["Z"][i]
            u=TestSet["U"][i]
            v=TestSet["V"][i]
            w=TestSet["W"][i]
            beta.append([DistancedeManhattan(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j])
            #beta.append([DistanceEclidienne(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j]) 
        beta = sorted(beta, key = lambda x : x[0])
        best=[beta[x][-1] for x in range(k)]
        #print("le meilleur triplet pos newdf",best," la valeur de pos de newdf2 associé :",beta[i][1])
        LC.append(best)
        Pos.append(beta[i][1])
    pourcentage=Comparer_2(TrainSet,TestSet, LC,Pos,k)
    print("Le pourcentage de réussite est : ",pourcentage)
    fin = time.time()
    print("Temps :",fin-debut,"secondes")
    

def AlgoK2():
    debut = time.time()
    TrainSet,TestSet=LireCSV_2()
    LC=[]
    Pos=[]
    beta=[]
    LesK=[]
    LesPourcentages=[]
    for k in range(2,21):
        LC=[]
        Pos=[]
        for i in range(len(TestSet["X"])):
            beta = []
            for j in range(len(TrainSet["X"])):
                X=TrainSet["X"][j]
                Y=TrainSet["Y"][j]
                Z=TrainSet["Z"][j]
                U=TrainSet["U"][j]
                V=TrainSet["V"][j]
                W=TrainSet["W"][j]
                x=TestSet["X"][i]
                y=TestSet["Y"][i]
                z=TestSet["Z"][i]
                u=TestSet["U"][i]
                v=TestSet["V"][i]
                w=TestSet["W"][i]
                beta.append([DistancedeManhattan(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j])
                #beta.append([DistanceEclidienne(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j]) 
            beta = sorted(beta, key = lambda x : x[0])
            best=[beta[x][-1] for x in range(k)]
            #print("le meilleur triplet pos newdf",best," la valeur de pos de newdf2 associé :",beta[i][1])
            LC.append(best)
            Pos.append(beta[i][1])
        pourcentage=Comparer_2(TrainSet,TestSet, LC,Pos,k)
        print("Le pourcentage de réussite est : ",pourcentage,"et le k correspondant est :",k)
        LesK.append(k)
        LesPourcentages.append(pourcentage)
    print("\n")
    print("Le K qui permet d'avoir la valeur maximale est :",LesK[LesPourcentages.index(max(LesPourcentages))])
    Graphique(LesK,LesPourcentages)
    fin = time.time()
    print("Temps :",fin-debut,"secondes")
    

def Graphique2(x,y):
    plt.title("Graphique des pourcentages de réussite en fonction de la valeur de K")
    plt.xlabel('K')
    plt.ylabel('Pourcentage') 
    plt.plot(x,y)
    plt.scatter(x,y,c = 'red')
    plt.show()
    
    
    

#%% Partie 3 (partie final) : Etudie et construction du TrainSet et du TestSet avec les fichiers Data et finalTest

def LireCSV_3():
    TrainSet=pd.read_csv("D:/ESILV/Projet IA/Projet KNN/data.csv",sep=',',names=['X','Y','Z','U','V','W','Classe'], header=None)
    TestSet=pd.read_csv("D:/ESILV/Projet IA/Projet KNN/finalTest.csv",sep=',',names=['X','Y','Z','U','V','W'], header=None)
    return TrainSet,TestSet

def Comparer_3(TrainSet,LC,Pos,k):
    x=1
    compteur=0
    l=0
    Classe=[]
    for j in range(len(Pos)):
        for i in range(k):
            if(i<k-1):
                if(TrainSet["Classe"][LC[l][i]]==TrainSet["Classe"][LC[l][i+1]]):
                    x+=i
            else:
                if(TrainSet["Classe"][LC[l][i-1]]==TrainSet["Classe"][LC[l][i]]):
                     x+=i
                if(x>=round(k/2)):
                    compteur+=1
                    Classe.append(TrainSet["Classe"][LC[l][i]])
                    break
                if(x<round(k/2)):
                    index=random.randint(0,k-1)
                    Classe.append(TrainSet["Classe"][LC[l][index]])
                    break
        x=0
        l+=1
    pourcentage=((compteur/len(Pos))*100)
    return pourcentage,Classe  

def Algo3():
    debut = time.time()
    TrainSet,TestSet=LireCSV_3()
    LC=[]
    Pos=[]
    beta=[]
    k = 3
    compteur=0
    for i in range(len(TestSet["X"])):
        beta = []
        for j in range(len(TrainSet["X"])):
            X=TrainSet["X"][j]
            Y=TrainSet["Y"][j]
            Z=TrainSet["Z"][j]
            U=TrainSet["U"][j]
            V=TrainSet["V"][j]
            W=TrainSet["W"][j]
            x=TestSet["X"][i]
            y=TestSet["Y"][i]
            z=TestSet["Z"][i]
            u=TestSet["U"][i]
            v=TestSet["V"][i]
            w=TestSet["W"][i]
            beta.append([DistancedeManhattan(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j])
            #beta.append([DistanceEclidienne(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j])
        compteur+=1
        if(compteur==len(TrainSet["X"])):
            compteur=0
        if(i<len(TrainSet["X"])):
             beta = sorted(beta, key = lambda x : x[0])
             best=[beta[x][-1] for x in range(k)]
             #print("Les plus proches voisins selon k =",k,"sont :",best,"et l'index de l'élément dans le TestSet associé est :",beta[i][1])
             LC.append(best)
             Pos.append(beta[i][1])
        else:
             beta = sorted(beta, key = lambda x : x[0])
             best=[beta[x][-1] for x in range(k)]
             #print("Les plus proches voisins selon k =",k,"sont :",best,"et l'index de l'élément dans le TestSet associé est :",beta[compteur][1])
             LC.append(best)
             Pos.append(beta[compteur][1])
    pourcentage,Classe=Comparer_3(TrainSet, LC,Pos,k)
    print("Le pourcentage d'estimation de réussite, qui se base la majorité de la classe des plus proches voisins selon le k, est : ",pourcentage,"%.")
    EcrireFichier(Classe)
    fin = time.time()
    print("Temps :",fin-debut,"secondes")

# creer un tableau de char pour les couleurs 

def EcrireFichier(x):
    fichier=open("D:/ESILV/Projet IA/Projet KNN/FichierdedataF.txt","w")
    for i in range(len(x)):
        if(i<len(x)-1):
            fichier.write(str(x[i]))
            fichier.write("\n")
        else:
            fichier.write(str(x[i]))
    fichier.close()
    
#%% Partie avec deux dataset en un 

def LireCSV_4():
    TrainSet1=pd.read_csv("D:/ESILV/Projet IA/Projet KNN/data.csv",sep=',',names=['X','Y','Z','U','V','W','Classe'], header=None)
    TrainSet2=pd.read_csv("D:/ESILV/Projet IA/Projet KNN/preTest.csv",sep=',',names=['X','Y','Z','U','V','W','Classe'], header=None)
    TrainSet=pd.merge(TrainSet1,TrainSet2,how = 'outer')
    TestSet=pd.read_csv("D:/ESILV/Projet IA/Projet KNN/finalTest.csv",sep=',',names=['X','Y','Z','U','V','W'], header=None)
    return TrainSet,TestSet

def Comparer_4(TrainSet,LC,Pos,k):
    x=1
    compteur=0
    l=0
    Classe=[]
    for j in range(len(Pos)):
        for i in range(k):
            if(i<k-1):
                if(TrainSet["Classe"][LC[l][i]]==TrainSet["Classe"][LC[l][i+1]]):
                    x+=i
            else:
                if(TrainSet["Classe"][LC[l][i-1]]==TrainSet["Classe"][LC[l][i]]):
                     x+=i
                if(x>=round(k/2)):
                    compteur+=1
                    Classe.append(TrainSet["Classe"][LC[l][i-1]])
                    break
                if(x<round(k/2)):
                    index=random.randint(0,k-1)
                    Classe.append(TrainSet["Classe"][LC[l][index]])
                    break
        x=0
        l+=1
    pourcentage=((compteur/len(Pos))*100)
    return pourcentage,Classe  



def Algo4():
    debut = time.time()
    TrainSet,TestSet=LireCSV_4()
    LC=[]
    Pos=[]
    beta=[]
    k = 5
    compteur=0
    for i in range(len(TestSet["X"])):
        beta = []
        for j in range(len(TrainSet["X"])):
            X=TrainSet["X"][j]
            Y=TrainSet["Y"][j]
            Z=TrainSet["Z"][j]
            U=TrainSet["U"][j]
            V=TrainSet["V"][j]
            W=TrainSet["W"][j]
            x=TestSet["X"][i]
            y=TestSet["Y"][i]
            z=TestSet["Z"][i]
            u=TestSet["U"][i]
            v=TestSet["V"][i]
            w=TestSet["W"][i]
            beta.append([DistancedeManhattan(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j])
            #beta.append([DistanceEclidienne(X,Y,Z,U,V,W,x,y,z,u,v,w), i, j])
        compteur+=1
        if(compteur==len(TrainSet["X"])):
            compteur=0
        if(i<len(TrainSet["X"])):
             beta = sorted(beta, key = lambda x : x[0])
             best=[beta[x][-1] for x in range(k)]
             #print("Les plus proches voisins selon k =",k,"sont :",best,"et l'index de l'élément dans le TestSet associé est :",beta[i][1])
             LC.append(best)
             Pos.append(beta[i][1])
        else:
             beta = sorted(beta, key = lambda x : x[0])
             best=[beta[x][-1] for x in range(k)]
             #print("Les plus proches voisins selon k =",k,"sont :",best,"et l'index de l'élément dans le TestSet associé est :",beta[compteur][1])
             LC.append(best)
             Pos.append(beta[compteur][1])
    pourcentage,Classe=Comparer_4(TrainSet, LC,Pos,k)
    print("Le pourcentage d'estimation de réussite, qui se base la majorité de la classe des plus proches voisins selon le k, est : ",pourcentage,"%.")
    EcrireFichier2(Classe)
    fin = time.time()
    print("Temps :",fin-debut,"secondes")

# creer un tableau de char pour les couleurs 

def EcrireFichier2(x):
    fichier=open("D:/ESILV/Projet IA/Projet KNN/FichierdedataFK52.txt","w")
    for i in range(len(x)):
        if(i<len(x)-1):
            fichier.write(str(x[i]))
            fichier.write("\n")
        else:
            fichier.write(str(x[i]))
    fichier.close()
    
