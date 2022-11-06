import numpy as np
import math
import pprint
import sys


razredi = {}

def read_file(file_name):

    # prebere vektorje vzorcev in jih zapise v seznam

    f = open(file_name, "r")

    lines = f.read().splitlines()

    seznam = []

    for l in lines:
        temp1 = l.split(",")
        temp2 = [ float(x) for x in temp1[:-1] ]
        seznam.append(temp2)
        razredi[str(temp2)] = temp1[-1]

    return seznam

def ascii(clust2, clust1, i):

    i = i + 1

    if np.array(clust2,dtype="object").shape == (4,):
        for a in range(i):
            print("     ", end = "")
        print(" ---" + str(clust2) + ' ' + razredi[str(clust2)])

    else:
        ascii(clust2[len(clust2) - 1], clust2[len(clust2) - 2], i)

    for a in range(i - 1):
        print("     ", end = "")
    print("---|")

    if np.array(clust1,dtype="object").shape == (4,):
        for a in range(i):
            print("     ", end = "")
        print(" ---" + str(clust1) + ' ' + razredi[str(clust1)])
    else:
        ascii(clust1[len(clust1) - 1], clust1[len(clust1) - 2], i)


dict = {}
dict2 = {}
rojenja = []

class HierarchicalClustering:

    def __init__(self, seznam):
        self.seznam = seznam

    def cluster_distance1(self, c1, c2): # izračuna razdaljo med rojema = razdalja med najbližjima vzorcema v rojih


        if (np.array(c1,dtype="object").shape) == (4,) and (np.array(c2,dtype="object").shape) == (4,): # če sta oba roja še ne zdržena v skupine, torej enaka vhodnim vektorjem vzorcev - torej oba predstavljata le vsak svoj vhodni vzorec

            if str([c1,c2]) in dict: # pogledamo če sta roja shranjena v slovarju
                raz = dict[str([c1,c2])] # če ja, potem uporabimo shranjeno razdaljo
            elif str([c2,c1]) in dict:
                raz = dict[str([c2,c1])]
            else:
                raz = np.linalg.norm(np.array(c1)-np.array(c2)) # če ne, potem izračunamo ebklidovo razdaljo med vzorcema
                dict[str([c1,c2])] = raz # ter jo shranimo v slovar

        elif (np.array(c1,dtype="object").shape) == (4,): # če prvi roj vsebuje le en sam vzorec

            if str([c1,c2]) in dict:
                raz = dict[str([c1,c2])]
            elif str([c2,c1]) in dict:
                raz = dict[str([c2,c1])]
            else: # rekurzivno izračunamo razdaljo med rojema po enačbi, podani v poročilu
                d1 = self.cluster_distance1(c1, c2[0]) # drugi roj razdelimo na dva roja ter izračunamo razdlje s prvim rojem -- glej enačbo
                d2 = self.cluster_distance1(c1, c2[1])
                raz = 0.5 * d1 + 0.5 * d2 - 0.5 * abs(d1 - d2)
                dict[str([c1,c2])] = raz

        else: # če drugi roj vsebuje le en sam vzorec ali oba roja vsebujeta več vzorcev

            if str([c1,c2]) in dict:
                raz = dict[str([c1,c2])]
            elif str([c2,c1]) in dict:
                raz = dict[str([c2,c1])]
            else:
                d1 = self.cluster_distance1(c1[0],c2) # prvi roj razdelimo na dva roja
                d2 = self.cluster_distance1(c1[1],c2)
                raz = 0.5 * d1 + 0.5 * d2 - 0.5 * abs(d1 - d2)
                dict[str([c1,c2])] = raz

        return raz

    def cluster_distance2(self, c1, c2): # izračuna razdaljo med rojema = razdalja med najbolj oddaljenima vzorcema v rojih

        # algoritem je enak kot v cluster_distance_1, le enačba za izračun razdalje se razlikuje (+0.5*abs namesto -0.5*abs)

            if (np.array(c1,dtype="object").shape) == (4,) and (np.array(c2,dtype="object").shape) == (4,):

                if str([c1,c2]) in dict:
                    raz = dict[str([c1,c2])]
                elif str([c2,c1]) in dict:
                    raz = dict[str([c2,c1])]
                else:
                    raz = np.linalg.norm(np.array(c1)-np.array(c2))
                    dict[str([c1,c2])] = raz

            elif (np.array(c1,dtype="object").shape) == (4,):

                if str([c1,c2]) in dict:
                    raz = dict[str([c1,c2])]
                elif str([c2,c1]) in dict:
                    raz = dict[str([c2,c1])]
                else:
                    d1 = self.cluster_distance2(c2[0], c1)
                    d2 = self.cluster_distance2(c2[1],c1)
                    raz = 0.5 * d1 + 0.5 * d2 + 0.5 * abs(d1 - d2)
                    dict[str([c1,c2])] = raz

            else:

                if str([c1,c2]) in dict:
                    raz = dict[str([c1,c2])]
                elif str([c2,c1]) in dict:
                    raz = dict[str([c2,c1])]
                else:
                    d1 = self.cluster_distance2(c1[0],c2)
                    d2 = self.cluster_distance2(c1[1],c2)
                    raz = 0.5 * d1 + 0.5 * d2 + 0.5 * abs(d1 - d2)
                    dict[str([c1,c2])] = raz

            return raz

    def cluster_distance3(self, c1, c2): # izračuna razdaljo med rojema = razdalja med središčema rojev


            if (np.array(c1,dtype="object").shape) == (4,) and (np.array(c2,dtype="object").shape) == (4,):

                if str([c1,c2]) in dict:
                    raz = dict[str([c1,c2])]
                elif str([c2,c1]) in dict:
                    raz = dict[str([c2,c1])]
                else:
                    raz = np.linalg.norm(np.array(c1)-np.array(c2))
                    dict[str([c1,c2])] = raz

            elif (np.array(c1,dtype="object").shape) == (4,):

                if str([c1,c2]) in dict:
                    raz = dict[str([c1,c2])]
                elif str([c2,c1]) in dict:
                    raz = dict[str([c2,c1])]
                else:
                    N_k = 1 # število vzorcev v c1 je 1
                    if (np.array(c2[0],dtype="object").shape) == (4,): # izračunamo število vzorcev v prvem in drugem delu roja c2
                        N_i = 1
                    else:
                        N_i = self.count_samples(c2[0])
                    if (np.array(c2[1],dtype="object").shape) == (4,):
                        N_j = 1
                    else:
                        N_j = self.count_samples(c2[1])

                    a_i = (N_k + N_i) / (N_i + N_j + N_k) # izračunamo koeficiente za enačbo
                    a_j = (N_k + N_j) / (N_i + N_j + N_k)
                    b = - N_k / (N_i + N_j + N_k)

                    d1 = self.cluster_distance3(c1,c2[0])
                    d2 = self.cluster_distance3(c1,c2[1])
                    d3 = self.cluster_distance3(c2[0],c2[1])

                    raz = a_i * d1 + a_j * d2 + b * d3

                    dict[str([c1,c2])] = raz

            else:

                if str([c1,c2]) in dict:
                    raz = dict[str([c1,c2])]
                elif str([c2,c1]) in dict:
                    raz = dict[str([c2,c1])]
                else:
                    if np.array(c2,dtype="object").shape == (4,): # izračunamo število vzorcev v prvem roju in prvem in drugem delu roja c2
                        N_k = 1
                    else:
                        N_k = self.count_samples(c2)
                    if (np.array(c1[0],dtype="object").shape) == (4,):
                        N_i = 1
                    else:
                        N_i = self.count_samples(c1[0])
                    if (np.array(c1[1],dtype="object").shape) == (4,):
                        N_j = 1
                    else:
                        N_j = self.count_samples(c1[1])

                    a_i = (N_k + N_i) / (N_i + N_j + N_k)
                    a_j = (N_k + N_j) / (N_i + N_j + N_k)
                    b = - N_k / (N_i + N_j + N_k)

                    d1 = self.cluster_distance3(c1[0],c2)
                    d2 = self.cluster_distance3(c1[1],c2)
                    d3 = self.cluster_distance3(c1[0],c1[1])

                    raz = a_i * d1 + a_j * d2 + b * d3

                    dict[str([c1,c2])] = raz

            return raz

    def count_samples(self, l): # prešteje število vzorcev v podanem roju

        count = 0
        for elem in l:
            if (np.array(elem,dtype="object").shape) == (4,): # če ostane le en vzorec
                count += 1
            else:
                count += self.count_samples(elem) # seznam nima le enega vzorca, ga ponovno razbijemo
        return count

    def closest_clusters(self, razdalja):

        distance = 1000000000 # inicializiramo začetno razdaljo med rojema in začetna roja
        clust1 = self.seznam[0]
        clust2 = self.seznam[0]

        for i in range(len(self.seznam)):
            for j in range(i+1,len(self.seznam)): # za vsak par rojev v seznamu

                if j != i:
                    if razdalja==1: # izračunamo razdaljo med rojema
                        dist = self.cluster_distance1(self.seznam[i], self.seznam[j])
                    if razdalja==2:
                        dist = self.cluster_distance2(self.seznam[i], self.seznam[j])
                    if razdalja==3:
                        dist = self.cluster_distance3(self.seznam[i], self.seznam[j])

                    if dist < distance: # pogledamo, če je to trenutno najmanjša razdalja
                        distance = dist # če ja, potem shranimo to razdaljo in roja, ki jo tvorita
                        clust1 = self.seznam[i]
                        clust2 = self.seznam[j]


        return (clust1, clust2, distance)

    def run(self, razdalja):

        while len(self.seznam) > 1: # dokler nam ne ostane le en roj

            clust1, clust2, distanceClust = self.closest_clusters(razdalja) # izračunamo najbližja si roja in razdaljo med njima

            newCluster = [clust1, clust2] # združimo roja v en roj

            rojenja.append(self.seznam[:]) # shranimo trenutno rojenje

            self.seznam.remove(clust1)
            self.seznam.remove(clust2)
            self.seznam.append(newCluster) # odstranimo najbližja si roja in ju zamenjamo z novo ustvarjenim rojem

            dict2[str(newCluster)] = distanceClust # shranimo razdaljo med rojema v slovar

        if len(self.seznam) == 1:
            self.seznam = self.seznam[0] # končnemu rojenju odstranimo zadnje [ ], saj jih je preveč

        return self.seznam

    def plot_tree(self):

        ascii(self.seznam[len(self.seznam) - 1], self.seznam[len(self.seznam) - 2], 0)

def mera_q(DATA_FILE):

    sez = read_file(DATA_FILE)
    st = len(sez) # začetni seznam rojev -- vsak vektor vzorca kot svoj roj

    num = 0
    den = 0

    for i in range(st-1):
        for j in range(i+1,st): # za vsak par vzorcev

            if str([sez[i],sez[j]]) in dict:  # iz slovarja pridobimo razdaljo med vzorcema = D
                D = dict[str([sez[i],sez[j]])]
            elif str([sez[j],sez[i]]) in dict:
                D = dict[str([sez[j],sez[i]])]

            l = 100000

            for key in dict2.keys(): # za vsak ključ v seznamu, kjer so shranjene razdalje med roji, ki smo jih združili
                if str(sez[i]) in key and str(sez[j]) in key:
                    if len(key) < l: # poiščemo najmanjši tak roj, v katerem sta oba vzorca
                        l = len(key)
                        D_c = dict2[key] # razdalja pri takšnem roju je razdalja med rojema, ki sta združena v tem roju = D_c

            num = num + (D - D_c)**2 # seštejemo pri vsakem paru vzorcev -- glej enačbo
            den = den + D**2

    Q = num/den

    return Q


def mera_cpcc(DATA_FILE):

    sez = read_file(DATA_FILE)
    st = len(sez)

    D_sum = 0 # inicializiramo vse parametre, potrebne za izračun
    D_c_sum = 0
    D_D_c_sum = 0
    D_sum_sqrt = 0
    D_c_sum_sqrt = 0

    for i in range(st):
        for j in range(i+1,st):

            if str([sez[i],sez[j]]) in dict:
                D = dict[str([sez[i],sez[j]])] # razdalja med vzorcema
            elif str([sez[j],sez[i]]) in dict:
                D = dict[str([sez[j],sez[i]])]

            l = 100000
            for key in dict2.keys():
                if str(sez[i]) in key and str(sez[j]) in key:
                    if len(key) < l:
                        l = len(key)
                        D_c = dict2[key] # razdalja, pri kateri se vzorca združita

            D_sum += D # dodamo izračune za vzorca potrebnim parametrom
            D_c_sum += D_c
            D_D_c_sum += D * D_c
            D_sum_sqrt += D**2
            D_c_sum_sqrt += D_c**2

    # izračunamo po enačbi, predstavljeni v poročilu
    R = st*(st-1)/2
    D_ = (1/R) * D_sum
    D_c_ = (1/R) * D_c_sum
    num = (1/R) * D_D_c_sum - D_ * D_c_
    den = math.sqrt((1/R) * D_sum_sqrt - D_**2) * math.sqrt((1/R) * D_c_sum_sqrt - D_c_**2)
    cpcc = num/den

    return cpcc

def PB(DATA_FILE): # poda optimalno število rojev za rojenje

    sez = read_file(DATA_FILE)
    st = len(sez)

    max_pb = 0
    st_rojev = 0
    max_roj = []

    for i in rojenja: # za vsako rojenje

        D_b_sum = 0 # vsota vseh razdalj vzorcev, ki so v rojih
        D_w_sum = 0 # vsota vseh razdalj med vzorci, ki so v rojih in vzorci, ki so izven rojev
        f_b = 0 # število parov vorcev v rojih
        f_w = 0 # število parov vzorcev izven rojev
        std_sum = [] # vse razdalje med vzorci

        for roj in i:  # za vsak roj v rojenju

            for vz1 in range(st):
                for vz2 in range(vz1+1,st): # za vsak par vzorcev

                    if str(sez[vz1]) in str(roj) and str(sez[vz2]) in str(roj):  # če je par v roju

                        if str([sez[vz1],sez[vz2]]) in dict:
                            w = dict[str([sez[vz1],sez[vz2]])] # razdalja med vzorcema
                        elif str([sez[vz2],sez[vz1]]) in dict:
                            w = dict[str([sez[vz2],sez[vz1]])]

                        D_w_sum += w # prištejemo k razdalji vseh vzorcev, ki so v rojih
                        f_w += 1

                    elif str(sez[vz1]) in str(roj) or str(sez[vz2]) in str(roj) or sez[vz1] == roj or sez[vz2] == roj:  # pari vzorcev, pri katerih je eden iz enega, drugi iz drugega roja --> so iz različnih rojev

                        if str([sez[vz1],sez[vz2]]) in dict:
                            b = dict[str([sez[vz1],sez[vz2]])]
                        elif str([sez[vz2],sez[vz1]]) in dict:
                            b = dict[str([sez[vz2],sez[vz1]])]

                        D_b_sum += b # prištejemo k razdalji vzorcev iz različnih rojev
                        f_b += 1

        for vz1 in range(st):    # seznam vseh razdalj med vzorci
            for vz2 in range(vz1+1,st):
                if str([sez[vz1],sez[vz2]]) in dict:
                    r = dict[str([sez[vz1],sez[vz2]])]
                elif str([sez[vz2],sez[vz1]]) in dict:
                    r = dict[str([sez[vz2],sez[vz1]])]
                std_sum.append(r)

        # izračunamo po enačbi, predstavljeni v poročilu

        if f_w == 0: # zaradi deljenja z 0
            f_w = 1
        if f_b == 0:
            f_b = 1

        D_b_ = D_b_sum / f_b
        D_w_ = D_w_sum / f_w
        n_d = f_w + f_b
        std = np.std(std_sum)

        PB = (D_b_ - D_w_) * (math.sqrt(f_w * f_b / n_d**2) / std)

        if PB > max_pb: # iščemo največji koeficient PB
            max_pb = PB
            st_rojev = len(i) # število rojev v rojenju
            max_roj = i

    return st_rojev, max_roj

def oznake(rojenje, raz): # vrnse seznam oznak (razredov) v vhodnem roju

    for roj in rojenje:

        if (np.array(roj,dtype="object").shape) == (4,):
            raz.append(razredi[str(roj)])
        else:
            raz = oznake(roj, raz)
    return raz

def porazdelitev(rojenje): # izračuna in izpiše porazdelitev vzorcev po razredih za vsak roj v vhodnem rojenju

    idx = 0

    for roj in rojenje:

        idx += 1

        ozn = oznake(roj, [])

        A = sum(1 for i in ozn if i == 'Iris-setosa')
        B = sum(1 for i in ozn if i == 'Iris-versicolor')
        C = sum(1 for i in ozn if i == 'Iris-virginica')

        A = A/len(ozn)
        B = B/len(ozn)
        C = C/len(ozn)

        print("Roj ", idx)
        print('Iris-setosa: ', A)
        print('Iris-versicolor: ', B)
        print('Iris-virginica: ', C)


if __name__ == "__main__":


    DATA_FILE = "iris.data"

    razdalja = input("Katero razdaljo med skupinami zelite uporabiti?\n1: razdalja med najbližjima vzorcema\n2: razdalja med najbolj oddaljenima vzorcema\n3: razdalja med središčema skupin\n")

    hc = HierarchicalClustering(read_file(DATA_FILE)) # inicializira ačetni seznam vektorjev vzorcev
    roj = hc.run(int(razdalja)) # končno rojenje, zapisano kot seznam seznamov. Vsak seznam predstavlja svoj roj.

    print(roj) # seznam seznamov

    hc.plot_tree() # narisi dendrogram

    Q = mera_q(DATA_FILE) # izračunamo mero Q
    print("mera Q: ", Q)

    cpcc = mera_cpcc(DATA_FILE) # izračunamo mero CPCC
    print("mera CPCC: ", cpcc)

    st, rojenje = PB(DATA_FILE) # izračunamo število naravnih rojev
    print("število naravnih rojev: ", st)

    porazdelitev(rojenje) # izpiši porazdelitev vzorcev v rojih po razredih
