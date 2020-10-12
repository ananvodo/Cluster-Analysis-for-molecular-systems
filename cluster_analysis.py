#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Author: Andres Vodopivec
# @Date:   2020-05-12
# @Filename: cluster_analysis.py
# @Last modified by:   Andres Vodopivec
# @Last modified time: 2020-09-24T10:32
#
# Usage:
# This program is a standalone program to calculate clusters using an
# aglomerative cluster single-linkage algorithm. All atoms inside every
# molecule are considered to determine meighbor molecules.
# Please use python cluster_analysis.py --help for usage information.
# Please be sure to read carefully all the help info and comments to clear any
# doubts.

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# Importing modules
# -----------------
import numpy as np
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import traceback
import functools # to flatten lists
import operator # to flatten lists
import math
import pandas as pd
import argparse
import concurrent.futures
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# Class for reading and writing files
# -----------------------------------


class FileUtil():
    """ FileUtil is a class to read and wrtie files ONLY """

    def __init__(self, file):
        """
        This is the Constructor to instanciate the object for FileUtil() class
        """
        self.file = file

    @classmethod
    def FileReader(cls, a_destinationFilePath, a_fileName):
        """
        FileUtil.FileReader is a Constructor that takes a path
        (/destinationFilePath/fileName) to instanciate the object.
        """
        FileUtil.CheckFuncInput(a_fileName, str,
                                'MoleculesInfo.fromList_GROreader')

        FileUtil.CheckFuncInput(a_destinationFilePath, str,
                                'MoleculesInfo.fromList_GROreader')

        print('\nReading new file with name ' + a_fileName +
              ' in directory: ' + a_destinationFilePath + '\n')
        filepath = os.path.join(a_destinationFilePath, a_fileName)
        fileList = []

        if not os.path.exists(filepath):
            raise FileNotFoundError('The file ' + a_fileName +
                                    ' in directory: '
                                    + a_destinationFilePath +
                                    ' was not found.')

        else:
            with open(filepath, 'r') as fin:
                fileList = fin.readlines()

        return cls(fileList)

    def FileWriter(self, a_destinationFilePath, a_fileName):
        """
        FileUtil.FileWriter is an instance method that print a object to
        /destinationFilePath/fileName.
        """

        print('\nWriting the new file with name ' + a_fileName +
              ' in directory: ' + a_destinationFilePath + '\n')
        filepath = os.path.join(a_destinationFilePath, a_fileName)

        if not os.path.exists(a_destinationFilePath):
            raise FileNotFoundError('The directory: ' +
                                    a_destinationFilePath +
                                    ' was not found.')

        else:
            with open(filepath, 'w') as fout:
                fout.writelines(self.file)

        return None

    @staticmethod
    def CheckFuncInput(a_inputVar, a_inputType, a_classDotFuncName):
        '''
        FileUtil.CheckFuncInput is a static method to check that input to any
        function in other classes has the correct type (i.e. string, int, ...)
        '''
        if isinstance(a_inputVar, a_inputType):
            pass
        else:
            raise ValueError('\nArgument {} in {} is not a/an {}\n'.format(
                    a_inputVar,
                    a_classDotFuncName,
                    str(a_inputType).split()[1]))

        return None

    @staticmethod
    def MakePathDir(a_classDotFuncName, a_destinationFilePath,
                    a_subFolder=None):
        '''
        FileUtil.MakePathDir is a static method to create the path for any
        file that is required as input for any function.
        This funtions looks that the directory provided is string format and
        that it exists.
        '''
        FileUtil.CheckFuncInput(a_destinationFilePath, str,
                                a_classDotFuncName)

        if a_subFolder is None:
            workPath = a_destinationFilePath
            print("\nUsing {} in path {}\n".format(
                    a_classDotFuncName, workPath))
        else:
            FileUtil.CheckFuncInput(a_subFolder, str,
                                    'MoleculesInfo.fromList_GROreader')
            workPath = os.path.join(a_destinationFilePath, a_subFolder)
            print("\nUsing {} in path {}\n".format(
                    a_subFolder, workPath))

        if not os.path.exists(workPath):
            raise FileNotFoundError('{} did not found path {}'.format(
                    a_classDotFuncName, workPath))

        return workPath

    def __del__(self):
        print("\nFreeing memory by deleting FileUtil object\n")
        # This is my destructor (it is not mandatory for python)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# Class to get system information
# -------------------------------

class MoleculesInfo():
    """
    MoleculesInfo class is a class to process and create .gro file objects
    and dictionary of molecules from a list ONLY
    """

    def __init__(self, totalSystemAtoms, HPs, oils, residTypeUnique, xBoxLen,
                 yBoxLen, zBoxLen):
        """
        This is the Constructor to instanciate the object for MoleculesInfo()
        class
        """
        self.HPs = HPs
        self.oils = oils
        self.totalSystemAtoms = totalSystemAtoms
        self.systemResidues = residTypeUnique
        self.xBoxLen = xBoxLen
        self.yBoxLen = yBoxLen
        self.zBoxLen = zBoxLen



    @classmethod
    def fromList_GROreader(cls, a_groFileName, a_itpFileName,
                           a_destinationFilePath, a_subFolder=None):
        """
        MoleculesInfo.fromList_GROreader is a class constructor to get
        the important data from the groFile.
        It needs two input fileNames: a_itpFileName and a_groFileName.
        The a_groFileName is the groFile that has the molecule described in
        itpFileName and it must be compatible with gromacs GRO format and
        martiniFF.
        The a_itpFileName is the Protein_A.itp or any itp file that is the
        output of martinize.py or that complies with martini itp molecules
        topology.
        """
        # ---------------------------------------
        # GetDict is a function to get the dictionary for the HP and oil
        # molecules independently
        # --------------------------------------
        def GetDict(a_indexMolec, a_beadNum, a_molecName, a_xCoord, a_yCoord,
                    a_zCoord, a_atomNum):
            molecDict = {}
            keyDictNum = 1

            for i in range(a_indexMolec[0], a_indexMolec[-1], a_beadNum):
                index1 = i
                index2 = i + a_beadNum
                atomNumList = []
                xCoordList = []
                yCoordList = []
                zCoordList = []

                for k in range(index1, index2):
                    atomNumList.append(a_atomNum[k])
                    xCoordList.append(a_xCoord[k])
                    yCoordList.append(a_yCoord[k])
                    zCoordList.append(a_zCoord[k])

                keyDict = a_molecName + '_' + str(keyDictNum)
                molecDict[keyDict] = {'atomNum': atomNumList,
                                      'X_coord': xCoordList,
                                      'Y_coord': yCoordList,
                                      'Z_coord': zCoordList}
                keyDictNum += 1

            return molecDict

        # ------------
        # Getting the work directory where a_groFileName and a_itpFileName
        # exist
        # ------------
        workPath = FileUtil.MakePathDir('MoleculesInfo.fromList_GROreader',
                                        a_destinationFilePath,
                                        a_subFolder)

        # -----------------------------------------------------
        # Reading and getting the information from the gro file
        # -----------------------------------------------------

        print("\nCreating FileUtil object from {} in {}\n".format(
                a_groFileName, workPath))
        groFileOb = FileUtil.FileReader(workPath, a_groFileName)
        grofile = groFileOb.file

        # Initializing all variables that will be used
        totalSystemAtoms = 0
        residNum = []
        residType = []
        atomType = []
        atomNum = []
        xCoord = []
        yCoord = []
        zCoord = []

        # Saving the first line to get the total atoms in the system
        totalSystemAtoms = int(grofile[1].strip())

        # Making the lists of each variable accordingly to the spaces stated by GMX standard format.
        for line in range(2, len(grofile) - 1):

            residNum.append(int(grofile[line][0:5].strip()))
            residType.append(grofile[line][5:10].strip())
            atomType.append(grofile[line][10:15].strip())
            atomNum.append(int(grofile[line][15:20].strip()))
            xCoord.append(float(grofile[line][20:28].strip()))
            yCoord.append(float(grofile[line][28:36].strip()))
            zCoord.append(float(grofile[line][36:44].strip()))

        lastline = grofile[-1].split()
        xBoxLen = float(lastline[0])
        yBoxLen = float(lastline[1])
        zBoxLen = float(lastline[2])
        residTypeUnique = list(set(residType))

        # -----------------------------------------------------
        # Reading and getting the information from the itp file
        # -----------------------------------------------------

        print("\nCreating FileUtil object from {} in {}\n".format(
                a_itpFileName, workPath))
        itpFileOb = FileUtil.FileReader(workPath, a_itpFileName)
        itpFile = itpFileOb.file

        # Getting the information needed from the itpFile (Protein_A.itp)
        for i, line in enumerate(itpFile):
            if line.find('[ atoms ]') >= 0:
                atomtypeStart = i + 1
            if line.find('[ virtual_sites2 ]') >= 0:
                atomtypeEnds = i - 1

        itpFile = itpFile[atomtypeStart:atomtypeEnds]

        # Getting the sequence of Amino acids that make a HP molecule ONLY
        # residTypeUnique involves oils and water too, thus we cannot use it.
        hpAminoSequence = [] # sequence of animo acids (i.e. ALA, VAL, etc.)

        for line in itpFile:
            hpAmino = line.strip().split()[3]
            hpAmino = hpAmino.strip()
            hpAminoSequence.append(hpAmino)

        # Getting the indexes that correspond to HPs and to oils
        hpBeadNum = len(hpAminoSequence) # total number of beads in a single HP
        hpAminoSequence = list(set(hpAminoSequence)) # eliminating duplicates
        oilBeadNum = 3 # Benz and Dec have 3 beads only

        print("\nGetting the oil and HP indeces in the grofile\n")
        hpIndex = [] # indexes where HPs are located
        oilIndex = [] # indexes where oils are located

        for index, resid in enumerate(residType):
            if resid in hpAminoSequence:
                hpIndex.append(index)
            elif resid in ['DEC', 'BENZ']:
                oilIndex.append(index)

        # Getting the dictionaries based upon the indexes
        print("\nCreating the oil and HP dictionaries\n")
        oils = GetDict(oilIndex, oilBeadNum, 'oil', xCoord, yCoord,
                       zCoord, atomNum)
        HPs = GetDict(hpIndex, hpBeadNum, 'HP', xCoord, yCoord,
                      zCoord, atomNum)

        return cls(totalSystemAtoms, HPs, oils, residTypeUnique, xBoxLen,
                   yBoxLen, zBoxLen)


    def Caculate_molecs_weight(self, a_itpFileName, a_martiniItpName,
                               a_destinationFilePath, a_subFolder=None):
        '''
        MoleculesInfo.Caculate_molecs_weight is an instance method to calculate
        the weight of the molecules. In this case the HP and oil molecules.
        It needs two input fileNames: a_itpFileName and a_martiniItpName.
        The a_itpFileName is the Protein_A.itp or any itp file that is the
        output of martinize.py or that complies with martini itp molecules
        topology.
        The a_martiniItpName is the martini_v2.2P.itp. It is the file that has
        all the weight of the martini beads and non-bonded parameters.
        Both files must be located in /a_destinationFilePath/a_subFolder. Or
        only in a_destinationFilePath if a_subFolder was not provided.
        '''

        workPath = FileUtil.MakePathDir('MoleculesInfo.Caculate_molecs_weight',
                                        a_destinationFilePath, a_subFolder)

        # Getting the file objects
        print("\nCreating FileUtil object from {} in {}\n".format(
                a_itpFileName, workPath))
        itpFileOb = FileUtil.FileReader(workPath, a_itpFileName)

        print("\nCreating FileUtil object from {} in {}\n".format(
                a_martiniItpName, workPath))
        martiniItpFileOb = FileUtil.FileReader(workPath, a_martiniItpName)

        # Geeting the file member in FileUtil objects
        itpFile = itpFileOb.file
        martiniItpFile = martiniItpFileOb.file

        # -------------------------
        # Calculating the HP weight
        # -------------------------

        # Getting the piece needed in the martini_v2.2P.itp file
        # to create a bead-weight dictionary
        # ----------------------------------
        for index, line in enumerate(martiniItpFile):
            if line.find('; STANDARD types, 4:1 mapping') >= 0:
                topIndex = index + 1
            if line.find('[ nonbond_params ]') >= 0:
                bottomIndex = index - 1

        martiniItpFile = martiniItpFile[topIndex:bottomIndex]

        # Eliminating unwanted lines in the file
        for line in martiniItpFile:
            if (line.find(';') == 0):
                martiniItpFile.remove(line)

        # Creating a dictionary of the weights for each bead present in
        # martini_v2.2P.itp file

        beadsWeight = {}

        for line in martiniItpFile:
        # There are few empty lines. Try and except will help us to pass
        # those lines
            try:
                parts = line.strip().split()
                key = parts[0].strip() # this is the beadtype
                weight = int(float(parts[1].strip())) # this is the weight
                beadsWeight[key] = weight # filling the dictionary
            except:
                pass

        for i, line in enumerate(itpFile):
            if line.find('[ atoms ]') >= 0:
                atomtypeStart = i + 1
            if line.find('[ virtual_sites2 ]') >= 0:
                atomtypeEnds = i - 1

        # Getting the piece needed in the Protein_A file to get all the beads
        # present in the HP molecule
        # --------------------------
        itpFile = itpFile[atomtypeStart:atomtypeEnds]
        hpAtomtypeSequence = []
        for line in itpFile:
            hpAtomtypeSequence.append(line.split()[1])

        # Getting the weight of the HP molecule
        # ------------------------------------
        hpWweightSum = 0
        for atomtype in hpAtomtypeSequence:
            hpWweightSum = hpWweightSum + beadsWeight[atomtype]

        self.hpWeight = hpWweightSum

        # --------------------------
        # Calculating the oil weight
        # --------------------------

        beadsPerOil = 3

        for resid in self.systemResidues:
            if resid == 'BENZ':
                weightPerBear = 45
                break

            if resid == 'DEC':
                weightPerBear = 72
                break

        self.oilWeight = beadsPerOil * weightPerBear

        return None


    def __del__(self):
        print("\nFreeing memory by deleting MoleculesInfo object\n")
        # This is my destructor (it is not mandatory for python)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# Class to do Cluster Analysis and geometry analysis
# --------------------------------------------------

class ClusterAnalysis:
    def __init__(self, xBoxLen, yBoxLen, zBoxLen, length=0):
        self.xBoxLen = xBoxLen
        self.yBoxLen = yBoxLen
        self.zBoxLen = zBoxLen
        self.length = length

    def fit(self, data_dict):
        '''
        ClusterAnalysis.fit is a member method to calculate the clusters in
        the dictionary provided.
        The dictionary must contain:
            key = is the the molecule name. It must be a str.
            [atomnum, x, y, z] = it is a list of x, y, and z coordinates.
                        X, y, and z must be numpy arrays.
        '''
        # Verifying the input format is correct
        FileUtil.CheckFuncInput(data_dict, dict, 'ClusterAnalysis.fit')

        print('\nFinding clusters in the dictionary provided in ' +
              'ClusterAnalysis.fit\n')

        cluster = [] # A list of two molecules that are close to each other
        molecs_list = list(data_dict) # getting all the molecules (the dict keys)

        for i, moleci in enumerate(molecs_list):
            for j, molecj in enumerate(molecs_list):

                if (i != j and i < j):
                    # This if statement is to avoid double count
                    # (i.e. once molec1 with molec2 is considered, molec2 with molec1 will be skipped)
                    atomsNumi = len(data_dict[moleci]['atomNum'])
                    atomsNumj = len(data_dict[molecj]['atomNum'])
                    inside = False # the atoms are first not considered close to each other

                    for atomi in range(atomsNumi):
                        xi1 = data_dict[moleci]['X_coord'][atomi]
                        xi2 = data_dict[moleci]['X_coord'][atomi] + self.xBoxLen
                        xi3 = data_dict[moleci]['X_coord'][atomi] - self.xBoxLen
                        xiList = [xi1, xi2, xi3]

                        yi1 = data_dict[moleci]['Y_coord'][atomi]
                        yi2 = data_dict[moleci]['Y_coord'][atomi] + self.yBoxLen
                        yi3 = data_dict[moleci]['Y_coord'][atomi] - self.yBoxLen
                        yiList = [yi1, yi2, yi3]

                        zi1 = data_dict[moleci]['Z_coord'][atomi]
                        zi2 = data_dict[moleci]['Z_coord'][atomi] + self.zBoxLen
                        zi3 = data_dict[moleci]['Z_coord'][atomi] - self.zBoxLen
                        ziList = [zi1, zi2, zi3]

                        for atomj in range(atomsNumj):
                            xj = data_dict[molecj]['X_coord'][atomj]
                            yj = data_dict[molecj]['Y_coord'][atomj]
                            zj = data_dict[molecj]['Z_coord'][atomj]

                            # Calculating the eucledean distance considering
                            # the minimum image convention
                            for xi in xiList:
                                for yi in yiList:
                                    for zi in ziList:
                                        dist = np.linalg.norm([xi-xj, yi-yj, zi-zj])

                                        if dist < self.length:
                                            # append the pair of molecs that are close, we append to the cluster list
                                            cluster.append([moleci, molecj])
                                            inside = True
                                            # If the atoms are next to each other.
                                            # There is no need to look for more atoms in the same molecules.
                                            # We move directly to another molecj

                                        if inside:
                                            break # break from ziList
                                    if inside:
                                        break # break from yiList
                                if inside:
                                    break # break from xiList
                            if inside:
                                break # break from range(atomsNumj)
                                # no need to look for more atomj in the same molecj
                        if inside:
                            break # break from range(atomsNumi)
                            # no need to look for more atomi in the same moleci
                            # We will now look for another molecj and analyze from the first atomi.

        # Now we will group the molecs that have repeated molecs in the pairs list
        # i.e.: [molec1, molec2] and [molec2, molec3], the new group is [molec1, molec2, molec3]
        new_cluster = [] # this is the list where the clusters will be saved
        empty = False

        while not empty:

            if len(cluster) < 1: # if there is no pair left in the cluster list
                empty = True

            if len(cluster) == 1: # if there is one pair left in the cluster list
                new_cluster.append(cluster[0])
                empty = True

            if len(cluster) == 2: # if there are two pairs left in the cluster list
                if (set(cluster[0]) & set(cluster[1])): # check if they have repeated molecs
                    cluster[0] = cluster[0] + cluster[1] # combine the two pairs
                    cluster[0] = sorted(list(set(cluster[0]))) # eliminate repetead molecs.
                    new_cluster.append(cluster[0])

                else: # if no repetead molecs in the two pairs, then append both.
                    new_cluster.append(cluster[0])
                    new_cluster.append(cluster[1])
                empty = True

            if len(cluster) > 2: # if there are more than two pairs left in the cluster list
                add_element = True

                while add_element:
                    count = 0

                    for i in range(0, len(cluster)):
                        count += 1 # This is to know in what position we are analyzing in the cluster list. It is equal to j.
                        j = i + 1

                        try: # we will always compare with the first element if the cluster list
                            if (set(cluster[0]) & set(cluster[j])): # the pairs have common molecs
                                cluster[0] = cluster[0] + cluster[j] # combine the two pairs
                                cluster[0] = sorted(list(set(cluster[0]))) # eliminate repetead molecs.
                                break # break from the for loop.
                                #The new cluster[0] must be compared with the entire cluster list from start.

                        except: # This is just in case j is higher than len(cluster)
                            pass # this is so the program does not get terminated.

                    if count >= len(cluster): # if I reach to the end of cluster list
                        add_element = False # change to false to break from if len(cluster) > 2
                        # This means that the molecs in cluster[0] are uniques, because we reach the end of the cluster list.

                    else: # remove the pair added to the new_cluster list
                        cluster.remove(cluster[j])
                        # This new cluster[0] must be analized from cluster[1] element again.
                        # that is why add_element stays as False and  count=0 again.
                        # We will do this until cluster[j] is the last element to compare with cluster[0]

                new_cluster.append(cluster[0]) # Save the unique cluster[0]. All its molecs are not in cluster list
                cluster.remove(cluster[0]) # cluster[0] elements are in a unique new cluster.
                                            # We remove it from cluster list to start a new analysis.

        print('''\nGetting the molecules not in a clusters in the dictionary
              provided in ClusterAnalysis.fit\n''')

        flatten_new_cluster = functools.reduce(operator.iconcat, new_cluster, [])
        no_cluster = list(set(flatten_new_cluster).symmetric_difference(molecs_list))

        # Creating the cluster dictionary
        clusters = {}
        for index, cluster in enumerate(new_cluster):
            key = 'Cluster_' + str(index)
            clusters[key] = cluster

        clusters['no_cluster'] = no_cluster # molecs in no cluster

        # Removing clusters that are made by 2 oils only or one HP and one oil
        cluster_to_del = []
        for key, cluster in clusters.items():

            if len(cluster) == 2: # if there is cluster with only 2 molecules
                onlyOils = [molec for molec in cluster if molec.find('oil') >= 0] # counting the oils

                if len(onlyOils) >= 1: # if this is a cluster of 2 oils only, or an HP and an oil, then it is not a cluster
                    clusters['no_cluster'].extend(cluster) # append this cluster in the no_cluster list
                    cluster_to_del.append(key)

        for key in cluster_to_del:
            clusters.pop(key) # remove this cluster from clusters dictionary

        self.clusters = clusters
        self.molecules = data_dict # saving all the molecules inside this class

        return None


    def GetCentroids(self):
        '''
        ClusterAnalysis.getCentroids is a member method to calculate the
        center of geometry of the clusters in the grofile.
        This function will calculate the centroids of every cluster considering
        periodic boundary conditions.
        '''

        print('''\nUsing ClusterAnalysis.getCentroids to get the clusters'
              centroids from the clusters dictionary\n''')

        def fixPBC(self, a_cluster, a_XYZ_Cog, a_XYZ_boxLen, a_XYZ_str):
            '''
            GetCentroids.fixPBC() is a function that fix the X, Y or Z coord
            considering the preiodic boundary condition.
            a_cluster: list of molecules in the cluster
            a_XYZ_cog: initial COG coord of the cluster
            a_XYZ_boxLen: box length in the respective coordinate
            a_XYZ_str: it can be 'X_Coord', 'Y_Coord', or 'Z_Coord', depending
            on the coordinate. It is a key inside each cluster of the clusters
            dictionary.
            '''

            while True:
                xyz_sum_all = 0
                atom_sum = 0

                for m, molec in enumerate(a_cluster):
                    atomPerMolec = len(self.molecules[molec]['atomNum'])

                    for atom in range(atomPerMolec):
                        XYZ_i = self.molecules[molec][a_XYZ_str][atom]

                        coord1 = XYZ_i
                        coord2 = XYZ_i + a_XYZ_boxLen
                        coord3 = XYZ_i - a_XYZ_boxLen

                        coord_dict = {}
                        coord_dict[coord1] = abs(coord1 - a_XYZ_Cog)
                        coord_dict[coord2] = abs(coord2 - a_XYZ_Cog)
                        coord_dict[coord3] = abs(coord3 - a_XYZ_Cog)

                        coord_dict1 = {k: v for k, v in sorted(coord_dict.items(),
                                                                key=lambda item: item[1])}

                        XYZ_smallesti = next(iter(coord_dict1))
                        self.molecules[molec][a_XYZ_str][atom] = XYZ_smallesti

                    xyz_coords = sum(self.molecules[molec][a_XYZ_str])
                    xyz_sum_all += xyz_coords
                    atom_sum += atomPerMolec

                xyz_cg_temp = xyz_sum_all / atom_sum

                if (xyz_cg_temp == a_XYZ_Cog):
                    break

                else:
                    a_XYZ_Cog = xyz_cg_temp

            return None


        # Calculating the initial center of geometry of all clusters
        # ---------------------------------------------------------
        print('\nUsing ClusterAnalysis.GetCentroids() to calculate the ' +
              'initial COG of each cluster in the system\n')

        centroid_initial = {}

        for key, cluster in self.clusters.items():

            if key == 'no_cluster':
                continue

            else:
                # for the sum of the coords of all molecules
                x_sum_all = 0
                y_sum_all = 0
                z_sum_all = 0
                atom_sum = 0

                for molec in cluster:
                    x_coords = sum(self.molecules[molec]['X_coord'])
                    y_coords = sum(self.molecules[molec]['Y_coord'])
                    z_coords = sum(self.molecules[molec]['Z_coord'])
                    atonmum = len(self.molecules[molec]['atomNum'])

                    x_sum_all += x_coords
                    y_sum_all += y_coords
                    z_sum_all += z_coords
                    atom_sum += atonmum

                x_cg_ini = x_sum_all / atom_sum
                y_cg_ini = y_sum_all / atom_sum
                z_cg_ini = z_sum_all / atom_sum

                centroid_initial[key] = {'numberMolecules': len(cluster),
                                          'X_coord': x_cg_ini,
                                          'Y_coord': y_cg_ini,
                                          'Z_coord': z_cg_ini}

        # Correcting the XYZ coord of all atoms consodering PBC
        # -----------------------------------------------------
        print('\nUsing ClusterAnalysis.GetCentroids() to correct the coords ' +
              'of each atoms in the system that is part of a cluster using ' +
              'PBC criterion.\n')

        for key, cluster in self.clusters.items():

            if key == 'no_cluster':
                continue

            else:
                x_cog = centroid_initial[key]['X_coord']
                y_cog = centroid_initial[key]['Y_coord']
                z_cog = centroid_initial[key]['Z_coord']

                fixPBC(self, cluster, x_cog, self.xBoxLen, 'X_coord')
                fixPBC(self, cluster, y_cog, self.yBoxLen, 'Y_coord')
                fixPBC(self, cluster, z_cog, self.zBoxLen, 'Z_coord')

        # Calculating the centroids of all, HPs and oils independently
        # ------------------------------------------------------------
        print('\nUsing ClusterAnalysis.GetCentroids() to calculate the ' +
              'final COG of each cluster in the system\n')

        all_centroids = {} # COG of all molecs wrt COG of each HP
        hps_centroids = {} # COG of HPs only wrt COG of each HP
        oils_centroids = {} # COG of oils only wrt COG of each oil

        all_molecNum = {} # for later use to calculate mass percent content
        hps_molecNum = {} # for later use to calculate mass percent content
        oils_molecNum = {} # for later use to calculate mass percent content

        for key, cluster in self.clusters.items():
            if key == 'no_cluster':
                continue

            else:
                # for the sum of the coords of all molecules
                x_sum_all = 0
                y_sum_all = 0
                z_sum_all = 0

                # for the sum of the coords of for HPs only
                x_sum_hps = 0
                y_sum_hps = 0
                z_sum_hps = 0

                # for the sum of the coords of for oils only
                x_sum_oils = 0
                y_sum_oils = 0
                z_sum_oils = 0

                #to get the total atoms present in the cluster (all, HPs, and oils)
                atoms_all = 0
                atoms_hps = 0
                atoms_oils = 0

                #to get the total molecules present in the cluster (all, HPs, and oils)
                molecs_all = 0
                molecs_hps = 0
                molecs_oils = 0

                for molec in cluster:
                    # Getting the coords sum of all atom in the molecules
                    x_coords = sum(self.molecules[molec]['X_coord'])
                    y_coords = sum(self.molecules[molec]['Y_coord'])
                    z_coords = sum(self.molecules[molec]['Z_coord'])

                    # Getting the number of atoms in the molecule
                    # We can use X, Y or Z. Here I used X.
                    # There is one X coordinate per atom.
                    atom_num = len(self.molecules[molec]['X_coord'])

                    if molec.find('HP') >= 0 or molec.find('oil') >= 0: # for all molecules
                        x_sum_all += x_coords
                        y_sum_all += y_coords
                        z_sum_all += z_coords
                        atoms_all += atom_num
                        molecs_all += 1

                    if molec.find('HP') >= 0: # for HPs only
                        x_sum_hps += x_coords
                        y_sum_hps += y_coords
                        z_sum_hps += z_coords
                        atoms_hps += atom_num
                        molecs_hps += 1

                    elif molec.find('oil') >= 0: # For oils only
                        x_sum_oils += x_coords
                        y_sum_oils += y_coords
                        z_sum_oils += z_coords
                        atoms_oils += atom_num
                        molecs_oils += 1

                if atoms_all != 0:
                    # Getting the Center of geometries for all molecules in the cluster
                    x_mean_all = x_sum_all / atoms_all
                    y_mean_all = y_sum_all / atoms_all
                    z_mean_all = z_sum_all / atoms_all
                    all_centroids[key] = {'numberMolecules': molecs_all,
                                          'X_coord': x_mean_all,
                                          'Y_coord': y_mean_all,
                                          'Z_coord': z_mean_all}
                    all_molecNum[key] = molecs_all

                else:
                    # In case atoms_all == 0, we set dummy values.
                    # Division by ero is not possible.
                    # This else is when there are no atoms or molecules in the cluster.
                    all_centroids[key] = {'numberMolecules': np.nan,
                                          'X_coord': np.nan,
                                          'Y_coord': np.nan,
                                          'Z_coord': np.nan}
                    all_molecNum[key] = np.nan

                if atoms_hps != 0:
                    # Getting the Center of geometries for HPs only in the cluster
                    x_mean_hps = x_sum_hps / atoms_hps
                    y_mean_hps = y_sum_hps / atoms_hps
                    z_mean_hps = z_sum_hps / atoms_hps
                    hps_centroids[key] = {'numberMolecules': molecs_hps,
                                          'X_coord': x_mean_hps,
                                          'Y_coord': y_mean_hps,
                                          'Z_coord': z_mean_hps}
                    hps_molecNum[key] = molecs_hps

                else:
                    # In case atoms_all == 0, we set dummy values.
                    # Division by ero is not possible.
                    # This else is when there are NO HPs in the cluster.
                    hps_centroids[key] = {'numberMolecules': np.nan,
                                          'X_coord': np.nan,
                                          'Y_coord': np.nan,
                                          'Z_coord': np.nan}
                    hps_molecNum[key] = np.nan

                if atoms_oils != 0:
                    # Getting the Center of geometries for oils Only in the cluster
                    x_mean_oils = x_sum_oils / atoms_oils
                    y_mean_oils = y_sum_oils / atoms_oils
                    z_mean_oils = z_sum_oils / atoms_oils
                    oils_centroids[key] = {'numberMolecules': molecs_oils,
                                           'X_coord': x_mean_oils,
                                           'Y_coord': y_mean_oils,
                                           'Z_coord': z_mean_oils}
                    oils_molecNum[key] = molecs_oils

                else:
                    # In case atoms_all == 0, we set dummy values.
                    # Division by ero is not possible.
                    # This else is when there are NO OILS in the cluster.
                    oils_centroids[key] = {'numberMolecules': np.nan,
                                          'X_coord': np.nan,
                                          'Y_coord': np.nan,
                                          'Z_coord': np.nan}
                    oils_molecNum[key] = np.nan

            print('''\nGetting the clusters' centroids for HPs+oils, HPs only,
                  and oils only\n''')

            self.centroids = {'all': all_centroids,
                              'HPs': hps_centroids,
                              'oils': oils_centroids}

            self.molecPerCluster = {'all_molecNum': all_molecNum,
                                    'hps_molecNum': hps_molecNum,
                                    'oils_molecNum': oils_molecNum}

        return None


    def GetGeometryDescriptionParams(self):
        '''
        ClusterAnalysis.getGeometryDescriptionParams is a member method to
        calculate the clusters' eigenvalues, ShapeDesciptor, Asphericity, and
        acylindricity.
        '''

        print('''\nUsing ClusterAnalysis.getGeometryDescriptionParams to get
              the eigenvalues, ShapeDesciptor, Asphericity, and acylindricity
              of all clusters present in the grofile.\n''')

        # Creating the dictionaries where all values of each param will be stored.
        eigenvalues1 = {'all_ev1': {}, 'HPs_ev1': {}, 'oils_ev1': {}}
        eigenvalues2 = {'all_ev2': {}, 'HPs_ev2': {}, 'oils_ev2': {}}
        eigenvalues3 = {'all_ev3': {}, 'HPs_ev3': {}, 'oils_ev3': {}}
        shapeDescrip = {'all_shapeD': {}, 'HPs_shapeD': {}, 'oils_shapeD': {}}
        asphericity = {'all_asp': {}, 'HPs_asp': {}, 'oils_asp': {}}
        acylindricity = {'all_acy': {}, 'HPs_acy': {}, 'oils_acy': {}}
        rog = {'all_rog': {}, 'HPs_rog': {}, 'oils_rog': {}} # Radius of gyration

        for typeCentorid, centroid in self.centroids.items():
            # typeCentorid is 'all', 'HPs', and 'oils' keys
            # centroid is all_centroids, hps_centroids, and oils_centroids dictionaries

            print('\nCalculating the eigenvalues, ShapeDesciptor,' +
                  ' Asphericity, acylindricity \nof all clusters ' +
                  'using {} molecules.\n'. format(typeCentorid))

            for clusterNum, cluster in self.clusters.items():
                # clusterNum is 'Cluster_0', 'Cluster_1', .... keys
                # cluster is the list of molecules that form the cluster
                if clusterNum == 'no_cluster':
                    continue

                else:
                    # Geeting the center of geometry of the specific cluster(i.e. 'Cluster_0')
                    # i.e. centroid[Cluster_0][X_coord] could be the same as centroids[all]['Cluster_0'][X_coord]
                    Xcm = centroid[clusterNum]['X_coord']
                    Ycm = centroid[clusterNum]['Y_coord']
                    Zcm = centroid[clusterNum]['Z_coord']

                    if not math.isnan(Xcm): # if it exists
                        # Cuadratic terms
                        Xi_Xcm_2 = 0
                        Yi_Ycm_2 = 0
                        Zi_Zcm_2 = 0
                        # Cross terms
                        Xi_Xcm_Yi_Ycm = 0
                        Xi_Xcm_Zi_Zcm = 0
                        Yi_Ycm_Zi_Zcm = 0

                        if typeCentorid == 'all' or typeCentorid == 'HPs':
                            molecules = [molec for molec in cluster if molec.find('HP') >= 0]

                        elif typeCentorid == 'oils':
                           molecules = [molec for molec in cluster if molec.find('oil') >= 0]

                        for molec in molecules:
                            # molec is the specific element inside the cluster list
                            atom_num = len(self.molecules[molec]['X_coord'])
                            xi_cog = sum(self.molecules[molec]['X_coord']) / atom_num
                            yi_cog = sum(self.molecules[molec]['Y_coord']) / atom_num
                            zi_cog = sum(self.molecules[molec]['Z_coord']) / atom_num

                            Xi_Xcm = xi_cog - Xcm
                            Yi_Ycm = yi_cog - Ycm
                            Zi_Zcm = zi_cog - Zcm

                            # Cuadratic terms
                            Xi_Xcm_2 += Xi_Xcm**2
                            Yi_Ycm_2 += Yi_Ycm**2
                            Zi_Zcm_2 += Zi_Zcm**2
                            # Cross terms
                            Xi_Xcm_Yi_Ycm += Xi_Xcm * Yi_Ycm
                            Xi_Xcm_Zi_Zcm += Xi_Xcm * Zi_Zcm
                            Yi_Ycm_Zi_Zcm += Yi_Ycm * Zi_Zcm

                        # Applying the tensor Matrix Equation
                        # Q = matrix( [ [A, K, G],[B, E, H],[J, F, M] ] )
                        particleNum = len(molecules)

                        if particleNum > 0:

                            # First row:
                            A = Xi_Xcm_2 / particleNum
                            K = Xi_Xcm_Yi_Ycm / particleNum
                            G = Xi_Xcm_Zi_Zcm / particleNum

                            # Second row:
                            B = Xi_Xcm_Yi_Ycm / particleNum
                            E = Yi_Ycm_2 / particleNum
                            H = Yi_Ycm_Zi_Zcm / particleNum

                            # Third row:
                            J = Xi_Xcm_Zi_Zcm / particleNum
                            F = Yi_Ycm_Zi_Zcm / particleNum
                            M = Zi_Zcm_2 / particleNum

                            # Getting det(Q)=0, which gives the cubic equation
                            # Ax^3+Bx^2+Cx+D
                            a1 = 1
                            a2 = (-A-E-M)
                            a3 = (M*E)+(A*M)+(A*E)-(K*B)-(F*H)-(G*J)
                            a4 = -(A*M*E)+(M*K*B)+(A*F*H)+(E*G*J)+(K*H*J)+(G*B*F)

                            # Getting the eigenvalues
                            coeff = [a1, a2, a3, a4]
                            eigenVal = np.roots(coeff)
                            e1 = eigenVal[0]
                            e2 = eigenVal[1]
                            e3 = eigenVal[2]
                            eigenvalues1[typeCentorid + '_ev1'][clusterNum] = e1
                            eigenvalues2[typeCentorid + '_ev2'][clusterNum] = e2
                            eigenvalues3[typeCentorid + '_ev3'][clusterNum] = e3

                            # Calculating Radius of Gyration (RoG)
                            RadiusOfGyr = (e1 + e2 + e3)**(1/2)
                            rog[typeCentorid + '_rog'][clusterNum] = RadiusOfGyr

                            # Calculating shape descriptor or shape anisotropy (k^2)
                            shapeD = 1 - 3 * (e1*e2 + e2*e3 + e3*e1)/((e1+e2+e3)**(2))
                            shapeDescrip[typeCentorid + '_shapeD'][clusterNum] = shapeD

                            # Calculating Asphericity (b)
                            Asp = (e1 - 0.5 * (e2 + e3)) / RadiusOfGyr
                            asphericity[typeCentorid + '_asp'][clusterNum] = Asp

                            # Calculating Acylindrity (c)
                            Acy = (e2 - e3) / RadiusOfGyr
                            acylindricity[typeCentorid + '_acy'][clusterNum] = Acy

                        else:
                            # In case the cluster is empty (meaning no HPs or no oils)
                            eigenvalues1[typeCentorid + '_ev1'][clusterNum] = np.nan
                            eigenvalues2[typeCentorid + '_ev2'][clusterNum] = np.nan
                            eigenvalues3[typeCentorid + '_ev3'][clusterNum] = np.nan
                            rog[typeCentorid + '_rog'][clusterNum] = np.nan
                            shapeDescrip[typeCentorid + '_shapeD'][clusterNum] = np.nan
                            asphericity[typeCentorid + '_asp'][clusterNum] = np.nan
                            acylindricity[typeCentorid + '_acy'][clusterNum] = np.nan

                    else:
                        # In case the cluster is empty (meaning no HPs or no oils)
                        eigenvalues1[typeCentorid + '_ev1'][clusterNum] = np.nan
                        eigenvalues2[typeCentorid + '_ev2'][clusterNum] = np.nan
                        eigenvalues3[typeCentorid + '_ev3'][clusterNum] = np.nan
                        rog[typeCentorid + '_rog'][clusterNum] = np.nan
                        shapeDescrip[typeCentorid + '_shapeD'][clusterNum] = np.nan
                        asphericity[typeCentorid + '_asp'][clusterNum] = np.nan
                        acylindricity[typeCentorid + '_acy'][clusterNum] = np.nan

        self.eigenvalues1 = eigenvalues1
        self.eigenvalues2 = eigenvalues2
        self.eigenvalues3 = eigenvalues3
        self.shapeDescrip = shapeDescrip
        self.asphericity = asphericity
        self.acylindricity = acylindricity
        self.rog = rog # Radius of gyration

        return None


    def Mass_percent_calc(self, a_hpWeight, a_oilWeight, a_oilsTotSys,
                          a_HPsTotSys):
        '''
        ClusterAnalysis.Mass_percent_calc is a member method to calculate the
        mass percent of HPsOnly, oilsOnly, and HPs+oils in all clusters.
        '''

        print('''\nUsing ClusterAnalysis.Mass_percent_calc to calculate the
              mass percent of HPsOnly, oilsOnly, and HPs+oils in all
              clusters.\n''')

        HPsWeightSys = a_HPsTotSys * a_hpWeight
        oilsWeightSys = a_oilsTotSys * a_oilWeight

        mass_percent = {'all_mass': {}, 'HPs_mass': {}, 'oils_mass': {}}

        for clusterNum, cluster in self.clusters.items():
            hpCount = 0
            oilCount = 0

            for molec in cluster:
                if molec.find('HP') >= 0:
                    hpCount += 1

                if molec.find('oil') >= 0:
                    oilCount += 1

            HPsWeight = hpCount * a_hpWeight
            oilsWeight = oilCount * a_oilWeight

            HpPercent = HPsWeight / HPsWeightSys
            oilPercent = oilsWeight / oilsWeightSys
            allWeight = (HPsWeight + oilsWeight) / (HPsWeightSys + oilsWeightSys)

            mass_percent['all_mass'][clusterNum] = allWeight
            mass_percent['HPs_mass'][clusterNum] = HpPercent
            mass_percent['oils_mass'][clusterNum] = oilPercent

        self.mass_percent = mass_percent

        return None


    def __del__(self):
        print("\nFreeing memory by deleting ClusterAnalysis object\n")
        # This is my destructor (it is not mandatory for python)


# ============================================================================
# ============================================================================
# DRIVER PROJECT SECTION
# ============================================================================
# ============================================================================

def run_all_script(a_groFileName):
    '''
    run_all_script is a function that does all the cluster analysis for
    each grofiles. In multiprocessin, this function will be used.
    It requires only the grofileName.
    '''
    print('\n\n##############################################')
    print('         ' + a_groFileName)
    print('##############################################')

    gro = MoleculesInfo.fromList_GROreader(a_groFileName, itpFileName, cwd)
    gro.Caculate_molecs_weight(itpFileName, martiniItpName, cwd)
    xBoxLen, yBoxLen, zBoxLen = gro.xBoxLen, gro.yBoxLen, gro.zBoxLen

    allMolecDict = {**gro.HPs, **gro.oils}

    groCA = ClusterAnalysis(xBoxLen, yBoxLen, zBoxLen, length=cutoff)
    groCA.fit(allMolecDict)
    del(allMolecDict)
    groCA.GetCentroids()
    groCA.GetGeometryDescriptionParams()

    hpWeight, oilWeight = gro.hpWeight, gro.oilWeight
    HPsTotSys, oilsTotSys = len(gro.HPs), len(gro.oils)
    groCA.Mass_percent_calc(hpWeight, oilWeight, oilsTotSys, HPsTotSys)
    del(gro) # freeing memory gro object is not needed

    variables = [groCA.eigenvalues1, groCA.eigenvalues2,
                  groCA.eigenvalues3, groCA.shapeDescrip, groCA.rog,
                  groCA.asphericity, groCA.acylindricity,
                  groCA.mass_percent, groCA.molecPerCluster]
    del(groCA) # freeing memory becuase the info needed is already in the list

    data = pd.DataFrame([])
    for var in variables:
        df = pd.DataFrame(var).transpose()
        emptyRow = pd.Series([np.nan for i in range(len(df.columns))],
                              index=df.columns,
                              name= ' ').transpose()
        df = df.append(emptyRow)
        data = pd.concat([data, df], sort=False)

    data.drop('no_cluster', axis=1, inplace=True)
    # data['Mean_values'] = data.mean(axis=1)

    return data

# ----------------------------------------------------------------------------
# Using argparse to read input data from the commanline
# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gro', nargs='+',
                    default=[file for file in os.listdir(os.getcwd()) if file.find('.gro') >= 0],
                    help='grofile list input (example: system.gro)', type=str)

parser.add_argument('-p', '--protein', default='Protein_A.itp',
                    help='protein itp topology file input (default: Protein_A.itp)',
                    type=str)

parser.add_argument('-m', '--martini', default='martini_v2.2P.itp',
                    help='martini topology file input (default: martini_v2.2P.itp)',
                    type=str)

parser.add_argument('-f', '--folder', default='cluster_analysis',
                    help='folder to save excel file with cluster information (default: cluster_analysis)',
                    type=str)

parser.add_argument('-o', '--output', required=True,
                    help='name of the excel file with all cluster information (example: out.xlsx)',
                    type=str)

parser.add_argument('-c', '--cutoff', default=0.55,
                    help='cutoff value to determine neighbors (default: 0.55 nm)',
                    type=float)

parser.add_argument('-i', '--plot', default='out',
                    help='base name for plot (all are in png format) (default: out)',
                    type=str)

parser.add_argument('-b', '--begin', required=True,
                    help='simulation time of the first gro frame file (units in ns)',
                    type=float)

parser.add_argument('-e', '--end', required=True,
                    help='simulation time of the last gro frame file (units in ns)',
                    type=float)

args = parser.parse_args()

# ----------------------------------------------------------------------------
# Input variables taken from argparse
# ----------------------------------------------------------------------------

grofiles = sorted(args.gro, key=lambda x: int("".join([i for i in x if i.isdigit()])))
itpFileName = args.protein
martiniItpName = args.martini
clusterFolder = args.folder
cutoff = args.cutoff
excelFileName = args.output
pngBaseName = args.plot
firstFrame = args.begin
lastFrame = args.end

# grofiles = ['trial_box1.gro', 'trial_box4.gro']
# itpFileName = 'Protein_A.itp'
# martiniItpName = 'martini_v2.2P.itp'
# clusterFolder = 'cluster_analysis'
# cutoff = 0.55
# excelFileName = 'out.xlsx'
# pngBaseName = 'out'

cwd = os.getcwd()
clusterPath = os.path.join(cwd, clusterFolder)
excelPath = os.path.join(clusterPath, excelFileName)
dataframeList = []

# ----------------------------------------------------------------------------
# Running the program in multiprocessing
# ----------------------------------------------------------------------------

try:
    if not clusterPath:
        raise FileNotFoundError('{} did not found in path {}'.format(
            clusterFolder, cwd))

    os.chdir(clusterPath)
    print('\n\nThe gro files to use are: \n')

    for grofile in grofiles:
        print(grofile)

    with concurrent.futures.ProcessPoolExecutor() as executor: # Doing multiprocessing
        results = executor.map(run_all_script, grofiles) # evaluation the function for all grofiles

        for result in results: # saving the results. Dataframes in this case.
            dataframeList.append(result)

except Exception:
    print('\n\nCaught an error:\n----------------')
    traceback.print_exc()

# ----------------------------------------------------------------------------
# Rearanging the dataframes according to the clusters
# ----------------------------------------------------------------------------

# Getting all the columns in each dataframe
cols_all = []
for df in dataframeList:
    columns = list(df.columns)
    cols_all.extend(columns)

# Eliminating repeated items and selecting the ones that belong to clusters
cols_all = list(set(cols_all)) # cols_all = ['Cluster_0', 'Cluster_1', ....]
cols_all = [col for col in cols_all if col.find('Cluster') >= 0] # making sure there are only 'Cluster_x' in the list
cols_all = sorted(cols_all, key=lambda x: int("".join([i for i in x if i.isdigit()]))) # sorting in order

# Getting the final df list that will be stored in the excel file.
final_dflist = []

for col in cols_all:
    df_col = pd.DataFrame([])

    for i, df in enumerate(dataframeList):
        temp = pd.DataFrame([])
        name = grofiles[i][:-4] # the name of the column is the gro frame name

        try:
            temp[name] = df[col] # cpying the cluster info but using the grofile name for the specific gro frame
            df_col = pd.concat([df_col, temp], sort=False, axis=1) # concat all frames info for the specific cluster

        except: # there are some grofiles that do not have the same clusters
            pass # pass them to avoid errors

    final_dflist.append(df_col)

# ----------------------------------------------------------------------------
# Saving the data in an excel file
# ----------------------------------------------------------------------------

print('\n\n##############################################')
print('     Saving all data into an excel file')
print('##############################################\n\n')

if os.path.isfile(excelPath):
    os.remove(excelPath)

print("Saving all gro clusters' data into an excel file\n\n")
writer = pd.ExcelWriter(excelPath, engine = 'xlsxwriter')

for index, dataframe in enumerate(final_dflist):
    dataframe['Mean_values'] = dataframe.mean(axis=1)
    dataframe['standard_deviation'] = dataframe.std(axis=1)
    name = cols_all[index] # all the info in each df belongs to a 'Cluster_x'
    print('Saving {} info in the excel file\n\n'.format(name))
    dataframe = dataframe.transpose()
    dataframe.to_excel(writer, sheet_name=name)

print('Excel file is ready !!!!\n\n')
print('The excel file name is {}, located in {}\n'. format(
    excelFileName, excelPath))
writer.save()
writer.close()

# ----------------------------------------------------------------------------
# Plotting the data for each cluster (varaibles vs simulation_time)
# ----------------------------------------------------------------------------

columns = list(final_dflist[0].columns) # all dataframes have the same columns
ns_per_frame = (lastFrame - firstFrame) / len(grofiles) # GMX extract frames in ns
rename = {} # dict to rename the columns according to frame time

for index, col in enumerate(columns):
    if col == 'Mean_values' or col == 'standard_deviation':
        continue

    else:
        rename[col] = index * ns_per_frame

yLabels = {'all_shapeD': 'Shape Descriptor Blob COG vs. each HP COG',
            'HPs_shapeD': 'Shape Descriptor HPs COG vs. each HP COG',
            'oils_shapeD': 'Shape Descriptor oils COG vs. each oil COG',
            'all_rog': 'Radius of Gyration of Blob',
            'HPs_rog': 'Radius of Gyration of HPs',
            'oils_rog': 'Radius of Gyration of oils',
            'all_asp': 'Asphericity Blob COG vs. each HP COG',
            'HPs_asp': 'Asphericity HPs COG vs. each HP COG',
            'oils_asp': 'Asphericity oils COG vs. each oil COG',
            'all_acy': 'Acylindricity Blob COG vs. each HP COG',
            'HPs_acy': 'Acylindricity HPs COG vs. each HP COG',
            'oils_acy': 'Acylindricity oils COG vs. each oil COG',
            'all_mass': 'Mass percent of HPs and oils in blob',
            'HPs_mass': 'Mass percent of HPs in blob',
            'oils_mass': 'Mass percent of oils in blob'}


for index, dataframe in enumerate(final_dflist):
    dataframe.dropna(inplace=True)
    dataframe.drop(['Mean_values', 'standard_deviation'], axis=1, inplace=True)
    dataframe.rename(columns=rename, inplace=True)
    dataframe = dataframe.transpose()

    for key, ylabel in yLabels.items():
        try: # there may be some keys that do not exist in the dataframe
            fig, ax = plt.subplots()
            dataframe.plot(kind='line', y=key, use_index=True, style='ro-', ax=ax)
            ax.set_ylabel(ylabel + ' in ' + cols_all[index])
            ax.set_xlabel('time (ns)')
            pngName = pngBaseName + '_' + cols_all[index] + '_' + key + '.png'
            pngPath = os.path.join(clusterPath, pngName)
            plt.savefig(pngPath, dpi=900)

        except:
            pass



# color = ['cyan', 'green', 'blue', 'black', 'orange', 'red', 'magenta', 'red', 'red', 'red', 'red']
# color2 = ['black', 'black', 'black', 'red', 'red']
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for molec, coords in groCA.molecules.items():
#     if molec.find('HP') >= 0:
#         ax.scatter(coords['X_coord'], coords['Y_coord'], coords['Z_coord'], c=color[0])

#     if molec.find('oil') >= 0:
#         ax.scatter(coords['X_coord'], coords['Y_coord'], coords['Z_coord'], c=color[1])
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# i = 0
# for key, cluster in groCA.clusters.items():
#     for molec in cluster:
#         # if molec.find('HP') >= 0:
#         ax.scatter(groCA.molecules[molec]['X_coord'], groCA.molecules[molec]['Y_coord'], groCA.molecules[molec]['Z_coord'], c=color[i])
#         # elif molec.find('oil') >= 0:
#             # ax.scatter(groCA.molecules[molec][1], groCA.molecules[molec][2], groCA.molecules[molec][3], c=color2[i])
#     i += 1

# plt.show()







