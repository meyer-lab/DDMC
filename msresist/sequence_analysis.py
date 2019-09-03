"""Sequence Analysis Functions. """

import os
import re
import numpy as np
import pandas as pd
from Bio import SeqIO

path = os.path.dirname(os.path.abspath(__file__))
###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###

def pYmotifs(ABC, ABC_names):
    directory = os.path.join(path, "./data/Sequence_analysis/")
    names, motifs, pXpos = GeneratingKinaseMotifs(ABC_names, FormatSeq(ABC), directory + "proteome_uniprot.fa")
    ABC['Master Protein Descriptions'] = names
    ABC['peptide-phosphosite'] = motifs
    ABC.insert(12, 'position', pXpos)
    return ABC


def FormatName(X):
    """ Keep only the general protein name, without any other accession information """
    names = []
    list(map(lambda v: names.append(v.split("OS")[0]), X.iloc[:, 1]))
    return names


def FormatSeq(X):
    """ Deleting -1/-2 for mapping to uniprot's proteome"""
    seqs = []
    list(map(lambda v: seqs.append(v.split("-")[0]), X.iloc[:, 0]))
    return seqs


def DictProteomeNameToSeq(X):
    """ To generate proteom's dictionary """
    DictProtToSeq_UP = {}
    for rec2 in SeqIO.parse(X, "fasta"):
        UP_seq = str(rec2.seq)
        UP_name = rec2.description.split("HUMAN ")[1].split(" OS")[0]
        DictProtToSeq_UP[UP_name] = str(UP_seq)
    return DictProtToSeq_UP


def getKeysByValue(dictOfElements, valueToFind):
    """ Find the key of a given value within a dictionary. """
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if valueToFind in item[1]:
            listOfKeys.append(item[0])
    return listOfKeys


def MatchProtNames(ProteomeDict, MS_names, MS_seqs):
    """ Match protein names of MS and Uniprot's proteome. """
    matchedNames = []
    for i, MS_seq in enumerate(MS_seqs):
        MS_seqU = MS_seq.upper()
        MS_name = MS_names[i].strip()
        if MS_name in ProteomeDict and MS_seqU in ProteomeDict[MS_name]:
            matchedNames.append(MS_name)
        else:
            matchedNames.append(getKeysByValue(ProteomeDict, MS_seqU)[0])
    return matchedNames

def GeneratingKinaseMotifs(names, seqs, PathToProteome):
    """ Generates phosphopeptide motifs accounting for doubly phospho-peptides. """
    motif_size = 5
    proteome = open(PathToProteome, 'r')
    ProteomeDict = DictProteomeNameToSeq(proteome)
    protnames = MatchProtNames(ProteomeDict, names, seqs)
    MS_names, motifs, uni_pos = [], [], []
    Allseqs, Testseqs = [], []

    for i, MS_seq in enumerate(seqs):
        MS_seqU = MS_seq.upper()
        MS_name = protnames[i]
        try:
            UP_seq = ProteomeDict[MS_name]
            assert MS_seqU in UP_seq, "check " + MS_name + " with seq " + MS_seq
            assert MS_name == list(ProteomeDict.keys())[list(ProteomeDict.values()).index(str(UP_seq))], \
                "check " + MS_name + " with seq " + MS_seq
            Allseqs.append(MS_seq)
            regexPattern = re.compile(MS_seqU)
            MatchObs = list(regexPattern.finditer(UP_seq))
#             assert len(MatchObs) == 1, str(MatchObs) yIEVFk appears twice in HNRPF_HUMAN, only case
            if "y" in MS_seq:
                pY_idx = list(re.compile("y").finditer(MS_seq))
                assert len(pY_idx) != 0
                center_idx = pY_idx[0].start()
                y_idx = center_idx + MatchObs[0].start()
                DoS_idx = None
                if len(pY_idx) > 1:
                    DoS_idx = pY_idx[1:]
                    assert len(DoS_idx) != 0
                elif "t" in MS_seq or "s" in MS_seq:
                    DoS_idx = list(re.compile("y|t|s").finditer(MS_seq))
                    assert len(DoS_idx) != 0
                uni_pos.append("Y" + str(y_idx+1) + "-p")
                MS_names.append(MS_name)
                Testseqs.append(MS_seq)
                motifs.append(makeMotif(UP_seq, MS_seq, motif_size, y_idx, center_idx, DoS_idx))

            if "y" not in MS_seq:
                pTS_idx = list(re.compile("t|s").finditer(MS_seq))
                assert len(pTS_idx) != 0
                center_idx = pTS_idx[0].start()
                ts_idx = center_idx + MatchObs[0].start()
                DoS_idx = None
                if len(pTS_idx) > 1:
                    DoS_idx = pTS_idx[1:]
                uni_pos.append(str(MS_seqU[center_idx]) + str(ts_idx+1) + "-p")
                MS_names.append(MS_name)
                Testseqs.append(MS_seq)
                motifs.append(makeMotif(UP_seq, MS_seq, motif_size, ts_idx, center_idx, DoS_idx=None))

        except BaseException:
            print("find and replace", MS_name, "in proteome_uniprot.txt. Use: ", MS_seq)
            raise


    li_dif = [i for i in Testseqs + Allseqs if i not in Allseqs or i not in Testseqs]
    if li_dif:
        print(" Testseqs vs Allseqs may have different peptide sequences: ", li_dif)

    assert len(names) == len(MS_names), "mapping incosistent number of names" \
        + str(len(names)) + " " + str(len(MS_names))
    assert len(seqs) == len(motifs), "mapping incosistent number of peptides" \
        + str(len(seqs)) + " " + str(len(motifs))
    assert len(uni_pos) == len(seqs), "inconsistent nubmer of pX positions" \
        + str(len(seqs)) + " " + str(len(uni_pos))

    proteome.close()
    return MS_names, motifs, uni_pos


def makeMotif(UP_seq, MS_seq, motif_size, y_idx, center_idx, DoS_idx):
    """ Make a motif out of the matched sequences. """
    UP_seq_copy = list(UP_seq[max(0, y_idx - motif_size):y_idx + motif_size + 1])
    assert len(UP_seq_copy) > motif_size, "Size seems too small. " + UP_seq

    # If we ran off the end of the sequence at the beginning or at the end, append a gap
    if y_idx - motif_size < 0:
        for ii in range(motif_size - y_idx):
            UP_seq_copy.insert(0, "-")

    elif y_idx + motif_size > len(UP_seq):
        for jj in range(y_idx + motif_size - len(UP_seq) + 1):
            UP_seq_copy.extend("-")

    UP_seq_copy[motif_size] = UP_seq_copy[motif_size].lower()

#     Now go through and copy over phosphorylation
    if DoS_idx:
        for ppIDX in DoS_idx:
            position = ppIDX.start() - center_idx
            # If the phosphosite is within the motif
            if abs(position) < motif_size:
                editPos = position + motif_size
                UP_seq_copy[editPos] = UP_seq_copy[editPos].lower()
                assert UP_seq_copy[editPos] == MS_seq[ppIDX.start()], \
                    UP_seq_copy[editPos] + " " + MS_seq[ppIDX.start()]

    return ''.join(UP_seq_copy)
