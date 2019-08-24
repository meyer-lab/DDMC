"""Sequence Analysis Functions. """

import os
import re
import pandas as pd
from Bio import SeqIO

path = os.path.dirname(os.path.abspath(__file__))
###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###


def pYmotifs(ABC_conc_mc, ABC_names):
    ABC_seqs = FormatSeq(ABC_conc_mc)
    ABC_conc_mc["peptide-phosphosite"] = ABC_seqs

    directory = os.path.join(path, "./data/Sequence_analysis/")
    names, motifs = GeneratingKinaseMotifs("FaFile.fa", ABC_names, ABC_seqs, "MatchedFaFile.fa", directory + "proteome_uniprot.fa")
    ABC_conc_mc["peptide-phosphosite"] = motifs
    ABC_conc_mc["Master Protein Descriptions"] = names
    return ABC_conc_mc


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


def GenerateFastaFile(PathToFaFile, MS_names, MS_seqs):
    """ Sequence processor. """
    FileHandle = open(PathToFaFile, "w+")
    for i in range(len(MS_seqs)):
        FileHandle.write(">" + str(MS_names[i]))
        FileHandle.write("\n")
        FileHandle.write(str(MS_seqs[i]))
        FileHandle.write("\n")
    FileHandle.close()


def DictProteomeNameToSeq(X):
    """ Goal: Generate dictionary key: protein name | val: sequence of Uniprot's proteome or any
    large data set where looping is not efficient.
    Input: fasta file.
    Output: Dictionary. """
    DictProtToSeq_UP = {}
    for rec2 in SeqIO.parse(X, "fasta"):
        UP_seq = str(rec2.seq)
        UP_name = rec2.description.split("HUMAN ")[1].split(" OS")[0]
        DictProtToSeq_UP[UP_name] = str(UP_seq)
    return DictProtToSeq_UP


def getKeysByValue(dictOfElements, valueToFind):
    """ Goal: Find the key of a given value within a dictionary.
    Input: Dicitonary and value
    Output: Key of interest"""
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if valueToFind in item[1]:
            listOfKeys.append(item[0])
    return listOfKeys


def MatchProtNames(FaFile, PathToMatchedFaFile, ProteomeDict):
    """ Goal: Match protein names of MS and Uniprot's proteome.
    Input: Path to new file and MS fasta file
    Output: Fasta file with matching protein names.
    Note that ProteomeDict[MS_name] is what the function needs to try to find
    and jump to except if MS_name is not in ProteomDict. """
    FileHandle = open(PathToMatchedFaFile, "w+")
    for rec1 in SeqIO.parse(FaFile, "fasta"):
        MS_seq = str(rec1.seq)
        MS_seqU = str(rec1.seq.upper())
        MS_name = str(rec1.description.split(" OS")[0])
        try:
            ProteomeDict[MS_name]
            FileHandle.write(">" + MS_name)
            FileHandle.write("\n")
            FileHandle.write(MS_seq)
            FileHandle.write("\n")
        except BaseException:
            Fixed_name = getKeysByValue(ProteomeDict, MS_seqU)
            FileHandle.write(">" + Fixed_name[0])
            FileHandle.write("\n")
            FileHandle.write(MS_seq)
            FileHandle.write("\n")
    FileHandle.close()


def GeneratingKinaseMotifs(PathToFaFile, MS_names, MS_seqs, PathToMatchedFaFile, PathToProteome):
    """ Goal: Generate Phosphopeptide motifs.
    Input: Directory paths to fasta file, fasta file with matched names, and proteome
    Output: Protein names list and kinase motif list. Run with def GenerateFastaFile to obtain the final file.
    Kinase motif -5 +5 wrt the phosphorylation site. It accounts for doubly phosphorylated peptides (lowercase y, t, s). """
    counter = 0
    GenerateFastaFile(PathToFaFile, MS_names, MS_seqs)
    FaFile = open(PathToFaFile, "r")
    proteome = open(PathToProteome, "r")
    ProteomeDict = DictProteomeNameToSeq(proteome)
    MatchProtNames(FaFile, PathToMatchedFaFile, ProteomeDict)
    os.remove(PathToFaFile)
    MatchedFaFile = open(PathToMatchedFaFile, "r")
    MS_names, ExtSeqs = [], []
    Allseqs, Testseqs = [], []
    for rec1 in SeqIO.parse(MatchedFaFile, "fasta"):
        MS_seq = str(rec1.seq)
        MS_seqU = str(rec1.seq.upper())
        MS_name = str(rec1.description)
        try:
            UP_seq = ProteomeDict[MS_name]
            if MS_seqU in UP_seq and MS_name == list(ProteomeDict.keys())[list(ProteomeDict.values()).index(str(UP_seq))]:
                counter += 1
                Allseqs.append(MS_seq)
                regexPattern = re.compile(MS_seqU)
                MatchObs = regexPattern.finditer(UP_seq)
                indices = []
                for i in MatchObs:
                    indices.append(i.start())
                    indices.append(i.end())
                if "y" in MS_seq and "t" not in MS_seq and "s" not in MS_seq:
                    y_idx = MS_seq.index("y") + indices[0]
                    ExtSeqs.append(UP_seq[y_idx - 5 : y_idx] + "y" + UP_seq[y_idx + 1 : y_idx + 6])
                    MS_names.append(MS_name)
                    Testseqs.append(MS_seq)

                if "t" in MS_seq and "y" not in MS_seq and "s" not in MS_seq:
                    t_idx = MS_seq.index("t") + indices[0]
                    ExtSeqs.append(UP_seq[t_idx - 5 : t_idx] + "t" + UP_seq[t_idx + 1 : t_idx + 6])
                    MS_names.append(MS_name)
                    Testseqs.append(MS_seq)

                if "s" in MS_seq and "y" not in MS_seq and "t" not in MS_seq:
                    s_idx = MS_seq.index("s") + indices[0]
                    ExtSeqs.append(UP_seq[s_idx - 5 : s_idx] + "s" + UP_seq[s_idx + 1 : s_idx + 6])
                    MS_names.append(MS_name)
                    Testseqs.append(MS_seq)

                if "y" in MS_seq and "t" in MS_seq and "s" not in MS_seq:
                    y_idx = MS_seq.index("y") + indices[0]
                    ExtSeq = UP_seq[y_idx - 5 : y_idx] + "y" + UP_seq[y_idx + 1 : y_idx + 6]
                    y_idx = MS_seq.index("y")
                    if "t" in MS_seq[y_idx - 5 : y_idx + 6]:
                        t_idx = MS_seq[y_idx - 5 : y_idx + 6].index("t")
                        ExtSeqs.append(ExtSeq[:t_idx] + "t" + ExtSeq[t_idx + 1 :])
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)
                    else:
                        ExtSeqs.append(ExtSeq)
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)

                if "y" in MS_seq and "s" in MS_seq and "t" not in MS_seq:
                    y_idx = MS_seq.index("y") + indices[0]
                    ExtSeq = UP_seq[y_idx - 5 : y_idx] + "y" + UP_seq[y_idx + 1 : y_idx + 6]
                    y_idx = MS_seq.index("y")
                    if "s" in MS_seq[y_idx - 5 : y_idx + 6]:
                        s_idx = MS_seq[y_idx - 5 : y_idx + 6].index("s")
                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx + 1 :])
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)
                    else:
                        ExtSeqs.append(ExtSeq)
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)

                if "t" in MS_seq and "s" in MS_seq and "y" not in MS_seq:
                    t_idx = MS_seq.index("t") + indices[0]
                    ExtSeq = UP_seq[t_idx - 5 : t_idx] + "t" + UP_seq[t_idx + 1 : t_idx + 6]
                    t_idx = MS_seq.index("t")
                    if "s" in MS_seq[t_idx - 5 : t_idx + 6]:
                        s_idx = MS_seq[t_idx - 5 : t_idx + 6].index("s")
                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx + 1 :])
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)
                    else:
                        ExtSeqs.append(ExtSeq)
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)

                if "y" in MS_seq and "s" in MS_seq and "t" in MS_seq:
                    y_idx = MS_seq.index("y") + indices[0]
                    ExtSeq = UP_seq[y_idx - 5 : y_idx] + "y" + UP_seq[y_idx + 1 : y_idx + 6]
                    y_idx = MS_seq.index("y")
                    if "t" in MS_seq[y_idx - 5 : y_idx + 6]:
                        t_idx = MS_seq[y_idx - 5 : y_idx + 6].index("t")
                        ExtSeqs.append(ExtSeq[:t_idx] + "t" + ExtSeq[t_idx + 1 :])
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)
                    elif "s" in MS_seq[y_idx - 5 : y_idx + 6]:
                        s_idx = MS_seq[y_idx - 5 : y_idx + 6].index("s")
                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx + 1 :])
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)
                    else:
                        ExtSeqs.append(ExtSeq)
                        MS_names.append(MS_name)
                        Testseqs.append(MS_seq)
            else:
                print("check", MS_name, "with seq", MS_seq)
        except BaseException:
            print("find and replace", MS_name, "in proteome_uniprot.txt. Use: ", MS_seq)

    li_dif = [i for i in Testseqs + Allseqs if i not in Allseqs or i not in Testseqs]
    if li_dif:
        print(" Testseqs vs Allseqs may have different peptide sequences: ", li_dif)

    assert counter == len(MS_names) and counter == len(ExtSeqs), ("missing peptides", len(MS_names), len(ExtSeqs), counter)
    os.remove(PathToMatchedFaFile)
    proteome.close()
    return MS_names, ExtSeqs


def YTSsequences(X_seqs):
    """Goal: Generate dictionary to Check Motifs
       Input: Phosphopeptide sequences.
       Output: Dictionary to see all sequences categorized by singly or doubly phosphorylated.
       Useful to check def GeneratingKinaseMotifs results. """
    YTSdict = {}
    seq1, seq2, seq3, seq4, seq5, seq6, = [], [], [], [], [], []
    for seq in X_seqs:
        if "y" in seq and "t" not in seq and "s" not in seq:
            seq1.append(seq)
        if "t" in seq and "y" not in seq and "s" not in seq:
            seq2.append(seq)
            YTSdict["t: "] = seq2
        if "s" in seq and "y" not in seq and "t" not in seq:
            seq3.append(seq)
            YTSdict["s: "] = seq3
        if "y" in seq and "t" in seq and "s" not in seq:
            seq4.append(seq)
            YTSdict["y/t: "] = seq4
        if "y" in seq and "s" in seq and "t" not in seq:
            seq5.append(seq)
            YTSdict["y/s: "] = seq5
        if "t" in seq and "s" in seq and "y" not in seq:
            seq6.append(seq)

    YTSdict["y: "] = seq1
    YTSdict["t: "] = seq2
    YTSdict["s: "] = seq3
    YTSdict["y/t: "] = seq4
    YTSdict["y/s: "] = seq5
    YTSdict["t/s: "] = seq6

    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in YTSdict.items()]))
