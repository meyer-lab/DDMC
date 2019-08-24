"""Sequence Analysis Functions. """

import os
import re
import pandas as pd
from Bio import SeqIO

path = os.path.dirname(os.path.abspath(__file__))
###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###

def pYmotifs(ABC_conc_mc, ABC_names):
    ABC_seqs = FormatSeq(ABC_conc_mc)
    ABC_conc_mc['peptide-phosphosite'] = ABC_seqs

    directory = os.path.join(path, "./data/Sequence_analysis/")
    names, motifs = GeneratingKinaseMotifs(ABC_names, ABC_seqs, "MatchedFaFile.fa", directory + "proteome_uniprot.fa")
    ABC_conc_mc['peptide-phosphosite'] = motifs
    ABC_conc_mc['Master Protein Descriptions'] = names
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


def MatchProtNames(PathToMatchedFaFile, ProteomeDict, MS_names, MS_seqs):
    """ Goal: Match protein names of MS and Uniprot's proteome.
    Input: Path to new file and MS fasta file
    Output: Fasta file with matching protein names.
    Note that ProteomeDict[MS_name] is what the function needs to try to find
    and jump to except if MS_name is not in ProteomDict. """
    FileHandle = open(PathToMatchedFaFile, "w+")
    for i in range(len(MS_seqs)):
        MS_seq = str(MS_seqs[i])
        MS_seqU = str(MS_seqs[i]).upper()
        MS_name = str(MS_names[i].split(" OS")[0])
        if MS_name in ProteomeDict:
            FileHandle.write(">" + ProteomeDict[MS_name])
        else:
            FileHandle.write(">" + getKeysByValue(ProteomeDict, MS_seqU)[0])
        FileHandle.write("\n")
        FileHandle.write(MS_seq)
        FileHandle.write("\n")
    FileHandle.close()


def GeneratingKinaseMotifs(MS_names, MS_seqs, PathToMatchedFaFile, PathToProteome):
    """ Goal: Generate Phosphopeptide motifs.
    Input: Directory paths to fasta file, fasta file with matched names, and proteome
    Output: Protein names list and kinase motif list. Run with def GenerateFastaFile to obtain the final file.
    Kinase motif -5 +5 wrt the phosphorylation site. It accounts for doubly phosphorylated peptides (lowercase y, t, s). """
    counter = 0
    motif_size = 5
    proteome = open(PathToProteome, 'r')
    ProteomeDict = DictProteomeNameToSeq(proteome)
    MatchProtNames(PathToMatchedFaFile, ProteomeDict, MS_names, MS_seqs)
    MatchedFaFile = open(PathToMatchedFaFile, 'r')
    MS_names, ExtSeqs = [], []
    Allseqs, Testseqs = [], []
    for rec1 in SeqIO.parse(MatchedFaFile, "fasta"):
        MS_seq = str(rec1.seq)
        MS_seqU = str(rec1.seq.upper())
        MS_name = str(rec1.description)
        try:
            UP_seq = ProteomeDict[MS_name]
            assert MS_seqU in UP_seq, "check " + MS_name + " with seq " + MS_seq
            assert MS_name == list(ProteomeDict.keys())[list(ProteomeDict.values()).index(str(UP_seq))], \
                "check " + MS_name + " with seq " + MS_seq

            counter += 1
            Allseqs.append(MS_seq)
            regexPattern = re.compile(MS_seqU)
            MatchObs = list(regexPattern.finditer(UP_seq))
            # Assert there's only one match
            # assert len(MatchObs) == 1, print(MatchObs)
            # TODO: Note that we're only taking the first peptide match

            # For all y, t, s phosphosites
            phosphoIDX = list(re.compile("y|t|s").finditer(MS_seq))
            assert len(phosphoIDX) != 0
            for pIDX in phosphoIDX:
                y_idx = pIDX.start() + MatchObs[0].start()

                MS_names.append(MS_name)
                Testseqs.append(MS_seq)
                ExtSeqs.append(makeMotif(UP_seq, MS_seq, motif_size, y_idx, pIDX, phosphoIDX))
                
        except BaseException:
            print("find and replace", MS_name, "in proteome_uniprot.txt. Use: ", MS_seq)
            raise

    li_dif = [i for i in Testseqs + Allseqs if i not in Allseqs or i not in Testseqs]
    if li_dif:
        print(" Testseqs vs Allseqs may have different peptide sequences: ", li_dif)

    assert counter == len(MS_names), "missing peptides"
    assert counter == len(ExtSeqs), "missing peptides"

    os.remove(PathToMatchedFaFile)
    proteome.close()
    return MS_names, ExtSeqs


def makeMotif(UP_seq, MS_seq, motif_size, y_idx, pIDX, phosphoIDX):
    """ Make a motif out of the matched sequences. """
    UP_seq_copy = list(UP_seq[max(0, y_idx - motif_size):y_idx + motif_size + 1])
    assert len(UP_seq_copy) > motif_size, "Size seems too small. " + UP_seq

    # If we ran off the end of the sequence at the beginning, append a gap
    if y_idx - motif_size < 0:
        for ii in range(motif_size - y_idx):
            UP_seq_copy.insert(0, "-")

    # Now go through and copy over phosphorylation
    for pppIDX in phosphoIDX:
        position = pppIDX.start() - pIDX.start()
        # If the phosphosite is within the motif
        if abs(position) < motif_size:
            editPos = position + motif_size
            UP_seq_copy[editPos] = UP_seq_copy[editPos].lower()
            assert UP_seq_copy[editPos] == MS_seq[pppIDX.start()], \
                UP_seq_copy[editPos] + " " + MS_seq[pppIDX.start()]

    return ''.join(UP_seq_copy)
