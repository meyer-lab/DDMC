"""Mapping to Uniprot's Proteome To Generate +/-5AA p-site Motifs."""

import glob
import pandas as pd
import numpy as np
import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from .binomial import AAlist


path = os.path.dirname(os.path.abspath(__file__))


def MapMotifs(X, names):
    """Generate pY motifs for pre-processing."""
    names, seqs, pXpos, Xidx = GeneratingKinaseMotifs(names, FormatSeq(X))
    X = X.iloc[Xidx, :]
    X["Gene"] = names
    X["Sequence"] = seqs
    X.insert(3, "Position", pXpos)
    return X[~X["Sequence"].str.contains("-")]


def FormatName(X):
    """ Keep only the general protein name, without any other accession information """
    full = [v.split("OS")[0].strip() for v in X.iloc[:, 0]]
    gene = [v.split("GN=")[1].split(" PE")[0].strip() for v in X.iloc[:, 0]]
    return full, gene


def FormatSeq(X):
    """ Deleting -1/-2 for mapping to uniprot's proteome"""
    return [v.split("-")[0] for v in X["Sequence"]]


def DictProteomeNameToSeq(X, n):
    """ To generate proteom's dictionary """
    DictProtToSeq_UP = {}
    for rec2 in SeqIO.parse(X, "fasta"):
        UP_seq = str(rec2.seq)
        if n == "full":
            UP_name = rec2.description.split("HUMAN ")[1].split(" OS")[0]
            DictProtToSeq_UP[UP_name] = str(UP_seq)
        if n == "gene":
            try:
                UP_name = rec2.description.split(" GN=")[1].split(" ")[0]
                DictProtToSeq_UP[UP_name] = str(UP_seq)
            except BaseException:
                continue
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
    matchedNames, seqs, Xidx = [], [], []
    counter = 0
    for i, MS_seq in enumerate(MS_seqs):
        MS_seqU = MS_seq.upper()
        MS_name = MS_names[i].strip()
        if MS_name in ProteomeDict and MS_seqU in ProteomeDict[MS_name]:
            Xidx.append(i)
            seqs.append(MS_seq)
            matchedNames.append(MS_name)
        else:
            try:
                newname = getKeysByValue(ProteomeDict, MS_seqU)[0]
                assert MS_seqU in ProteomeDict[newname]
                Xidx.append(i)
                seqs.append(MS_seq)
                matchedNames.append(newname)
            except BaseException:
                print(MS_name, MS_seqU)
                counter += 1
                continue

    assert counter == 0, "Proteome is missing %s peptides" % (counter)
    assert len(matchedNames) == len(seqs)
    return matchedNames, seqs, Xidx


def findmotif(MS_seq, MS_name, ProteomeDict, motif_size):
    """ For a given MS peptide, finds it in the ProteomeDict, and maps the +/-5 AA from the p-site, accounting
    for peptides phosphorylated multiple times concurrently. """
    MS_seqU = MS_seq.upper()
    try:
        UP_seq = ProteomeDict[MS_name]
        assert MS_seqU in UP_seq, "check " + MS_name + " with seq " + MS_seq + ". Protein sequence found: " + UP_seq
        regexPattern = re.compile(MS_seqU)
        MatchObs = list(regexPattern.finditer(UP_seq))
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
            mappedMotif, pidx = makeMotif(UP_seq, MS_seq, motif_size, y_idx, center_idx, DoS_idx)
            if len(pidx) == 1:
                pos = pidx[0]
            if len(pidx) > 1:
                pos = ";".join(pidx)

        if "y" not in MS_seq:
            pTS_idx = list(re.compile("t|s").finditer(MS_seq))
            assert len(pTS_idx) != 0
            center_idx = pTS_idx[0].start()
            ts_idx = center_idx + MatchObs[0].start()
            DoS_idx = None
            if len(pTS_idx) > 1:
                DoS_idx = pTS_idx[1:]
            mappedMotif, pidx = makeMotif(UP_seq, MS_seq, motif_size, ts_idx, center_idx, DoS_idx)
            if len(pidx) == 1:
                pos = pidx[0]
            if len(pidx) > 1:
                pos = ";".join(pidx)

    except BaseException:
        print(MS_name + " not in ProteomeDict.")
        raise

    return pos, mappedMotif


def GeneratingKinaseMotifs(names, seqs):
    """ Main function to generate motifs using 'findmotif'. """
    motif_size = 5
    proteome = open(os.path.join(path, "./data/Sequence_analysis/proteome_uniprot2019.fa"), "r")
    ProteomeDict = DictProteomeNameToSeq(proteome, n="gene")
    protnames, seqs, Xidx = MatchProtNames(ProteomeDict, names, seqs)
    MS_names, mapped_motifs, uni_pos, = [], [], []

    for i, MS_seq in enumerate(seqs):
        pos, mappedMotif = findmotif(MS_seq, protnames[i], ProteomeDict, motif_size)
        MS_names.append(protnames[i])
        mapped_motifs.append(mappedMotif)
        uni_pos.append(pos)

    proteome.close()
    return MS_names, mapped_motifs, uni_pos, Xidx


def makeMotif(UP_seq, MS_seq, motif_size, ps_protein_idx, center_motif_idx, DoS_idx):
    """ Make a motif out of the matched sequences. """
    UP_seq_copy = list(UP_seq[max(0, ps_protein_idx - motif_size): ps_protein_idx + motif_size + 1])
    assert len(UP_seq_copy) > motif_size, "Size seems too small. " + UP_seq

    # If we ran off the end of the sequence at the beginning or at the end, append a gap
    if ps_protein_idx - motif_size < 0:
        for _ in range(motif_size - ps_protein_idx):
            UP_seq_copy.insert(0, "-")

    elif ps_protein_idx + motif_size + 1 > len(UP_seq):
        for _ in range(ps_protein_idx + motif_size - len(UP_seq) + 1):
            UP_seq_copy.extend("-")

    UP_seq_copy[motif_size] = UP_seq_copy[motif_size].lower()

    pidx = [str(UP_seq_copy[motif_size]).upper() + str(ps_protein_idx + 1) + "-p"]

    # Now go through and copy over phosphorylation
    if DoS_idx:
        for ppIDX in DoS_idx:
            position = ppIDX.start() - center_motif_idx
            # If the phosphosite is within the motif
            if abs(position) < motif_size:
                editPos = position + motif_size
                UP_seq_copy[editPos] = UP_seq_copy[editPos].lower()
                assert UP_seq_copy[editPos] == MS_seq[ppIDX.start()], UP_seq_copy[editPos] + " " + MS_seq[ppIDX.start()]
                if position != 0:
                    pidx.append(str(UP_seq_copy[editPos]).upper() + str(ps_protein_idx + position + 1) + "-p")

    return "".join(UP_seq_copy), pidx


def preprocess_seqs(X, pYTS):
    """ Filter out any sequences with different than the specified central p-residue
    and/or any containing gaps."""
    X = X[~X["Sequence"].str.contains("-")]

    Xidx = []
    for seq in X["Sequence"]:
        Xidx.append(seq[5] == pYTS.lower())
    return X.iloc[Xidx, :]


def ForegroundSeqs(sequences):
    """ Build Background data set for either "Y", "S", or "T". """
    seqs = []
    yts = ["Y", "T", "S"]
    for motif in sequences:
        motif = motif.upper()
        assert "-" not in motif, "gap in motif"
        assert motif[5] in yts, "WRONG CENTRAL AMINO ACID"
        seqs.append(Seq(motif, alphabet=AAlist))
    return seqs


def PSPLdict():
    """Generate dictionary with kinase name-specificity profile pairs"""
    pspl_dict = {}
    # individual files
    PSPLs = glob.glob("./msresist/data/PSPL/*.csv")
    for sp in PSPLs:
        if sp == "./msresist/data/PSPL/pssm_data.csv":
            continue
        sp_mat = pd.read_csv(sp).sort_values(by="Unnamed: 0")

        if sp_mat.shape[0] > 20:  # Remove profiling of fixed pY and pT, include only natural AA
            assert np.all(sp_mat.iloc[:-2, 0] == AAlist), "aa don't match"
            sp_mat = sp_mat.iloc[:-2, 1:].values
        else:
            assert np.all(sp_mat.iloc[:, 0] == AAlist), "aa don't match"
            sp_mat = sp_mat.iloc[:, 1:].values

        if np.all(sp_mat >= 0):
            sp_mat = np.log2(sp_mat)

        pspl_dict[sp.split("PSPL/")[1].split(".csv")[0]] = sp_mat

    # NetPhores PSPL results
    f = pd.read_csv("msresist/data/PSPL/pssm_data.csv", header=None)
    matIDX = [np.arange(16) + i for i in range(0, f.shape[0], 16)]
    for ii in matIDX:
        kin = f.iloc[ii[0], 0]
        mat = f.iloc[ii[1:], :].T
        mat.columns = np.arange(mat.shape[1])
        mat = mat.iloc[:-1, 2:12].drop(8, axis=1).astype("float64").values
        mat = np.ma.log2(mat)
        mat = mat.filled(0)
        mat = np.clip(mat, a_min=0, a_max=3)
        pspl_dict[kin] = mat

    return pspl_dict


def compute_control_pssm(bg_sequences):
    """Generate PSSM."""
    back_pssm = np.zeros((len(AAlist), 11), dtype=float)
    for _, seq in enumerate(bg_sequences):
        for kk, aa in enumerate(seq):
            back_pssm[AAlist.index(aa), kk] += 1.0
    for pos in range(back_pssm.shape[1]):
        back_pssm[:, pos] /= np.mean(back_pssm[:, pos])
    back_pssm = np.ma.log2(back_pssm)
    return back_pssm.filled(0)


KinToPhosphotypeDict = {
    "ABL": "Y",
    "AKT": "S/T",
    "ALK": "Y",
    "BLK": "Y",
    "BRK": "Y",
    "CK2": "S/T",
    "ERK2": "S/T",
    "FRK": "Y",
    "HCK": "Y",
    "INSR": "Y",
    "LCK": "Y",
    "LYN": "Y",
    "MET": "Y",
    "NEK1": "S/T",
    "NEK2": "S/T",
    "NEK3": "S/T",
    "NEK4": "S/T",
    "NEK5": "S/T",
    "NEK6": "S/T",
    "NEK7": "S/T",
    "NEK8": "S/T",
    "NEK9": "S/T",
    "NEK10_S": "S/T",
    "NEK10_Y": "Y",
    "PKA": "S/T",
    "PKC-theta": "S/T",
    "PKD": "S/T",
    "PLM2": "S/T",
    "RET": "Y",
    "SRC": "Y",
    "TbetaRII": "S/T",
    "YES": "Y",
    "BRCA1": "S/T",
    "AMPK": "S/T",
    "CDK5": "S/T",
    "CK1": "S/T",
    "DMPK1": "S/T",
    "EGFR": "Y",
    "InsR": "Y",
    "p38": "S/T",
    "ERK1": "S/T",
    "SHC1": "Y",
    "SH2_PLCG1": "Y",
    "SH2_INPP5D": "Y",
    "SH2_SH3BP2": "Y",
    "SH2_SHC2": "Y",
    "SH2_SHE": "Y",
    "SH2_Syk": "Y",
    "SH2_TNS4": "Y",
    "CLK2": "S/T",
    "DAPK3": "S/T",
    "ICK": "S/T",
    "STK11": "S/T",
    "MST1": "S/T",
    "MST4": "S/T",
    "PAK2": "S/T",
    "Pim1": "S/T",
    "Pim2": "S/T",
    "SLK": "S/T",
    "TGFbR2": "S/T",
    "TLK1": "S/T",
    "TNIK": "S/T",
    "p70S6K": "S/T",
    "EphA3": "Y"
}
