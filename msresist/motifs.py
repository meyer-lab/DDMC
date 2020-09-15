"""Mapping to Uniprot's Proteome To Generate +/-5AA p-site Motifs."""

import os
import re
from Bio import SeqIO
from Bio.Seq import Seq


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
        seqs.append(Seq(motif))
    return seqs
