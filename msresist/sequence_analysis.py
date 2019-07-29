from Bio import SeqIO
import os
import numpy as np
import random


def GenerateFastaFile(PathToFile, PN, X_seqs):
    """ Sequence processor """
    FileHandle = open(PathToFile, "w+")
    for i in range(len(X_seqs)):
        FileHandle.write('>' + str(PN[i]))
        FileHandle.write("\n")
        FileHandle.write(str(X_seqs[i]))
        FileHandle.write("\n")
    FileHandle.close()


# def YTSsequences(X_seqs):
#    """ Dictionary to Check Motifs
#        Input: Phosphopeptide sequences
#        Output: Dictionary to see all sequences categorized by singly or doubly phosphorylated.
#        Useful to check def GeneratingKinaseMotifs results """
#    YTSsequences = {}
#    seq1 , seq2, seq3, seq4, seq5, seq6, = [], [], [], [], [], []
#    for i, seq in enumerate(X_seqs):
#        if "y" in seq and "t" not in seq and "s" not in seq:
#            seq1.append(seq)
#        if "t" in seq and "y" not in seq and "s" not in seq:
#            seq2.append(seq)
#            DictProtNameToPhospho["t: "] = seq2
#        if "s" in seq and "y" not in seq and "t" not in seq:
#            seq3.append(seq)
#            DictProtNameToPhospho["s: "] = seq3
#        if "y" in seq and "t" in seq and "s" not in seq:
#            seq4.append(seq)
#            DictProtNameToPhospho["y/t: "] = seq4
#        if "y" in seq and "s" in seq and "t" not in seq:
#            seq5.append(seq)
#            DictProtNameToPhospho["y/s: "] = seq5
#        if "t" in seq and "s" in seq and "y" not in seq:
#            seq6.append(seq)
#
#    DictProtNameToPhospho["y: "] = seq1
#    DictProtNameToPhospho["t: "] = seq2
#    DictProtNameToPhospho["s: "] = seq3
#    DictProtNameToPhospho["y/t: "] = seq4
#    DictProtNameToPhospho["y/s: "] = seq5
#    DictProtNameToPhospho["t/s: "] = seq6
#
#    SeqsBySites = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in DictProtNameToPhospho.items() ]))
#
#    return SeqsBySites


###------------ Match protein names from MS to Uniprot's data set ------------------###

"""Input: Path to new file and MS fasta file
   Output: Protein names list and kinase motif list. Run with def GenerateFastaFile to obtain the final file
   Kinase motif -5 +5 wrt the phosphorylation site. It accounts for doubly phosphorylated peptides (lowercase y, t, s)
"""


# def MatchProtNames(PathToNewFile, MS_seqs):
##     FileHandle = open("./msresist/data/MS_seqs_matched.fa", "w+")
#    FileHandle = open(PathToNewFile, "w+")
#    # counter = 0
#    for rec1 in SeqIO.parse(MS_seqs, "fasta"):
#        MS_seq = str(rec1.seq)
#        MS_seqU = str(rec1.seq.upper())
#        MS_name = str(rec1.description.split(" OS")[0])
#        try:
#            UP_seq = DictProtToSeq_UP[MS_name]
#            FileHandle.write(">" + MS_name)
#            FileHandle.write("\n")
#            FileHandle.write(MS_seq)
#            FileHandle.write("\n")
#        except:
#            # counter += 1
#            Fixed_name = getKeysByValue(DictProtToSeq_UP, MS_seqU)
#            FileHandle.write(">" + Fixed_name[0])
#            FileHandle.write("\n")
#            FileHandle.write(MS_seq)
#            FileHandle.write("\n")
#    FileHandle.close()


###------------ Generate Phosphopeptide Motifs ------------------###

"""Input: Fasta file and Uniprot's proteome dictionary key: Protein accession value: protein sequence
   Output: Protein names list and kinase motif list. Run with def GenerateFastaFile to obtain the final file
   Kinase motif -5 +5 wrt the phosphorylation site. It accounts for doubly phosphorylated peptides (lowercase y, t, s)
"""


# def GeneratingKinaseMotifs(PathToFile, DictProtToSeq_UP):
#    ExtSeqs = []
#    MS_names = []
#    for rec1 in SeqIO.parse(MS_seqs_matched, "fasta"):
#        MS_seq = str(rec1.seq)
#        MS_seqU = str(rec1.seq.upper())
#        MS_name = str(rec1.description.split(" OS")[0])
#        MS_names.append(MS_name)
#        try:
#            UP_seq = DictProtToSeq_UP[MS_name]
#            if MS_seqU in UP_seq and MS_name == list(DictProtToSeq_UP.keys())[list(DictProtToSeq_UP.values()).index(UP_seq)]:
#                counter += 1
#                regexPattern = re.compile(MS_seqU)
#                MatchObs = regexPattern.finditer(UP_seq)
#                indices = []
#                for i in MatchObs:
#                    indices.append(i.start())  # VHLENATEYAtLR   #YNIANtV
#                    indices.append(i.end())
#                if "y" in MS_seq and "t" not in MS_seq and "s" not in MS_seq:
#                    y_idx = MS_seq.index("y") + indices[0]
#                    ExtSeqs.append(UP_seq[y_idx - 5:y_idx] + "y" + UP_seq[y_idx + 1:y_idx + 6])
#
#                if "t" in MS_seq and "y" not in MS_seq and "s" not in MS_seq:
#                    t_idx = MS_seq.index("t") + indices[0]
#                    ExtSeqs.append(UP_seq[t_idx - 5:t_idx] + "t" + UP_seq[t_idx + 1:t_idx + 6])
#
#                if "s" in MS_seq and "y" not in MS_seq and "t" not in MS_seq:
#                    s_idx = MS_seq.index("s") + indices[0]
#                    ExtSeqs.append(UP_seq[s_idx - 5:s_idx] + "s" + UP_seq[s_idx + 1:s_idx + 6])
#
#                if "y" in MS_seq and "t" in MS_seq and "s" not in MS_seq:
#                    y_idx = MS_seq.index("y") + indices[0]
#                    ExtSeq = UP_seq[y_idx - 5:y_idx] + "y" + UP_seq[y_idx + 1:y_idx + 6]
#                    y_idx = MS_seq.index("y")
#                    if "t" in MS_seq[y_idx - 5:y_idx + 6]:
#                        t_idx = MS_seq[y_idx - 5:y_idx + 6].index("t")
#                        ExtSeqs.append(ExtSeq[:t_idx] + "t" + ExtSeq[t_idx + 1:])
#                    else:
#                        ExtSeqs.append(ExtSeq)
#
#                if "y" in MS_seq and "s" in MS_seq and "t" not in MS_seq:
#                    y_idx = MS_seq.index("y") + indices[0]
#                    ExtSeq = UP_seq[y_idx - 5:y_idx] + "y" + UP_seq[y_idx + 1:y_idx + 6]
#                    y_idx = MS_seq.index("y")
#                    if "s" in MS_seq[y_idx - 5:y_idx + 6]:
#                        s_idx = MS_seq[y_idx - 5:y_idx + 6].index("s")
#                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx + 1:])
#                    else:
#                        ExtSeqs.append(ExtSeq)
#
#                if "t" in MS_seq and "s" in MS_seq and "y" not in MS_seq:
#                    t_idx = MS_seq.index("t") + indices[0]
#                    ExtSeq = UP_seq[t_idx - 5:t_idx] + "t" + UP_seq[t_idx + 1:t_idx + 6]
#                    t_idx = MS_seq.index("t")
#                    if "s" in MS_seq[t_idx - 5:t_idx + 6]:
#                        s_idx = MS_seq[t_idx - 5:t_idx + 6].index("s")
#                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx + 1:])
#                    else:
#                        ExtSeqs.append(ExtSeq)
#        except BaseException:
#            print("find and replace", MS_name, "in proteome_uniprot.txt. Use: ", MS_seq)
#            pass
#
#        return MS_names, ExtSeqs

###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###

# Code from Adam Weiner, obtained March 2019


def trim(seqFile):
    cwd = os.getcwd()
    homeDir = cwd[1:5]
    if (homeDir == 'home'):
        print('using path from server to load sequences')
#         pathToFile = os.path.join("/home","zoekim","Desktop",str(seqFile)) #aretha server
        pathToFile = os.path.join("/home", "marcc", "resistance-MS", "msresist", "data", str(seqFile))  # /home/marcc/resistance-MS/msresist/data

    else:
        print('using path from mac machine to load sequences')
        pathToFile = os.path.join("/Users", "zoekim", "Desktop", str(seqFile))  # mac machine

    allSeqs = []
    allLabels = []
    for seq_record in SeqIO.parse(pathToFile, """fasta"""):
        allSeqs.append(seq_record.seq)
        allLabels.append(seq_record.id)

    seqMat = np.array(allSeqs)
    label = np.array(allLabels)

    sequence = seqMat[:, 0:317]

    # filtering out residues not included in PAM250 pymsa distance matrix (http://www.matrixscience.com/blog/non-standard-amino-acid-residues.html)
    for i in range(0, sequence.shape[0]):
        for j in range(0, sequence.shape[1]):
            if (sequence[i, j] == 'J'):
                sequence[i, j] = random.choice(['I', 'L'])
    print(label)
    print(sequence)

    return (label, sequence)


class Distance:
    """ Seq Distance Calculator
        Code from Adam Weiner, obtained March 2019 """

    def __init__(self, seqFile, subMat):
        self.labels, self.sequences = trim(seqFile)
        self.subMat = subMat
        if subMat is "PAM250":
            self.M = PAM250()
        elif subMat is "FLU":
            self.M = FLU_sub()
        self.numSeq = self.sequences.shape[0]

    def seq_dist(self, seq1, seq2):
        if (len(seq1) != len(seq2)):
            print('the sequences are of different length')
            return -1
        else:
            dist = np.zeros((len(seq1)))
            for ii in range(len(seq1)):
                temp_dist = self.M.get_distance(seq1[ii], seq2[ii])
                if self.subMat is "PAM250":  # convert log-scaled PAM250 values to true values
                    temp_dist = np.exp(temp_dist)
                dist[ii] = 1 / temp_dist  # large distances have small values in matrices
#             avg_dist = np.sum(dist) / 317.0

            return np.sum(dist)

    def test_mat(self):
        """ function is the same as "dist_mat()" except that it only looks at first 10 sequences
        in order to get a proof of concept for all my functions before scaling up to the full dataset
        """
        testMat = np.zeros((1000, 1000))
        print('calculating the test distance matrix based on PAM250')
        for i in range(0, 1000):
            for j in range(i, 1000):
                testMat[i, j] = self.seq_dist(self.sequences[i], self.sequences[j])
                # plug in values for mirror images
                testMat[j, i] = testMat[i, j]

        return testMat

    def dist_mat(self):
        distMat = np.zeros((self.numSeq, self.numSeq))
        print(self.numSeq)
        print(self.sequences.shape)
        print('calculating the full distance matrix based on PAM250')
        for i in range(0, self.numSeq):
            for j in range(i, self.numSeq):
                distMat[i, j] = self.seq_dist(self.sequences[i], self.sequences[j])
                distMat[j, i] = distMat[i, j]  # plug in mirror image values
        return distMat


###------------ Substitution Matrix (PAM250) ------------------###
# Code from Adam Weiner, obtained March 2019

""" Code for substitution matrices is inspired by https://github.com/benhid/pyMSA/blob/master/pymsa/core/substitution_matrix.py """


class SubstitutionMatrix:
    """ Class representing a substitution matrix, such as PAM250, Blosum62, etc. """

    def __init__(self, gap_penalty: int, gap_character: str):
        self.gap_penalty = gap_penalty
        self.gap_character = gap_character
        self.distance_matrix = None

    def get_distance(self, char1, char2) -> int:
        """ Returns the distance between two characters.
        :param char1: First character.
        :param char2: Second character.
        :return: the distance value """
        print(char1)
        print(char2)
        if char1 is self.gap_character and char2 is self.gap_character:
            distance = 1
        elif char1 is self.gap_character or char2 is self.gap_character:
            distance = self.gap_penalty
        else:
            matrix = self.get_distance_matrix()
            try:
                distance = matrix[(char1, char2)] if (char1, char2) in matrix else matrix[(char2, char1)]
            except KeyError:
                print("The pair ({0},{1}) couldn't be found in the substitution matrix.".format(char1, char2))
                raise

        return distance

    def get_distance_matrix(self):
        return self.distance_matrix


class PAM250(SubstitutionMatrix):
    # Code from Adam Weiner, obtained March 2019
    """ Class implementing the PAM250 substitution matrix
    Reference: https://en.wikipedia.org/wiki/Point_accepted_mutation"""

    def __init__(self, gap_penalty=-8, gap_character: str = '-'):
        super(PAM250, self).__init__(gap_penalty, gap_character)
        self.distance_matrix = \
            {('W', 'F'): 0, ('L', 'R'): -3, ('S', 'P'): 1, ('V', 'T'): 0, ('Q', 'Q'): 4, ('N', 'A'): 0, ('Z', 'Y'): -4,
             ('W', 'R'): 2, ('Q', 'A'): 0, ('S', 'D'): 0, ('H', 'H'): 6, ('S', 'H'): -1, ('H', 'D'): 1, ('L', 'N'): -3,
             ('W', 'A'): -6, ('Y', 'M'): -2, ('G', 'R'): -3, ('Y', 'I'): -1, ('Y', 'E'): -4, ('B', 'Y'): -3,
             ('Y', 'A'): -3,
             ('V', 'D'): -2, ('B', 'S'): 0, ('Y', 'Y'): 10, ('G', 'N'): 0, ('E', 'C'): -5, ('Y', 'Q'): -4,
             ('Z', 'Z'): 3,
             ('V', 'A'): 0, ('C', 'C'): 12, ('M', 'R'): 0, ('V', 'E'): -2, ('T', 'N'): 0, ('P', 'P'): 6, ('V', 'I'): 4,
             ('V', 'S'): -1, ('Z', 'P'): 0, ('V', 'M'): 2, ('T', 'F'): -3, ('V', 'Q'): -2, ('K', 'K'): 5,
             ('P', 'D'): -1,
             ('I', 'H'): -2, ('I', 'D'): -2, ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2,
             ('P', 'H'): 0,
             ('F', 'Q'): -5, ('Z', 'G'): 0, ('X', 'L'): -1, ('T', 'M'): -1, ('Z', 'C'): -5, ('X', 'H'): -1,
             ('D', 'R'): -1,
             ('B', 'W'): -5, ('X', 'D'): -1, ('Z', 'K'): 0, ('F', 'A'): -3, ('Z', 'W'): -6, ('F', 'E'): -5,
             ('D', 'N'): 2,
             ('B', 'K'): 1, ('X', 'X'): -1, ('F', 'I'): 1, ('B', 'G'): 0, ('X', 'T'): 0, ('F', 'M'): 0, ('B', 'C'): -4,
             ('Z', 'I'): -2, ('Z', 'V'): -2, ('S', 'S'): 2, ('L', 'Q'): -2, ('W', 'E'): -7, ('Q', 'R'): 1,
             ('N', 'N'): 2,
             ('W', 'M'): -4, ('Q', 'C'): -5, ('W', 'I'): -5, ('S', 'C'): 0, ('L', 'A'): -2, ('S', 'G'): 1,
             ('L', 'E'): -3,
             ('W', 'Q'): -5, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 1, ('N', 'R'): 0, ('H', 'C'): -3,
             ('Y', 'N'): -2,
             ('G', 'Q'): -1, ('Y', 'F'): 7, ('C', 'A'): -2, ('V', 'L'): 2, ('G', 'E'): 0, ('G', 'A'): 1, ('K', 'R'): 3,
             ('E', 'D'): 3, ('Y', 'R'): -4, ('M', 'Q'): -1, ('T', 'I'): 0, ('C', 'D'): -5, ('V', 'F'): -1,
             ('T', 'A'): 1,
             ('T', 'P'): 0, ('B', 'P'): -1, ('T', 'E'): 0, ('V', 'N'): -2, ('P', 'G'): 0, ('M', 'A'): -1, ('K', 'H'): 0,
             ('V', 'R'): -2, ('P', 'C'): -3, ('M', 'E'): -2, ('K', 'L'): -3, ('V', 'V'): 4, ('M', 'I'): 2,
             ('T', 'Q'): -1,
             ('I', 'G'): -3, ('P', 'K'): -1, ('M', 'M'): 6, ('K', 'D'): 0, ('I', 'C'): -2, ('Z', 'D'): 3,
             ('F', 'R'): -4,
             ('X', 'K'): -1, ('Q', 'D'): 2, ('X', 'G'): -1, ('Z', 'L'): -3, ('X', 'C'): -3, ('Z', 'H'): 2,
             ('B', 'L'): -3,
             ('B', 'H'): 1, ('F', 'F'): 9, ('X', 'W'): -4, ('B', 'D'): 3, ('D', 'A'): 0, ('S', 'L'): -3, ('X', 'S'): 0,
             ('F', 'N'): -3, ('S', 'R'): 0, ('W', 'D'): -7, ('V', 'Y'): -2, ('W', 'L'): -2, ('H', 'R'): 2,
             ('W', 'H'): -3,
             ('H', 'N'): 2, ('W', 'T'): -5, ('T', 'T'): 3, ('S', 'F'): -3, ('W', 'P'): -6, ('L', 'D'): -4,
             ('B', 'I'): -2,
             ('L', 'H'): -2, ('S', 'N'): 1, ('B', 'T'): 0, ('L', 'L'): 6, ('Y', 'K'): -4, ('E', 'Q'): 2, ('Y', 'G'): -5,
             ('Z', 'S'): 0, ('Y', 'C'): 0, ('G', 'D'): 1, ('B', 'V'): -2, ('E', 'A'): 0, ('Y', 'W'): 0, ('E', 'E'): 4,
             ('Y', 'S'): -3, ('C', 'N'): -4, ('V', 'C'): -2, ('T', 'H'): -1, ('P', 'R'): 0, ('V', 'G'): -1,
             ('T', 'L'): -2,
             ('V', 'K'): -2, ('K', 'Q'): 1, ('R', 'A'): -2, ('I', 'R'): -2, ('T', 'D'): 0, ('P', 'F'): -5,
             ('I', 'N'): -2,
             ('K', 'I'): -2, ('M', 'D'): -3, ('V', 'W'): -6, ('W', 'W'): 17, ('M', 'H'): -2, ('P', 'N'): 0,
             ('K', 'A'): -1,
             ('M', 'L'): 4, ('K', 'E'): 0, ('Z', 'E'): 3, ('X', 'N'): 0, ('Z', 'A'): 0, ('Z', 'M'): -2, ('X', 'F'): -2,
             ('K', 'C'): -5, ('B', 'Q'): 1, ('X', 'B'): -1, ('B', 'M'): -2, ('F', 'C'): -4, ('Z', 'Q'): 3,
             ('X', 'Z'): -1,
             ('F', 'G'): -5, ('B', 'E'): 3, ('X', 'V'): -1, ('F', 'K'): -5, ('B', 'A'): 0, ('X', 'R'): -1,
             ('D', 'D'): 4,
             ('W', 'G'): -7, ('Z', 'F'): -5, ('S', 'Q'): -1, ('W', 'C'): -8, ('W', 'K'): -3, ('H', 'Q'): 3,
             ('L', 'C'): -6,
             ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4, ('W', 'S'): -2, ('S', 'E'): 0, ('H', 'E'): 1,
             ('S', 'I'): -1,
             ('H', 'A'): -1, ('S', 'M'): -2, ('Y', 'L'): -1, ('Y', 'H'): 0, ('Y', 'D'): -4, ('E', 'R'): -1,
             ('X', 'P'): -1,
             ('G', 'G'): 5, ('G', 'C'): -3, ('E', 'N'): 1, ('Y', 'T'): -3, ('Y', 'P'): -5, ('T', 'K'): 0, ('A', 'A'): 2,
             ('P', 'Q'): 0, ('T', 'C'): -2, ('V', 'H'): -2, ('T', 'G'): 0, ('I', 'Q'): -2, ('Z', 'T'): -1,
             ('C', 'R'): -4,
             ('V', 'P'): -1, ('P', 'E'): -1, ('M', 'C'): -5, ('K', 'N'): 1, ('I', 'I'): 5, ('P', 'A'): 1,
             ('M', 'G'): -3,
             ('T', 'S'): 1, ('I', 'E'): -2, ('P', 'M'): -2, ('M', 'K'): 0, ('I', 'A'): -1, ('P', 'I'): -2,
             ('R', 'R'): 6,
             ('X', 'M'): -1, ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 2, ('X', 'E'): -1, ('Z', 'N'): 1, ('X', 'A'): 0,
             ('B', 'R'): -1, ('B', 'N'): 2, ('F', 'D'): -6, ('X', 'Y'): -2, ('Z', 'R'): 0, ('F', 'H'): -2,
             ('B', 'F'): -4,
             ('F', 'L'): 2, ('X', 'Q'): -1, ('B', 'B'): 3}
