from Bio import SeqIO
import os
import numpy as np
import random


###------------ Sequence processor ------------------###

def GenerateFastaFile(PathToFile, PN, X_seqs):
    FileHandle = open(PathToFile, "w+")
    for i in range(len(X_seqs)):
        FileHandle.write('>' + str(PN[i]))
        FileHandle.write("\n")
        FileHandle.write(str(X_seqs[i]))
        FileHandle.write("\n")
    FileHandle.close()


###------------ Dictionary to Check Motifs ------------------###

"""Input: Phosphopeptide sequences 
   Output: Dictionary to see all sequences categorized by singly or doubly phosphorylated.
   Useful to check def GeneratingKinaseMotifs results 
"""

def YTSsequences(X_seqs):
    YTSsequences = {}
    seq1 , seq2, seq3, seq4, seq5, seq6, = [], [], [], [], [], []
    for i, seq in enumerate(X_seqs):
        if "y" in seq and "t" not in seq and "s" not in seq:
            seq1.append(seq)
        if "t" in seq and "y" not in seq and "s" not in seq:
            seq2.append(seq)
            DictProtNameToPhospho["t: "] = seq2
        if "s" in seq and "y" not in seq and "t" not in seq:
            seq3.append(seq)
            DictProtNameToPhospho["s: "] = seq3
        if "y" in seq and "t" in seq and "s" not in seq:
            seq4.append(seq)
            DictProtNameToPhospho["y/t: "] = seq4
        if "y" in seq and "s" in seq and "t" not in seq:
            seq5.append(seq)
            DictProtNameToPhospho["y/s: "] = seq5
        if "t" in seq and "s" in seq and "y" not in seq:
            seq6.append(seq)

    DictProtNameToPhospho["y: "] = seq1
    DictProtNameToPhospho["t: "] = seq2
    DictProtNameToPhospho["s: "] = seq3
    DictProtNameToPhospho["y/t: "] = seq4
    DictProtNameToPhospho["y/s: "] = seq5
    DictProtNameToPhospho["t/s: "] = seq6

    SeqsBySites = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in DictProtNameToPhospho.items() ]))
    
    return SeqsBySites


###------------ Match protein names from MS to Uniprot's data set ------------------###        

 """Input: Path to new file and MS fasta file  
    Output: Protein names list and kinase motif list. Run with def GenerateFastaFile to obtain the final file
    Kinase motif -5 +5 wrt the phosphorylation site. It accounts for doubly phosphorylated peptides (lowercase y, t, s)
 """


def MatchProtNames(PathToNewFile, MS_seqs)
#     FileHandle = open("./msresist/data/MS_seqs_matched.fa", "w+")
    FileHandle = open(PathToNewFile, "w+")
    # counter = 0
    for rec1 in SeqIO.parse(MS_seqs, "fasta"):
        MS_seq = str(rec1.seq)
        MS_seqU = str(rec1.seq.upper())
        MS_name = str(rec1.description.split(" OS")[0])
        try:
            UP_seq = DictProtToSeq_UP[MS_name]
            FileHandle.write(">" + MS_name)
            FileHandle.write("\n")
            FileHandle.write(MS_seq)
            FileHandle.write("\n")
        except:
            # counter += 1
            Fixed_name = getKeysByValue(DictProtToSeq_UP, MS_seqU)
            FileHandle.write(">" + Fixed_name[0])
            FileHandle.write("\n")
            FileHandle.write(MS_seq)
            FileHandle.write("\n")       
    FileHandle.close()

    
###------------ Generate Phosphopeptide Motifs ------------------###        

 """Input: Fasta file and Uniprot's proteome dictionary key: Protein accession value: protein sequence
    Output: Protein names list and kinase motif list. Run with def GenerateFastaFile to obtain the final file
    Kinase motif -5 +5 wrt the phosphorylation site. It accounts for doubly phosphorylated peptides (lowercase y, t, s)
 """


def GeneratingKinaseMotifs(PathToFile, DictProtToSeq_UP):
    ExtSeqs = []
    MS_names = []
    for rec1 in SeqIO.parse(MS_seqs_matched, "fasta"):
        MS_seq = str(rec1.seq)
        MS_seqU = str(rec1.seq.upper())
        MS_name = str(rec1.description.split(" OS")[0])
        MS_names.append(MS_name)
        try:
            UP_seq = DictProtToSeq_UP[MS_name] 
            if MS_seqU in UP_seq and MS_name == list(DictProtToSeq_UP.keys())[list(DictProtToSeq_UP.values()).index(UP_seq)]:
                counter += 1
                regexPattern = re.compile(MS_seqU)
                MatchObs = regexPattern.finditer(UP_seq)
                indices = []
                for i in MatchObs:
                    indices.append(i.start())   #VHLENATEYAtLR   #YNIANtV
                    indices.append(i.end())                 
                if "y" in MS_seq and "t" not in MS_seq and "s" not in MS_seq:
                    y_idx = MS_seq.index("y") + indices[0]
                    ExtSeqs.append(UP_seq[y_idx-5:y_idx] + "y" + UP_seq[y_idx+1:y_idx+6])

                if "t" in MS_seq and "y" not in MS_seq and "s" not in MS_seq:
                    t_idx = MS_seq.index("t") + indices[0]
                    ExtSeqs.append(UP_seq[t_idx-5:t_idx] + "t" + UP_seq[t_idx+1:t_idx+6])

                if "s" in MS_seq and "y" not in MS_seq and "t" not in MS_seq:
                    s_idx = MS_seq.index("s") + indices[0]
                    ExtSeqs.append(UP_seq[s_idx-5:s_idx] + "s" + UP_seq[s_idx+1:s_idx+6])

                if "y" in MS_seq and "t" in MS_seq and "s" not in MS_seq:     
                    y_idx = MS_seq.index("y") + indices[0]                  
                    ExtSeq = UP_seq[y_idx-5:y_idx] + "y" + UP_seq[y_idx+1:y_idx+6]
                    y_idx = MS_seq.index("y")
                    if "t" in MS_seq[y_idx-5:y_idx+6]:
                        t_idx = MS_seq[y_idx-5:y_idx+6].index("t")
                        ExtSeqs.append(ExtSeq[:t_idx] + "t" + ExtSeq[t_idx+1:])
                    else:
                        ExtSeqs.append(ExtSeq)

                if "y" in MS_seq and "s" in MS_seq and "t" not in MS_seq:
                    y_idx = MS_seq.index("y") + indices[0]
                    ExtSeq = UP_seq[y_idx-5:y_idx] + "y" + UP_seq[y_idx+1:y_idx+6]
                    y_idx = MS_seq.index("y")
                    if "s" in MS_seq[y_idx-5:y_idx+6]:
                        s_idx = MS_seq[y_idx-5:y_idx+6].index("s")
                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx+1:])
                    else:
                        ExtSeqs.append(ExtSeq)

                if "t" in MS_seq and "s" in MS_seq and "y" not in MS_seq:
                    t_idx = MS_seq.index("t") + indices[0]
                    ExtSeq = UP_seq[t_idx-5:t_idx] + "t" + UP_seq[t_idx+1:t_idx+6]
                    t_idx = MS_seq.index("t")
                    if "s" in MS_seq[t_idx-5:t_idx+6]:
                        s_idx = MS_seq[t_idx-5:t_idx+6].index("s")
                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx+1:])
                    else:
                        ExtSeqs.append(ExtSeq)
        except:
            print("find and replace", MS_name, "in proteome_uniprot.txt. Use: ", MS_seq)
            pass
        
        return MS_names, ExtSeqs

###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###        

# Code from Adam Weiner, obtained March 2019

def trim(seqFile):
    cwd = os.getcwd()
    homeDir = cwd[1:5]
    if (homeDir == 'home'):
        print('using path from server to load sequences')
#         pathToFile = os.path.join("/home","zoekim","Desktop",str(seqFile)) #aretha server
        pathToFile = os.path.join("/home", "marcc", "resistance-MS", "msresist", "data", str(seqFile))     #/home/marcc/resistance-MS/msresist/data

    else:
        print('using path from mac machine to load sequences')
        pathToFile = os.path.join("/Users","zoekim","Desktop",str(seqFile))  #mac machine
    
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
        for j in range(0,sequence.shape[1]):
            if (sequence[i,j] == 'J'):
                sequence[i,j] = random.choice(['I', 'L'])
    print(label)
    print(sequence)
    
    return (label, sequence)



###------------ Seq Distance Calculator ------------------###
# Code from Adam Weiner, obtained March 2019


class Distance:
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

    def get_distance_matrix(self) -> None:
        pass


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

    def get_distance_matrix(self) -> dict:
        return self.distance_matrix


class FLU_sub(SubstitutionMatrix):
    """ Class implementing the FLU substitution matrix
    Reference: https://doi.org/10.1186/1471-2148-10-99 """

    def __init__(self, gap_penalty=3.354626E-4, gap_character: str = '-'):
        super(FLU_sub, self).__init__(gap_penalty, gap_character)
        # 'B' values are averages of 'N' & 'D'. 'Z' values are averages of 'E' & 'Q'. 'X' is an unknown (kept same as in PAM250)
        self.distance_matrix = \
            {('W', 'F'): 5.39392424532822, ('L', 'R'): 15.3000966197798, ('S', 'P'): 0.54225109402693, ('V', 'T'): 0.0743386, ('Q', 'Q'): 1.19562912226203, ('N', 'A'): 0.584852305649886, ('Z', 'Y'): 0.1965496447776915,
             ('W', 'R'): 0.0998554972524385, ('Q', 'A'): 1.4842345032161, ('S', 'D'): 0.135481232622983, ('H', 'H'): 0.243190142026506, ('S', 'H'): 0.368713573381758, ('H', 'D'): 0.0140859174993809, ('L', 'N'): 2.6468479652886,
             ('W', 'A'): 0.0182892882245349, ('Y', 'M'): 4.90484223478739, ('G', 'R'): 1.87956993845887, ('Y', 'I'): 14.3940521944257, ('Y', 'E'): 0.285047948309311, ('B', 'Y'): 0.0102575172450253,
             ('Y', 'A'): 3.53200526987468,
             ('V', 'D'): 0.0478596, ('B', 'S'): 1.137906774491807, ('Y', 'Y'): 0.167581646770807, ('G', 'N'): 1.38709603234116, ('E', 'C'): 0.116941459124876, ('Y', 'Q'): 0.406697814049488,
             ('Z', 'Z'): 0.7512076573675385,  # 4-way average between (E,E), (E, Q), (Q, E) and (Q, Q)
             ('V', 'A'): 0.0470718, ('C', 'C'): 0.00254733397966779, ('M', 'R'): 0.0160550314767596, ('V', 'E'): 0.0545874, ('T', 'N'): 0.000536284040016542, ('P', 'P'): 2.08738534433198, ('V', 'I'): 0.0671336,
             ('V', 'S'): 0.0884091, ('Z', 'P'): 0.397481398280146, ('V', 'M'): 0.0181507, ('T', 'F'): 0.814753093809928, ('V', 'Q'): 0.0333036, ('K', 'K'): 1.33129161941264,
             ('P', 'D'): 0.338372183381345,
             ('I', 'H'): 0.321611693603646, ('I', 'D'): 0.00573068208525287, ('T', 'R'): 1.36942940801512, ('P', 'L'): 0.570766693213698, ('K', 'G'): 0.00150046692269255, ('M', 'N'): 0.000836445615590923,
             ('P', 'H'): 0.580704249811294,
             ('F', 'Q'): 0.712769599068934, ('Z', 'G'): 2.793402637822021, ('X', 'L'): 0.36788, ('T', 'M'): 0.0704600385245663, ('Z', 'C'): 0.116941459124876, ('X', 'H'): 0.36788,
             ('D', 'R'): 0.16720700818221,
             ('B', 'W'): 0.449250234731645, ('X', 'D'): 0.36788, ('Z', 'K'): 0.2521167663268005, ('F', 'A'): 0.659311477863896, ('Z', 'W'): 0.0881494028758037, ('F', 'E'): 0.319558828428154,
             ('D', 'N'): 1.30249856764315e-5,
             ('B', 'K'): 0.02350732575925, ('X', 'X'): 0.36788, ('F', 'I'): 0.0805433268150369, ('B', 'G'): 1.137333290877596, ('X', 'T'): 1, ('F', 'M'): 0.0568693216513547, ('B', 'C'): 0.338056021879858,
             ('Z', 'I'): 0.518433245428551, ('Z', 'V'): 0.0439455, ('S', 'S'): 2.2068599339404, ('L', 'Q'): 2.559587177122, ('W', 'E'): 0.104092870343653, ('Q', 'R'): 0.124897616909194,
             ('N', 'N'): 7.73739287051356,
             ('W', 'M'): 0.874272174533394, ('Q', 'C'): 3.91106992668137e-11, ('W', 'I'): 0.273934263183281, ('S', 'C'): 0.011975265782196, ('L', 'A'): 0.474333610192982, ('S', 'G'): 0.0188080299490973,
             ('L', 'E'): 3.88148880863814,
             ('W', 'Q'): 0.0722059354079545, ('H', 'G'): 1.62662283098296e-5, ('S', 'K'): 1.52696419998775, ('Q', 'N'): 0.0616521921873234, ('N', 'R'): 0.00677184253227681, ('H', 'C'): 0.00111215807314139,
             ('Y', 'N'): 0.0102575172450253,
             ('G', 'Q'): 5.33031341222104, ('Y', 'F'): 0.592587985458668, ('C', 'A'): 0.353753981649393, ('V', 'L'): 0.0714981, ('G', 'E'): 0.256491863423002, ('G', 'A'): 0.214757862168721, ('K', 'R'): 0.890162345593224,
             ('E', 'D'): 1.93483278448943, ('Y', 'R'): 0.103964386383736, ('M', 'Q'): 0.0326806570137471, ('T', 'I'): 0.0321321499585514, ('C', 'D'): 0.145469388422239, ('V', 'F'): 0.0304961,
             ('T', 'A'): 0.195966354027106,
             ('T', 'P'): 0.000431020702277328, ('B', 'P'): 2.109841356997958, ('T', 'E'): 0.155245492137294, ('V', 'N'): 0.0742143, ('P', 'G'): 1.58564657669139, ('M', 'A'): 0.0804909094320368, ('K', 'H'): 0.00127350890508147,
             ('V', 'R'): 0.0509102, ('P', 'C'): 0.336263344504404, ('M', 'E'): 0.00100350082518749, ('K', 'L'): 6.74693648486614, ('V', 'V'): 0.0632292, ('M', 'I'): 1.46335727834648,
             ('T', 'Q'): 0.0440205200833047,
             ('I', 'G'): 0.00651622937676521, ('P', 'K'): 0.283807671568883, ('M', 'M'): 0.279910508981581, ('K', 'D'): 0.0417629637305017, ('I', 'C'): 0.00561362724916376, ('Z', 'D'): 3.65267203158433,
             ('F', 'R'): 0.154027179890711,
             ('X', 'K'): 0.36788, ('Q', 'D'): 5.37051127867923, ('X', 'G'): 0.36788, ('Z', 'L'): 3.22053799288007, ('X', 'C'): 0.049787, ('Z', 'H'): 0.0215253310839905,
             ('B', 'L'): 1.468445472716209,
             ('B', 'H'): 0.1163289464577254, ('F', 'F'): 0.0071324304661639, ('X', 'W'): 0.0183156, ('B', 'D'): 0.0070660312743935, ('D', 'A'): 0.0264470951166826, ('S', 'L'): 0.0449263566753846, ('X', 'S'): 1,
             ('F', 'N'): 0.0364417719063219, ('S', 'R'): 0.183076905018197, ('W', 'D'): 0.525398542949365, ('V', 'Y'): 0.0314741, ('W', 'L'): 0.340058468374384, ('H', 'R'): 0.246117171830255,
             ('W', 'H'): 6.44895444648517,
             ('H', 'N'): 0.21857197541607, ('W', 'T'): 0.124898020409882, ('T', 'T'): 0.207066205546908, ('S', 'F'): 0.000134906239484254, ('W', 'P'): 0.000182294881489116, ('L', 'D'): 0.290042980143818,
             ('B', 'I'): 0.00328327762975,
             ('L', 'H'): 0.347302791211758, ('S', 'N'): 2.14033231636063, ('B', 'T'): 0.000275588955, ('L', 'L'): 0.129223639195248, ('Y', 'K'): 0.0731279296372675, ('E', 'Q'): 0.108051341246072, ('Y', 'G'): 0.337229618868315,
             ('Z', 'S'): 0.44123292927066, ('Y', 'C'): 0.0549045639492389, ('G', 'D'): 0.887570549414031, ('B', 'V'): 0.06103695, ('E', 'A'): 1.13231312248046, ('Y', 'W'): 0.256900461407996, ('E', 'E'): 1.59309882471598,
             ('Y', 'S'): 0.0882564232979724, ('C', 'N'): 0.530642655337477, ('V', 'C'): 0.0250216, ('T', 'H'): 0.0223729191088972, ('P', 'R'): 0.950138410087378, ('V', 'G'): 0.0763734,
             ('T', 'L'): 0.431277662888057,
             ('V', 'K'): 0.0567845, ('K', 'Q'): 0.190259181297527, ('R', 'A'): 0.0533665787145181, ('I', 'R'): 0.296045557460629, ('T', 'D'): 1.4893873721753e-5, ('P', 'F'): 0.996685669575839,
             ('I', 'N'): 0.000835873174542931,
             ('K', 'I'): 9.01795420287895, ('M', 'D'): 1.0600102849456e-6, ('V', 'W'): 0.0185237, ('W', 'W'): 0.42775543040588, ('M', 'H'): 0.119028506158521, ('P', 'N'): 3.88131053061457,
             ('K', 'A'): 0.0587454231508643,
             ('M', 'L'): 2.98680003596399, ('K', 'E'): 0.313974351356074, ('Z', 'E'): 0.85057507, ('X', 'N'): 1, ('Z', 'A'): 1.30827381284828, ('Z', 'M'): 0.0168420789, ('X', 'F'): 0.135335,
             ('K', 'C'): 0.111457310321926, ('B', 'Q'): 2.716081735433277, ('X', 'B'): 0.36788, ('B', 'M'): 0.000418752812795, ('F', 'C'): 1.59312060172652e-13, ('Z', 'Q'): 0.651840231754051,
             ('X', 'Z'): 0.36788,
             ('F', 'G'): 0.0386317614553493, ('B', 'E'): 1.128678716181999, ('X', 'V'): 0.36788, ('F', 'K'): 0.195750631825315, ('B', 'A'): 0.305649700384943, ('X', 'R'): 0.36788,
             ('D', 'D'): 0.014132062548787,
             ('W', 'G'): 0.0748149970972622, ('Z', 'F'): 0.516164213714077, ('S', 'Q'): 0.60234096342392, ('W', 'C'): 0.601692431136271, ('W', 'K'): 0.0124162215506117, ('H', 'Q'): 0.0288399502994541,
             ('L', 'C'): 3.83228119049152e-6,
             ('W', 'N'): 0.373101926513925, ('S', 'A'): 5.4182981753166, ('L', 'G'): 0.264148929349066, ('W', 'S'): 0.392552239890831, ('S', 'E'): 0.2801248951174, ('H', 'E'): 0.0142107118685268,
             ('S', 'I'): 2.90405228596936,
             ('H', 'A'): 0.149926734229061, ('S', 'M'): 2.03151132062208, ('Y', 'L'): 0.890598579382591, ('Y', 'H'): 0.0986313546653266, ('Y', 'D'): 0.297123975243582, ('E', 'R'): 1.19062446519178,
             ('X', 'P'): 0.36788,
             ('G', 'G'): 0.0587745274250666, ('G', 'C'): 0.0218446166959521, ('E', 'N'): 0.322524647863997, ('Y', 'T'): 0.654109108255219, ('Y', 'P'): 0.0589719751511691, ('T', 'K'): 4.97641445484395e-5, ('A', 'A'): 0.138658764751059,
             ('P', 'Q'): 0.487822498528951, ('T', 'C'): 0.0941066800969967, ('V', 'H'): 0.0199642, ('T', 'G'): 0.196486447133033, ('I', 'Q'): 1.02036695531654, ('Z', 'T'): 0.0996330061102993,
             ('C', 'R'): 3.29271694159791,
             ('V', 'P'): 0.0506561, ('P', 'E'): 0.307140298031341, ('M', 'C'): 0.10405366623526, ('K', 'N'): 0.00525168778853117, ('I', 'I'): 3.51207228207807, ('P', 'A'): 3.01134451903854,
             ('M', 'G'): 0.00123664495412902,
             ('T', 'S'): 0.0998357527014247, ('I', 'E'): 0.016499535540562, ('P', 'M'): 0.00702658828739369, ('M', 'K'): 0.319895904499071, ('I', 'A'): 0.0231169515264061, ('P', 'I'): 0.290381075260226,
             ('R', 'R'): 0.161000889039552,
             ('X', 'M'): 0.36788, ('L', 'I'): 0.227707997165566, ('X', 'I'): 0.36788, ('Z', 'B'): 1.922380225804995,  # 4-way average of (E, N), (E, D), (Q, N), (Q, D)
             ('X', 'E'): 0.36788, ('Z', 'N'): 0.1920884200256602, ('X', 'A'): 1,
             ('B', 'R'): 0.086989425265, ('B', 'N'): 3.86869643525678, ('F', 'D'): 0.188539456415654, ('X', 'Y'): 0.135335, ('Z', 'R'): 0.657761041050487, ('F', 'H'): 0.924466914225534,
             ('B', 'F'): 0.1124906141609879,
             ('F', 'L'): 0.634308520867322, ('X', 'Q'): 0.36788, ('B', 'B'): 1.93788774574589}  # 4-way average of (N, N), (N, D), (D, N), (D, D)

    def get_distance_matrix(self) -> dict:
        return self.distance_matrix

