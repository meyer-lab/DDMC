"""Mapping to Uniprot's Proteome To Generate +/-5AA p-site Motifs.

Contains:
    - `get_proteome_name_to_seq`: parses a UniProt FASTA proteome into a
      `{protein name: sequence}` dictionary; used by `ddmc.datasets.EBDT`.
    - `get_pspls`: loads kinase specificity profiles (position-specific
      peptide libraries) from `ddmc/data/PSPL/`, used by
      `ddmc.clustering.DDMC.predict_upstream_kinases`.
    - `compute_control_pssm`: builds a background PSSM from a set of
      sequences, used by `ddmc.clustering.DDMC.get_pssms`.
    - `KinToPhosphotypeDict`: maps each kinase named in the PSPL data set to
      the phosphoacceptor type(s) it targets (S/T or Y).
    - `match_protein_names`, `find_motif`, `make_motif`,
      `generate_kinase_motifs`, `get_keys_by_value`: an older
      proteome-mapping pipeline for turning MS peptide hits into
      sequence motifs. Not currently called elsewhere in this package
      (`ddmc.datasets.EBDT.pos_to_motif` reimplements the same idea more
      simply) but kept here in case new datasets need the same matching
      logic.
"""

import glob
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import IO, Literal

import numpy as np
import pandas as pd
from Bio import SeqIO

from .binomial import AAlist


def get_proteome_name_to_seq(X: IO[str], n: str) -> dict[str, str]:
    """Parse a UniProt FASTA proteome into a name-to-sequence dictionary.

    Args:
        X: An open file handle to a UniProt FASTA file.
        n: Which identifier to key the dictionary by: `"full"` for the
            full human-readable protein name (parsed out of the FASTA
            description between `"HUMAN "` and `" OS"`), or `"gene"` for
            the gene symbol (parsed out of the `GN=` field). Records
            without a `GN=` field are skipped when `n == "gene"`.

    Returns:
        Dictionary mapping protein name or gene symbol to its amino acid
        sequence.
    """
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


def get_keys_by_value(dictionary: dict, value: str) -> list:
    """Find every key whose value contains a given substring.

    Args:
        dictionary: Dictionary to search, e.g. a protein-name-to-sequence
            map from `get_proteome_name_to_seq`.
        value: Substring to search for within each dictionary value.

    Returns:
        The keys whose value contains `value` (empty if none match).
    """
    listOfKeys = list()
    listOfItems = dictionary.items()
    for item in listOfItems:
        if value in item[1]:
            listOfKeys.append(item[0])
    return listOfKeys


def match_protein_names(
    ProteomeDict: dict[str, str], MS_names: Sequence[str], MS_seqs: Sequence[str]
) -> tuple[list[str], list[str], list[int]]:
    """Match protein names of MS and Uniprot's proteome.

    For each MS peptide, first tries its given protein name directly; if
    that name isn't in the proteome (or the peptide sequence doesn't occur
    in that entry), falls back to searching the whole proteome for the
    peptide sequence.

    Args:
        ProteomeDict: Protein-name-to-sequence dictionary, as returned by
            `get_proteome_name_to_seq`.
        MS_names: Protein name reported for each MS peptide.
        MS_seqs: The corresponding MS peptide sequence for each entry.

    Returns:
        A tuple `(matchedNames, seqs, Xidx)` of the resolved protein name,
        original sequence, and original index for each peptide that could
        be matched to the proteome.

    Raises:
        AssertionError: If any peptide could not be matched to the proteome.
    """
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
                newname = get_keys_by_value(ProteomeDict, MS_seqU)[0]
                assert MS_seqU in ProteomeDict[newname]
                Xidx.append(i)
                seqs.append(MS_seq)
                matchedNames.append(newname)
            except BaseException:
                print(MS_name, MS_seqU)
                counter += 1
                continue

    assert counter == 0, f"Proteome is missing {counter} peptides"
    assert len(matchedNames) == len(seqs)
    return matchedNames, seqs, Xidx


def find_motif(
    MS_seq: str, MS_name: str, ProteomeDict: dict[str, str], motif_size: int
) -> tuple[str, str]:
    """For a given MS peptide, finds it in the ProteomeDict, and maps the +/-5 AA from the p-site, accounting
    for peptides phosphorylated multiple times concurrently.

    Args:
        MS_seq: The MS peptide sequence, with its primary phosphoacceptor
            lowercased (and any additional, concurrently phosphorylated
            residues also lowercased).
        MS_name: The protein name to look `MS_seq` up under in
            `ProteomeDict`.
        ProteomeDict: Protein-name-to-sequence dictionary, as returned by
            `get_proteome_name_to_seq`.
        motif_size: Number of residues to include on each side of the
            phosphoacceptor in the extracted motif.

    Returns:
        A tuple `(pos, mappedMotif)`:
            pos: The phosphosite position(s) in the full protein sequence,
                formatted as `"{residue}{1-indexed position}-p"`, joined
                with `";"` if there are multiple concurrent phosphosites.
            mappedMotif: The extracted sequence motif (see `make_motif`).
    """
    MS_seqU = MS_seq.upper()
    try:
        UP_seq = ProteomeDict[MS_name]
        assert MS_seqU in UP_seq, (
            "check "
            + MS_name
            + " with seq "
            + MS_seq
            + ". Protein sequence found: "
            + UP_seq
        )
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
            mappedMotif, pidx = make_motif(
                UP_seq, MS_seq, motif_size, y_idx, center_idx, DoS_idx
            )
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
            mappedMotif, pidx = make_motif(
                UP_seq, MS_seq, motif_size, ts_idx, center_idx, DoS_idx
            )
            if len(pidx) == 1:
                pos = pidx[0]
            if len(pidx) > 1:
                pos = ";".join(pidx)

    except BaseException:
        print(MS_name + " not in ProteomeDict.")
        raise

    return pos, mappedMotif


def generate_kinase_motifs(
    names: Sequence[str], seqs: Sequence[str]
) -> tuple[list[str], list[str], list[str], list[int]]:
    """Main function to generate motifs using 'findmotif'.

    Loads the bundled UniProt proteome, matches each peptide to it (via
    `match_protein_names`), and extracts a sequence motif for each (via
    `find_motif`). Must be run with the repository root as the working
    directory (loads `./data/Sequence_analysis/proteome_uniprot2019.fa`).

    Args:
        names: Protein name reported for each MS peptide.
        seqs: The corresponding MS peptide sequence for each entry.

    Returns:
        A tuple `(MS_names, mapped_motifs, uni_pos, Xidx)` of the resolved
        protein name, extracted motif, phosphosite position, and original
        index for each peptide that could be matched to the proteome.
    """
    motif_size = 5
    proteome = open("./data/Sequence_analysis/proteome_uniprot2019.fa")
    ProteomeDict = get_proteome_name_to_seq(proteome, n="gene")
    protnames, seqs, Xidx = match_protein_names(ProteomeDict, names, seqs)
    (
        MS_names,
        mapped_motifs,
        uni_pos,
    ) = (
        [],
        [],
        [],
    )

    for i, MS_seq in enumerate(seqs):
        pos, mappedMotif = find_motif(MS_seq, protnames[i], ProteomeDict, motif_size)
        MS_names.append(protnames[i])
        mapped_motifs.append(mappedMotif)
        uni_pos.append(pos)

    proteome.close()
    return MS_names, mapped_motifs, uni_pos, Xidx


def make_motif(
    UP_seq: str,
    MS_seq: str,
    motif_size: int,
    ps_protein_idx: int,
    center_motif_idx: int,
    DoS_idx: Sequence[re.Match] | None,
) -> tuple[str, list[str]]:
    """Make a motif out of the matched sequences.

    Slices `motif_size` residues on each side of the phosphosite out of the
    full protein sequence (padding with `"-"` if the phosphosite is near a
    sequence end), lowercases the phosphoacceptor, and lowercases any other
    concurrently phosphorylated residues that fall within the motif.

    Args:
        UP_seq: The full UniProt protein sequence.
        MS_seq: The MS peptide sequence (used to locate concurrent
            phosphosites' original characters).
        motif_size: Number of residues to include on each side of the
            phosphoacceptor.
        ps_protein_idx: 0-indexed position of the primary phosphoacceptor
            within `UP_seq`.
        center_motif_idx: 0-indexed position of the primary phosphoacceptor
            within `MS_seq`.
        DoS_idx: Regex match objects locating any additional, concurrently
            phosphorylated residues within `MS_seq` (or `None`/empty if
            there are none).

    Returns:
        A tuple `(motif, pidx)`:
            motif: The length `2 * motif_size + 1` sequence motif, with
                each phosphorylated residue lowercased.
            pidx: The phosphosite position(s), formatted as
                `"{residue}{1-indexed position}-p"`, for the primary site
                and any concurrent site that falls within the motif.
    """
    UP_seq_copy = list(
        UP_seq[max(0, ps_protein_idx - motif_size) : ps_protein_idx + motif_size + 1]
    )
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
                assert UP_seq_copy[editPos] == MS_seq[ppIDX.start()], (
                    UP_seq_copy[editPos] + " " + MS_seq[ppIDX.start()]
                )
                if position != 0:
                    pidx.append(
                        str(UP_seq_copy[editPos]).upper()
                        + str(ps_protein_idx + position + 1)
                        + "-p"
                    )

    return "".join(UP_seq_copy), pidx


_KINOME_ATLAS_DIR = Path(__file__).resolve().parent / "data" / "PSPL" / "kinome_atlas"
_FLANKING_POSITIONS = (-5, -4, -3, -2, -1, 1, 2, 3, 4)


@lru_cache(maxsize=1)
def get_kinome_metadata() -> pd.DataFrame:
    """Load kinase metadata for the bundled human kinome PSPL atlas (404
    Ser/Thr and Tyr kinase profiles from Johnson et al. 2023 & Yaron-Barir
    et al. 2024), indexed by `MATRIX_NAME`.
    """
    return pd.read_csv(
        _KINOME_ATLAS_DIR / "kinome_information.tsv", sep="\t"
    ).set_index("MATRIX_NAME")


@lru_cache(maxsize=2)
def _load_pspls_cached(
    source: Literal["kinome_atlas", "legacy"],
) -> tuple[np.ndarray, np.ndarray]:
    if source == "kinome_atlas":
        cols = [f"{pos}{aa}" for pos in _FLANKING_POSITIONS for aa in AAlist]
        st = pd.read_csv(
            _KINOME_ATLAS_DIR / "ser_thr_all_log2_matrices.tsv",
            sep="\t",
            index_col=0,
        )[cols]
        tyr = pd.read_csv(
            _KINOME_ATLAS_DIR / "tyrosine_all_log2_matrices.tsv",
            sep="\t",
            index_col=0,
        )[cols]
        combined = pd.concat([st, tyr], axis=0)
        kinases = combined.index.to_numpy(dtype=str)
        # Reshape (n_kinases, 9 * 20) -> (n_kinases, 9, 20) -> (n_kinases, 20, 9)
        pspls = (
            combined.to_numpy(dtype=np.float64)
            .reshape(len(kinases), len(_FLANKING_POSITIONS), len(AAlist))
            .transpose(0, 2, 1)
        )
        return kinases, pspls

    if source != "legacy":
        raise ValueError(f"Unknown PSPL source: {source!r}")

    pspls_arr = []
    kinases_list = []
    # individual files
    PSPLs = glob.glob("./ddmc/data/PSPL/*.csv")
    for sp in PSPLs:
        if sp == "./ddmc/data/PSPL/pssm_data.csv":
            continue
        sp_mat = pd.read_csv(sp, index_col=0)
        sp_mat = sp_mat.loc[AAlist]

        if np.all(sp_mat >= 0):
            sp_mat = np.log2(sp_mat)

        kinases_list.append(sp.split("PSPL/")[1].split(".csv")[0])
        pspls_arr.append(sp_mat.values)

    # NetPhores PSPL results
    f = pd.read_csv("ddmc/data/PSPL/pssm_data.csv", header=None)
    matIDX = [np.arange(16) + i for i in range(0, f.shape[0], 16)]
    for ii in matIDX:
        kin = f.iloc[ii[0], 0]
        mat = f.iloc[ii[1:], :].T
        mat.columns = np.arange(mat.shape[1])
        mat = mat.iloc[:-1, 2:12].drop(8, axis=1).astype("float64").values
        mat = np.ma.log2(mat)
        mat = mat.filled(0)
        mat = np.clip(mat, a_min=0, a_max=3)
        kinases_list.append(kin)
        pspls_arr.append(mat)

    return np.array(kinases_list), np.array(pspls_arr)


def get_pspls(
    source: Literal["kinome_atlas", "legacy"] = "kinome_atlas",
) -> tuple[np.ndarray, np.ndarray]:
    """Load kinase specificity profiles (PSPLs) bundled in `ddmc/data/PSPL/`.

    Args:
        source: `"kinome_atlas"` (default) loads the complete 404-kinase human
            Ser/Thr (Johnson et al. 2023) and Tyr (Yaron-Barir et al. 2024)
            PSPL atlas. `"legacy"` loads the original 63-kinase set from
            `ddmc/data/PSPL/*.csv` and `pssm_data.csv`.

    Returns:
        A tuple `(kinases, pspls)`:
            kinases: Kinase name for each profile, of shape (n_kinases,).
            pspls: Specificity profile for each kinase, of shape
                (n_kinases, 20, 9), aligned to `kinases` and to `AAlist`
                along the amino acid axis.
    """
    kinases, pspls = _load_pspls_cached(source)
    return kinases.copy(), pspls.copy()


def compute_control_pssm(bg_sequences: Sequence[str]) -> np.ndarray:
    """Build a background position-specific scoring matrix (PSSM) from a set
    of (typically random/background) sequences, for use as the normalizing
    background in `ddmc.clustering.DDMC.get_pssms`.

    Args:
        bg_sequences: Length-11 background peptide sequences, e.g. from
            `ddmc.binomial.BackgroundSeqs`.

    Returns:
        Array of shape (len(AAlist), 11) giving the log2 amino acid
        enrichment at each position, normalized per-position across
        residues.
    """
    back_pssm = np.zeros((len(AAlist), 11), dtype=float)
    for _, seq in enumerate(bg_sequences):
        for kk, aa in enumerate(seq):
            back_pssm[AAlist.index(aa.upper()), kk] += 1.0
    for pos in range(back_pssm.shape[1]):
        back_pssm[:, pos] /= np.mean(back_pssm[:, pos])
    back_pssm = np.log2(back_pssm)
    return np.nan_to_num(back_pssm)


def annotate_flanking_ptms(
    p_signal: pd.DataFrame,
    phospho_motifs: Sequence[str] | None = None,
    acetyl_motifs: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Dynamically annotate flanking PTM sites on length-11 peptide motifs
    using co-measured phosphorylation and acetylation motifs.

    For each peptide `seq` in `p_signal.index`, each flanking position
    `pos = 5 + d` (`d in [-5, 5], d != 0`) is checked for an exact overlapping
    `(11 - |d|)`-mer match against `phospho_motifs` (annotating `S/T/Y` as
    lowercase `s/t/y`) and `acetyl_motifs` (annotating `K` as lowercase `k`).

    Args:
        p_signal: Phosphorylation signal DataFrame indexed by length-11 motifs.
        phospho_motifs: Co-measured length-11 phosphosite motifs. If `None`,
            uses `p_signal.index` itself.
        acetyl_motifs: Optional co-measured length-11 acetyl-lysine motifs
            (with `K` or `k` at position 5).

    Returns:
        Copy of `p_signal` with flanking PTM tokens annotated in its index.
    """
    p_motifs = list(p_signal.index) if phospho_motifs is None else list(phospho_motifs)
    a_motifs = [] if acetyl_motifs is None else list(acetyl_motifs)

    # Pre-index overlapping prefixes/suffixes for each offset d in [-5..5] \ {0}
    phospho_index: dict[int, set[str]] = {}
    acetyl_index: dict[int, set[str]] = {}
    for d in range(-5, 6):
        if d == 0:
            continue
        if d > 0:
            phospho_index[d] = {
                m[: 11 - d].upper() for m in p_motifs if m[5].upper() in ("S", "T", "Y")
            }
            acetyl_index[d] = {
                m[: 11 - d].upper() for m in a_motifs if m[5].upper() == "K"
            }
        else:
            phospho_index[d] = {
                m[-d:].upper() for m in p_motifs if m[5].upper() in ("S", "T", "Y")
            }
            acetyl_index[d] = {m[-d:].upper() for m in a_motifs if m[5].upper() == "K"}

    annotated: list[str] = []
    for raw_seq in p_signal.index:
        chars = list(str(raw_seq))
        upper_seq = "".join(chars).upper()
        for d in range(-5, 6):
            if d == 0:
                continue
            pos = 5 + d
            overlap = upper_seq[d:] if d > 0 else upper_seq[: 11 + d]
            aa = upper_seq[pos]
            if aa in ("S", "T", "Y") and overlap in phospho_index[d]:
                chars[pos] = aa.lower()
            elif aa == "K" and overlap in acetyl_index[d]:
                chars[pos] = "k"
        annotated.append("".join(chars))

    out = p_signal.copy()
    out.index = pd.Index(annotated, name=p_signal.index.name)
    return out


# Maps each kinase named in the PSPL data (ddmc/data/PSPL/) to the
# phosphoacceptor type(s) it targets, used by
# ddmc.figures.common.plot_cluster_kinase_distances to filter kinase
# predictions to those matching a cluster's dominant phosphoacceptor.
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
    "EphA3": "Y",
}

for _kin_name, _kin_type in get_kinome_metadata()["TYPE"].items():
    KinToPhosphotypeDict[str(_kin_name)] = "Y" if _kin_type == "tyrosine" else "S/T"
