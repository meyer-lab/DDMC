# `ddmc.motifs`

Helpers for mapping peptides to their surrounding sequence motif via a
reference proteome, and for loading kinase specificity profiles (PSPLs)
used by `DDMC.predict_upstream_kinases`.

::: ddmc.motifs
    options:
      members:
        - get_proteome_name_to_seq
        - match_protein_names
        - find_motif
        - generate_kinase_motifs
        - get_pspls
        - compute_control_pssm
        - KinToPhosphotypeDict
