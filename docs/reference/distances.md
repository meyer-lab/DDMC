# Sequence distance models

DDMC supports two interchangeable ways of scoring how well a peptide
sequence matches a cluster's motif, selected via `DDMC(..., distance_method=...)`.

## `ddmc.binomial`

::: ddmc.binomial
    options:
      members:
        - Binomial
        - BackgroundSeqs
        - BackgProportions
        - CountPsiteTypes
        - position_weight_matrix
        - fast_position_weight_matrix
        - frequencies
        - GenerateBinarySeqID

## `ddmc.pam250`

::: ddmc.pam250
    options:
      members:
        - PAM250
        - get_pam250_scores
