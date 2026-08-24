# `ddmc.logistic_regression`

Helpers for using DDMC cluster centers as features in a logistic regression
classifier, to predict clinical/genetic features of CPTAC patients (e.g.
mutation status, tumor vs. NAT, hot/cold immune infiltration).

::: ddmc.logistic_regression
    options:
      members:
        - normalize_cluster_centers
        - get_highest_weighted_clusters
        - plot_cluster_regression_coefficients
        - plot_roc
