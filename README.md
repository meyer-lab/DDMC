# Co-clustering for mass spectrometry peptide analysis

![Build](https://github.com/meyer-lab/resistance-MS/workflows/Build/badge.svg)
![Test](https://github.com/meyer-lab/resistance-MS/workflows/Test/badge.svg)

Clusters peptides based on both sequence similarity and phosphorylation signal across samples.


## Usage

```
>>> from ddmc.clustering import DDMC

>>> # load dataset as p_signal...

>>> p_signal
             Sample 1  Sample 2  Sample 3  Sample 4  Sample 5
Sequence                                                     
AAAAAsQQGSA -3.583614       NaN -0.662659 -1.320029 -0.730832
AAAAGsASPRS -0.174779 -1.796899  0.891798 -3.092941  2.394315
AAAAGsGPSPP -1.951552 -2.937095  2.692876 -2.344894  0.556615
AAAAGsGPsPP  3.666782       NaN -2.081231  0.989394       NaN
AAAAPsPGSAR  1.753855 -2.135835  0.896778  3.369230  2.020967
...               ...       ...       ...       ...       ...
YYSPYsVSGSG -3.502871  2.831169  3.383486  2.589559  3.624968
YYSSRsQSGGY -0.870365  0.887317  2.600291 -0.374107  3.285459
YYTAGyNSPVK  0.249539  2.047050 -0.286033  0.042650  2.863317
YYTSAsGDEMV  0.662787  0.135326 -1.004350  0.879398 -1.609894
YYYSSsEDEDS       NaN -1.101679 -3.273987 -0.872370 -1.735891

>>> p_signal.index  # p_signal.index contains the peptide sequences
Index(['AAAAAsQQGSA', 'AAAAGsASPRS', 'AAAAGsGPSPP', 'AAAAGsGPsPP',
       'AAAAPsPGSAR', 'AAAAPsPGsAR', 'AAAARsLLNHT', 'AAAARsPDRNL',
       'AAAARtQAPPT', 'AAADFsDEDED',
       ...
       'YYDRMySYPAR', 'YYEDDsEGEDI', 'YYGGGsEGGRA', 'YYRNNsFTAPS',
       'YYSPDyGLPSP', 'YYSPYsVSGSG', 'YYSSRsQSGGY', 'YYTAGyNSPVK',
       'YYTSAsGDEMV', 'YYYSSsEDEDS'],
      dtype='object', name='Sequence', length=30561)
      
>>> model = DDMC(n_components=2, seq_weight=100).fit(p_signal)  # fit model

>>> model.transform(as_df=True)  # get cluster centers
                 0         1
Sample 1  0.017644  0.370375
Sample 2 -0.003625 -0.914869
Sample 3 -0.087624 -0.682140
Sample 4  0.014644 -0.658907
Sample 5  0.023885  0.196063
```