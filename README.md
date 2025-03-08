# State Of Health Prediction on Power Electronics

This is part of a research project at the ILH at the University of Stuttgart to estimate the aging of power MOSFETs with Deep learning algorithms.

## Abstract of the finished project

Determining the RUL of SiC MOSFETs is crucial for their industrial application. One approach to estimating the RUL is deriving it from TSEPs. Ideally, these TSEPs should be relatively easy to measure and not require continuous monitoring. A possible method involves determining the RUL based on the RDS,on curve, specifically from a single RDS,on cycle. Since the patterns of these curves lack clear trends, they are analyzed using Artificial Intelligence (AI).
Initially, various model architectures are trained and tested on complete cycles. A comparison of these models shows that a CNN delivers the best results. However, the predictions remain very inaccurate. To improve accuracy and detect more complex patterns, the number of sequence data points is reduced. It is observed that with 50 instead of 9940 data points, a MAPE of 25.6 % is achieved. To further enhance accuracy, it is determined which measured variables, besides RDS,on, correlate most strongly with RUL. As a result, Vth and ∆T are identified. These variables are added as additional inputs to the model. ∆T is indirectly derived from Vth. Based on previous findings, Vth can be reduced to 25 data points. Each input variable is trained on its own path. The model combines the results internally. A CNN is used for RDS,on and Vth , while an MLP is employed for ∆T . In total, the model has 76 input parameters. The model achieves a MAPE of 3.956 % on train data.
This results in an R2 value of X on validation data. The models inaccuracy is particularly pronounced in the first 30 % of the RUL. Toward the end of the RUL, the model becomes highly accurate, allowing precise determination of the EOL.

## Usage of the repository

Please refer to the [documentation](documentation.md).
