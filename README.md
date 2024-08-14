# FoolsUNet
Implementation of a lightweight U-Net only a fool would use

## Test of channel attention mechanisms

| Mechanism | step duration | train best| val best |
|---|---|---|---|
| Squeeze and Excite| 410ms | 0.8663 | 0.7253 |
| Efficient Channel Attention | 409ms | 0.9088 | 0.7529 |
| - | 388ms | 0.8783 | 0.7135 |
