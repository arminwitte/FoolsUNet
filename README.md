# FoolsUNet
Implementation of a lightweight U-Net only a fool would use

## Test of channel attention mechanisms

| Mechanism | step duration | train best| val best |
|---|---|---|---|
| Squeeze and Excite| 410ms | 0.8663 | 0.7253 |
| Efficient Channel Attention | 409ms | 0.9088 | 0.7529 |
| w/o attention | 388ms | 0.8783 | 0.7135 |
| SE ASPP | 780ms | 0.8588 | 0.7435 |
| ECA ASPP | 780ms | 0.9000 | 0.8471 |
| ASPP w/o attention | 760ms | 0.8918 | 0.8358 |
| ECA ASPP w/ GELU | 820ms | 0.9070 | 0.7682 |
