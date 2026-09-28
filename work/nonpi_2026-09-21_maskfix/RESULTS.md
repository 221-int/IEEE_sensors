# Complete source-domain experiment results

All 15 fits per candidate verified. Manuscript editing remains on hold.

| Model | AP mean | AP SD | Params | MMAC |
|---|---:|---:|---:|---:|
| vpres | 0.988227 | 0.005699 | 84049 | 12.489536 |
| image_head | 0.990494 | 0.004194 | 476161 | 31.852480 |
| vdrop | 0.986351 | 0.006969 | 55377 | 3.490624 |
| gap | 0.982621 | 0.015122 | 37889 | 31.414208 |

Paired subject-bootstrap AP differences (95% percentile intervals):

- vpres_minus_image_head: -0.003547 [-0.006487, -0.001382]
- vpres_minus_gap: +0.005648 [-0.000939, +0.014908]
- vdrop_minus_vpres: -0.001719 [-0.003904, +0.000176]
- vdrop_minus_gap: +0.003929 [-0.003207, +0.013395]
- vdrop_minus_image_head: -0.005266 [-0.008908, -0.002359]

These are exploratory results. Do not change the paper or designate a winner without checking uncertainty, convergence and external validation.