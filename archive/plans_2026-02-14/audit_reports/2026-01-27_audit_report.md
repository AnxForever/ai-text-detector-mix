# 数据集审计报告

> 生成日期: 2026-01-27

## final_clean

**Train/Val/Test 重复 (Exact text hash)**
- train ∩ val: 0
- train ∩ test: 0
- val ∩ test: 0
- train ∩ val ∩ test: 0

### train
- rows: 44350
- unique_rows: 44350
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 3810
- labels: {'1': 22600, '0': 21750}
- length(min/median/mean/max): 50 / 246.0 / 614.72 / 9080
- length_buckets: <80: 3055 (6.89%), 80-200: 15010 (33.84%), 200-500: 11685 (26.35%), 500-1000: 5251 (11.84%), 1000-2000: 5885 (13.27%), 2000+: 0 (0.0%)

### val
- rows: 5544
- unique_rows: 5544
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 456
- labels: {'0': 2719, '1': 2825}
- length(min/median/mean/max): 50 / 247.0 / 618.89 / 6212
- length_buckets: <80: 364 (6.57%), 80-200: 1897 (34.22%), 200-500: 1463 (26.39%), 500-1000: 663 (11.96%), 1000-2000: 703 (12.68%), 2000+: 0 (0.0%)

### test
- rows: 5544
- unique_rows: 5544
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 476
- labels: {'1': 2825, '0': 2719}
- length(min/median/mean/max): 50 / 241.0 / 610.78 / 7856
- length_buckets: <80: 378 (6.82%), 80-200: 1906 (34.38%), 200-500: 1468 (26.48%), 500-1000: 636 (11.47%), 1000-2000: 726 (13.1%), 2000+: 0 (0.0%)

### full
- rows: 55438
- unique_rows: 55438
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 4742
- labels: {'Human': 27188, 'AI': 28250}
- length(min/median/mean/max): 50 / 246.0 / 614.74 / 9080
- length_buckets: <80: 3797 (6.85%), 80-200: 18813 (33.94%), 200-500: 14616 (26.36%), 500-1000: 6550 (11.82%), 1000-2000: 7314 (13.19%), 2000+: 0 (0.0%)

### all_ai
- rows: 28250
- unique_rows: 28250
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 4655
- labels: {'AI': 28250}
- length(min/median/mean/max): 50 / 297.0 / 786.49 / 9080
- length_buckets: <80: 1304 (4.62%), 80-200: 8974 (31.77%), 200-500: 7330 (25.95%), 500-1000: 2193 (7.76%), 1000-2000: 4691 (16.61%), 2000+: 0 (0.0%)

### all_human
- rows: 27188
- unique_rows: 27188
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 87
- labels: {'Human': 27188}
- length(min/median/mean/max): 50 / 205.0 / 436.28 / 7063
- length_buckets: <80: 2493 (9.17%), 80-200: 9839 (36.19%), 200-500: 7286 (26.8%), 500-1000: 4357 (16.03%), 1000-2000: 2623 (9.65%), 2000+: 0 (0.0%)

## combined_v2

**Train/Val/Test 重复 (Exact text hash)**
- train ∩ val: 595
- train ∩ test: 569
- val ∩ test: 37
- train ∩ val ∩ test: 3

### train
- rows: 52800
- unique_rows: 49546
- dup_rows: 3254
- sep_rows: 1614
- phrase_rows: 4144
- labels: {'1': 27450, '0': 25350}
- length(min/median/mean/max): 21 / 277.0 / 608.34 / 9080
- length_buckets: <80: 3502 (6.63%), 80-200: 16573 (31.39%), 200-500: 13961 (26.44%), 500-1000: 8201 (15.53%), 1000-2000: 6937 (13.14%), 2000+: 0 (0.0%)

### val
- rows: 6600
- unique_rows: 6581
- dup_rows: 19
- sep_rows: 209
- phrase_rows: 491
- labels: {'0': 3169, '1': 3431}
- length(min/median/mean/max): 32 / 281.0 / 600.19 / 7189
- length_buckets: <80: 426 (6.45%), 80-200: 2062 (31.24%), 200-500: 1783 (27.02%), 500-1000: 1020 (15.45%), 1000-2000: 885 (13.41%), 2000+: 0 (0.0%)

### test
- rows: 6601
- unique_rows: 6572
- dup_rows: 29
- sep_rows: 211
- phrase_rows: 514
- labels: {'1': 3432, '0': 3169}
- length(min/median/mean/max): 21 / 287 / 602.3 / 7293
- length_buckets: <80: 434 (6.57%), 80-200: 2023 (30.65%), 200-500: 1762 (26.69%), 500-1000: 1060 (16.06%), 1000-2000: 898 (13.6%), 2000+: 0 (0.0%)

### test_hybrid_only
- rows: 1092
- unique_rows: 1088
- dup_rows: 4
- sep_rows: 211
- phrase_rows: 33
- labels: {'0': 472, '1': 620}
- length(min/median/mean/max): 21 / 498.0 / 574.82 / 3341
- length_buckets: <80: 60 (5.49%), 80-200: 181 (16.58%), 200-500: 307 (28.11%), 500-1000: 377 (34.52%), 1000-2000: 152 (13.92%), 2000+: 0 (0.0%)

## hybrid

**Train/Val/Test 重复 (Exact text hash)**
- train ∩ val: 0
- train ∩ test: 0
- val ∩ test: 0
- train ∩ val ∩ test: 0

### train
- rows: 4050
- unique_rows: 4050
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 109
- labels: {'1': 2850, '0': 1200}
- length(min/median/mean/max): 31 / 479.0 / 538.16 / 2994
- length_buckets: <80: 159 (3.93%), 80-200: 639 (15.78%), 200-500: 1364 (33.68%), 500-1000: 1431 (35.33%), 1000-2000: 433 (10.69%), 2000+: 0 (0.0%)

### val
- rows: 506
- unique_rows: 506
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 15
- labels: {'1': 356, '0': 150}
- length(min/median/mean/max): 31 / 471.5 / 540.03 / 2935
- length_buckets: <80: 22 (4.35%), 80-200: 81 (16.01%), 200-500: 169 (33.4%), 500-1000: 187 (36.96%), 1000-2000: 41 (8.1%), 2000+: 0 (0.0%)

### test
- rows: 507
- unique_rows: 507
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 15
- labels: {'1': 357, '0': 150}
- length(min/median/mean/max): 31 / 481 / 538.98 / 2987
- length_buckets: <80: 21 (4.14%), 80-200: 79 (15.58%), 200-500: 171 (33.73%), 500-1000: 182 (35.9%), 1000-2000: 49 (9.66%), 2000+: 0 (0.0%)

### hybrid_dataset
- rows: 5063
- unique_rows: 5063
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 139
- labels: {'1': 3563, '0': 1500}
- length(min/median/mean/max): 31 / 478 / 538.43 / 2994
- length_buckets: <80: 202 (3.99%), 80-200: 799 (15.78%), 200-500: 1704 (33.66%), 500-1000: 1800 (35.55%), 1000-2000: 523 (10.33%), 2000+: 0 (0.0%)

### hybrid_dataset_expanded
- rows: 7563
- unique_rows: 7563
- dup_rows: 0
- sep_rows: 0
- phrase_rows: 396
- labels: {'1': 6063, '0': 1500}
- length(min/median/mean/max): 31 / 561 / 616.93 / 2994
- length_buckets: <80: 209 (2.76%), 80-200: 831 (10.99%), 200-500: 2123 (28.07%), 500-1000: 3228 (42.68%), 1000-2000: 1122 (14.84%), 2000+: 0 (0.0%)

### hybrid_dataset_with_sep
- rows: 7563
- unique_rows: 7563
- dup_rows: 0
- sep_rows: 2034
- phrase_rows: 396
- labels: {'1': 6063, '0': 1500}
- length(min/median/mean/max): 31 / 564 / 618.28 / 2994
- length_buckets: <80: 208 (2.75%), 80-200: 825 (10.91%), 200-500: 2105 (27.83%), 500-1000: 3251 (42.99%), 1000-2000: 1124 (14.86%), 2000+: 0 (0.0%)

## Cross-dataset overlap
- final_clean_full_vs_combined_v2: 55438
