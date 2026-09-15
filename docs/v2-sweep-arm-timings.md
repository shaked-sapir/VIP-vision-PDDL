# v2 sweep — learning time per arm, per L

Array `20685590`, run `large-corpora__L-sweep__v2`.

Each figure is the mean of `learning_time_seconds` over that cell's 5 folds.
`—` means no fold completed at that L. This is **learning only**; a cell's
wall-clock is dominated by evaluation, which is not shown here.

| arm | key |
|---|---|
| PISAM(init) | `PISAM_MILP_LOOP__m=4` |
| PISAM(none) | `PISAM_MILP_LOOP__gt=none__m=4` |
| R_MILP | `ROSAME_MILP_24` |
| R_MILP_TAG | `ROSAME_MILP_24_TAG` |
| ROSAME_24 | `ROSAME_24` |

### task 0 — blocksworld, mask=0.0, noise=0.0
*COMPLETED · elapsed 05:14:58 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 1s | 1s | 2s | 5s | 16s |
| PISAM(none) | 1s | 2s | 2s | 5s | 18s |
| R_MILP | 6s | 32s | 49s | 232s | 15m |
| R_MILP_TAG | 5s | 25s | 48s | 221s | 15m |
| ROSAME_24 | 9s | 57s | 114s | 571s | 38m |

### task 1 — blocksworld, mask=0.0, noise=0.1
*COMPLETED · elapsed 07:37:48 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 46s | 129s | 160s | 438s | 25m |
| PISAM(none) | 56s | 140s | 205s | 485s | 26m |
| R_MILP | 7s | 31s | 58s | 296s | 19m |
| R_MILP_TAG | 7s | 31s | 57s | 294s | 18m |
| ROSAME_24 | 11s | 71s | 138s | 12m | 32m |

### task 2 — blocksworld, mask=0.0, noise=0.2
*COMPLETED · elapsed 10:18:30 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 49s | 124s | 166s | 450s | 26m |
| PISAM(none) | 289s | 11m | 14m | 28m | 47m |
| R_MILP | 42s | 27s | 67s | 247s | 18m |
| R_MILP_TAG | 35s | 30s | 50s | 302s | 17m |
| ROSAME_24 | 12s | 56s | 104s | 12m | 36m |

### task 3 — blocksworld, mask=0.01, noise=0.0
*COMPLETED · elapsed 05:07:11 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 1s | 1s | 2s | 5s | 16s |
| PISAM(none) | 1s | 2s | 2s | 5s | 18s |
| R_MILP | 6s | 25s | 46s | 229s | 20m |
| R_MILP_TAG | 6s | 25s | 47s | 228s | 15m |
| ROSAME_24 | 9s | 40s | 81s | 412s | 28m |

### task 4 — blocksworld, mask=0.01, noise=0.1
*COMPLETED · elapsed 05:48:17 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 39s | 99s | 137s | 370s | 22m |
| PISAM(none) | 45s | 113s | 154s | 390s | 22m |
| R_MILP | 6s | 24s | 47s | 220s | 15m |
| R_MILP_TAG | 6s | 24s | 47s | 220s | 15m |
| ROSAME_24 | 9s | 38s | 77s | 389s | 26m |

### task 5 — blocksworld, mask=0.01, noise=0.2
*COMPLETED · elapsed 08:13:23 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 52s | 124s | 167s | 430s | 25m |
| PISAM(none) | 337s | 12m | 17m | 25m | 48m |
| R_MILP | 37s | 27s | 52s | 256s | 17m |
| R_MILP_TAG | 25s | 27s | 60s | 255s | 17m |
| ROSAME_24 | 11s | 63s | 125s | 488s | 29m |

### task 6 — blocksworld, mask=0.1, noise=0.0
*COMPLETED · elapsed 05:17:35 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 2s | 2s | 2s | 5s | 15s |
| PISAM(none) | 1s | 2s | 2s | 6s | 16s |
| R_MILP | 7s | 24s | 52s | 256s | 15m |
| R_MILP_TAG | 6s | 25s | 58s | 245s | 15m |
| ROSAME_24 | 9s | 51s | 103s | 505s | 32m |

### task 7 — blocksworld, mask=0.1, noise=0.1
*COMPLETED · elapsed 06:00:36 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 38s | 95s | 133s | 355s | 20m |
| PISAM(none) | 44s | 113s | 153s | 377s | 21m |
| R_MILP | 5s | 23s | 45s | 220s | 15m |
| R_MILP_TAG | 5s | 23s | 44s | 219s | 15m |
| ROSAME_24 | 8s | 38s | 77s | 389s | 26m |

### task 8 — blocksworld, mask=0.1, noise=0.2
*COMPLETED · elapsed 08:59:39 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 56s | 140s | 196s | 527s | 32m |
| PISAM(none) | 279s | 554s | 12m | 24m | 48m |
| R_MILP | 20s | 36s | 73s | 342s | 24m |
| R_MILP_TAG | 11s | 37s | 73s | 337s | 24m |
| ROSAME_24 | 10s | 64s | 131s | 11m | 44m |

### task 9 — hanoi, mask=0.0, noise=0.0
*COMPLETED · elapsed 08:37:41 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 2s | 4s | 5s | 16s | 62s |
| PISAM(none) | 3s | 5s | 6s | 16s | 52s |
| R_MILP | 11s | 35s | 66s | 372s | 21m |
| R_MILP_TAG | 10s | 43s | 67s | 402s | 22m |
| ROSAME_24 | 15s | 66s | 103s | 10m | 45m |

### task 10 — hanoi, mask=0.0, noise=0.1
*COMPLETED · elapsed 17:22:23 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 87s | 212s | 308s | 16m | 67m |
| PISAM(none) | 123s | 288s | 405s | 18m | 70m |
| R_MILP | 103s | 127s | 115s | 330s | 23m |
| R_MILP_TAG | 47s | 100s | 78s | 324s | 23m |
| ROSAME_24 | 11s | 50s | 99s | 493s | 35m |

### task 11 — hanoi, mask=0.0, noise=0.2
*COMPLETED · elapsed 18:28:07 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 114s | 269s | 372s | 18m | 69m |
| PISAM(none) | 62m | 64m | 66m | 77m | 122m |
| R_MILP | 107s | 155s | 205s | 12m | 48m |
| R_MILP_TAG | 65s | 52s | 80s | 311s | 22m |
| ROSAME_24 | 11s | 51s | 97s | 481s | 33m |

### task 12 — hanoi, mask=0.01, noise=0.0
*COMPLETED · elapsed 07:37:04 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 3s | 4s | 5s | 16s | 52s |
| PISAM(none) | 2s | 4s | 5s | 15s | 53s |
| R_MILP | 9s | 36s | 68s | 329s | 21m |
| R_MILP_TAG | 9s | 36s | 69s | 324s | 21m |
| ROSAME_24 | 12s | 54s | 104s | 503s | 33m |

### task 13 — hanoi, mask=0.01, noise=0.1
*COMPLETED · elapsed 14:17:01 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 115s | 267s | 367s | 21m | 75m |
| PISAM(none) | 146s | 372s | 10m | 24m | 80m |
| R_MILP | 130s | 200s | 202s | 572s | 27m |
| R_MILP_TAG | 43s | 123s | 139s | 470s | 27m |
| ROSAME_24 | 15s | 75s | 166s | 14m | 48m |

### task 14 — hanoi, mask=0.01, noise=0.2 ⚠️
*TIMEOUT · elapsed 1-00:00:04 · 20/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 108s | 263s | 365s | 18m | — |
| PISAM(none) | 61m | 64m | 65m | 77m | — |
| R_MILP | 108s | 161s | 217s | 13m | — |
| R_MILP_TAG | 99s | 37s | 70s | 328s | — |
| ROSAME_24 | 12s | 53s | 105s | 12m | — |

### task 15 — hanoi, mask=0.1, noise=0.0
*COMPLETED · elapsed 07:34:03 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 3s | 4s | 5s | 15s | 50s |
| PISAM(none) | 3s | 4s | 5s | 15s | 52s |
| R_MILP | 9s | 36s | 68s | 336s | 22m |
| R_MILP_TAG | 8s | 35s | 68s | 330s | 22m |
| ROSAME_24 | 11s | 61s | 100s | 566s | 33m |

### task 16 — hanoi, mask=0.1, noise=0.1
*COMPLETED · elapsed 14:46:09 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 85s | 207s | 315s | 15m | 56m |
| PISAM(none) | 146s | 329s | 483s | 18m | 59m |
| R_MILP | 103s | 151s | 216s | 12m | 54m |
| R_MILP_TAG | 64s | 140s | 210s | 12m | 48m |
| ROSAME_24 | 11s | 73s | 144s | 523s | 33m |

### task 17 — hanoi, mask=0.1, noise=0.2 ⚠️
*TIMEOUT · elapsed 1-00:00:26 · 20/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 116s | 284s | 399s | 22m | — |
| PISAM(none) | 57m | 64m | 66m | 80m | — |
| R_MILP | 104s | 187s | 220s | 16m | — |
| R_MILP_TAG | 99s | 86s | 97s | 422s | — |
| ROSAME_24 | 17s | 84s | 144s | 11m | — |

### task 18 — npuzzle, mask=0.0, noise=0.0
*COMPLETED · elapsed 09:16:13 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 5s | 9s | 13s | 46s | 175s |
| PISAM(none) | 6s | 9s | 12s | 47s | 180s |
| R_MILP | 12s | 47s | 83s | 399s | 27m |
| R_MILP_TAG | 13s | 43s | 82s | 400s | 27m |
| ROSAME_24 | 10s | 48s | 82s | 414s | 28m |

### task 19 — npuzzle, mask=0.0, noise=0.1
*COMPLETED · elapsed 13:59:59 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 180s | 430s | 587s | 24m | 83m |
| PISAM(none) | 182s | 438s | 11m | 25m | 84m |
| R_MILP | 61s | 100s | 101s | 435s | 29m |
| R_MILP_TAG | 14s | 49s | 94s | 428s | 30m |
| ROSAME_24 | 9s | 42s | 98s | 444s | 30m |

### task 20 — npuzzle, mask=0.0, noise=0.2
*COMPLETED · elapsed 18:20:22 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 179s | 425s | 10m | 26m | 86m |
| PISAM(none) | 187s | 445s | 10m | 25m | 83m |
| R_MILP | 218s | 278s | 358s | 17m | 56m |
| R_MILP_TAG | 51s | 47s | 90s | 446s | 28m |
| ROSAME_24 | 9s | 42s | 90s | 433s | 29m |

### task 21 — npuzzle, mask=0.01, noise=0.0
*COMPLETED · elapsed 11:04:11 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 7s | 13s | 20s | 53s | 207s |
| PISAM(none) | 6s | 12s | 16s | 56s | 201s |
| R_MILP | 15s | 71s | 130s | 529s | 31m |
| R_MILP_TAG | 15s | 59s | 118s | 524s | 30m |
| ROSAME_24 | 13s | 66s | 136s | 474s | 37m |

### task 22 — npuzzle, mask=0.01, noise=0.1
*COMPLETED · elapsed 15:57:26 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 183s | 426s | 579s | 30m | 94m |
| PISAM(none) | 198s | 569s | 12m | 32m | 94m |
| R_MILP | 16s | 52s | 96s | 450s | 30m |
| R_MILP_TAG | 17s | 68s | 123s | 466s | 36m |
| ROSAME_24 | 10s | 43s | 135s | 581s | 31m |

### task 23 — npuzzle, mask=0.01, noise=0.2
*COMPLETED · elapsed 20:16:58 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 173s | 434s | 10m | 27m | 97m |
| PISAM(none) | 183s | 485s | 12m | 28m | 106m |
| R_MILP | 259s | 288s | 391s | 16m | 70m |
| R_MILP_TAG | 49s | 52s | 108s | 445s | 35m |
| ROSAME_24 | 11s | 52s | 109s | 475s | 31m |

### task 24 — npuzzle, mask=0.1, noise=0.0
*COMPLETED · elapsed 09:20:43 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 6s | 9s | 14s | 48s | 184s |
| PISAM(none) | 6s | 9s | 14s | 53s | 189s |
| R_MILP | 13s | 46s | 88s | 420s | 30m |
| R_MILP_TAG | 12s | 44s | 89s | 419s | 29m |
| ROSAME_24 | 9s | 42s | 88s | 430s | 30m |

### task 25 — npuzzle, mask=0.1, noise=0.1
*COMPLETED · elapsed 16:03:27 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 200s | 473s | 11m | 30m | 84m |
| PISAM(none) | 203s | 463s | 11m | 28m | 85m |
| R_MILP | 159s | 206s | 254s | 526s | 47m |
| R_MILP_TAG | 64s | 65s | 120s | 10m | 42m |
| ROSAME_24 | 11s | 58s | 99s | 568s | 33m |

### task 26 — npuzzle, mask=0.1, noise=0.2
*COMPLETED · elapsed 18:30:16 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 182s | 476s | 11m | 26m | 103m |
| PISAM(none) | 200s | 519s | 11m | 32m | 101m |
| R_MILP | 241s | 302s | 445s | 19m | 80m |
| R_MILP_TAG | 58s | 54s | 104s | 587s | 40m |
| ROSAME_24 | 13s | 44s | 91s | 455s | 43m |

### task 27 — depot, mask=0.0, noise=0.0
*COMPLETED · elapsed 10:26:20 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 2s | 4s | 8s | 28s | 39m |
| PISAM(none) | 2s | 5s | 7s | 27s | 37m |
| R_MILP | 11s | 50s | 89s | 382s | 28m |
| R_MILP_TAG | 11s | 43s | 84s | 412s | 25m |
| ROSAME_24 | 14s | 74s | 196s | 10m | 38m |

### task 28 — depot, mask=0.0, noise=0.1
*COMPLETED · elapsed 12:45:38 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 76s | 198s | 282s | 14m | 50m |
| PISAM(none) | 109s | 256s | 415s | 16m | 63m |
| R_MILP | 96s | 163s | 241s | 502s | 25m |
| R_MILP_TAG | 43s | 78s | 157s | 433s | 25m |
| ROSAME_24 | 12s | 56s | 110s | 573s | 36m |

### task 29 — depot, mask=0.0, noise=0.2
*COMPLETED · elapsed 20:04:09 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 321s | 13m | 21m | 40m | 78m |
| PISAM(none) | 57m | 64m | 66m | 75m | 120m |
| R_MILP | 105s | 176s | 240s | 17m | 54m |
| R_MILP_TAG | 90s | 177s | 180s | 396s | 25m |
| ROSAME_24 | 13s | 68s | 132s | 544s | 43m |

### task 30 — depot, mask=0.01, noise=0.0
*COMPLETED · elapsed 09:39:51 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 2s | 4s | 6s | 25s | 39m |
| PISAM(none) | 2s | 4s | 6s | 25s | 39m |
| R_MILP | 10s | 43s | 78s | 370s | 25m |
| R_MILP_TAG | 10s | 43s | 75s | 432s | 25m |
| ROSAME_24 | 12s | 55s | 109s | 549s | 37m |

### task 31 — depot, mask=0.01, noise=0.1
*COMPLETED · elapsed 12:24:53 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 83s | 192s | 279s | 14m | 52m |
| PISAM(none) | 114s | 253s | 392s | 16m | 80m |
| R_MILP | 100s | 164s | 236s | 13m | 26m |
| R_MILP_TAG | 66s | 113s | 228s | 11m | 26m |
| ROSAME_24 | 12s | 57s | 110s | 553s | 36m |

### task 32 — depot, mask=0.01, noise=0.2 ⚠️
*RUNNING · elapsed 22:25:34 · 22/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 547s | 15m | 32m | 52m | 76m |
| PISAM(none) | 53m | 64m | 67m | 75m | 112m |
| R_MILP | 116s | 217s | 245s | 17m | 50m |
| R_MILP_TAG | 101s | 152s | 176s | 516s | 23m |
| ROSAME_24 | 18s | 70s | 169s | 12m | 50m |

### task 33 — depot, mask=0.1, noise=0.0
*COMPLETED · elapsed 10:00:04 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 2s | 5s | 7s | 25s | 34m |
| PISAM(none) | 3s | 5s | 6s | 24s | 34m |
| R_MILP | 28s | 41s | 80s | 380s | 25m |
| R_MILP_TAG | 10s | 40s | 77s | 445s | 26m |
| ROSAME_24 | 12s | 56s | 111s | 555s | 36m |

### task 34 — depot, mask=0.1, noise=0.1
*COMPLETED · elapsed 17:24:05 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 77s | 194s | 273s | 13m | 46m |
| PISAM(none) | 133s | 331s | 447s | 17m | 64m |
| R_MILP | 95s | 161s | 240s | 14m | 56m |
| R_MILP_TAG | 72s | 133s | 233s | 14m | 55m |
| ROSAME_24 | 12s | 58s | 115s | 548s | 37m |

### task 35 — depot, mask=0.1, noise=0.2
*COMPLETED · elapsed 21:00:14 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 273s | 15m | 23m | 43m | 69m |
| PISAM(none) | 61m | 64m | 65m | 77m | 124m |
| R_MILP | 99s | 168s | 263s | 16m | 78m |
| R_MILP_TAG | 91s | 135s | 175s | 386s | 24m |
| ROSAME_24 | 12s | 57s | 128s | 534s | 36m |

### task 36 — gripper, mask=0.0, noise=0.0
*COMPLETED · elapsed 05:22:11 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 1s | 2s | 2s | 6s | 19s |
| PISAM(none) | 1s | 1s | 2s | 5s | 18s |
| R_MILP | 7s | 28s | 54s | 312s | 17m |
| R_MILP_TAG | 7s | 28s | 54s | 273s | 21m |
| ROSAME_24 | 10s | 54s | 110s | 472s | 32m |

### task 37 — gripper, mask=0.0, noise=0.1
*COMPLETED · elapsed 06:24:15 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 34s | 90s | 125s | 376s | 22m |
| PISAM(none) | 35s | 90s | 129s | 379s | 23m |
| R_MILP | 7s | 44s | 77s | 349s | 18m |
| R_MILP_TAG | 7s | 30s | 59s | 287s | 18m |
| ROSAME_24 | 10s | 49s | 96s | 483s | 33m |

### task 38 — gripper, mask=0.0, noise=0.2
*COMPLETED · elapsed 10:40:48 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 49s | 164s | 186s | 551s | 41m |
| PISAM(none) | 51s | 172s | 168s | 588s | 37m |
| R_MILP | 75s | 150s | 236s | 16m | 67m |
| R_MILP_TAG | 11s | 50s | 89s | 502s | 26m |
| ROSAME_24 | 13s | 83s | 168s | 14m | 56m |

### task 39 — gripper, mask=0.01, noise=0.0
*COMPLETED · elapsed 06:23:32 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 1s | 1s | 2s | 6s | 20s |
| PISAM(none) | 1s | 1s | 2s | 6s | 25s |
| R_MILP | 7s | 29s | 62s | 282s | 22m |
| R_MILP_TAG | 7s | 34s | 67s | 332s | 23m |
| ROSAME_24 | 11s | 62s | 125s | 496s | 33m |

### task 40 — gripper, mask=0.01, noise=0.1
*COMPLETED · elapsed 06:57:20 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 35s | 92s | 133s | 402s | 26m |
| PISAM(none) | 36s | 94s | 133s | 405s | 25m |
| R_MILP | 18s | 72s | 119s | 358s | 27m |
| R_MILP_TAG | 8s | 45s | 61s | 289s | 19m |
| ROSAME_24 | 11s | 49s | 98s | 495s | 33m |

### task 41 — gripper, mask=0.01, noise=0.2
*COMPLETED · elapsed 06:43:28 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 36s | 101s | 122s | 392s | 24m |
| PISAM(none) | 38s | 97s | 125s | 396s | 24m |
| R_MILP | 55s | 108s | 143s | 10m | 36m |
| R_MILP_TAG | 15s | 30s | 52s | 275s | 17m |
| ROSAME_24 | 11s | 48s | 91s | 459s | 30m |

### task 42 — gripper, mask=0.1, noise=0.0
*COMPLETED · elapsed 05:45:45 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 1s | 2s | 2s | 6s | 18s |
| PISAM(none) | 1s | 1s | 2s | 5s | 17s |
| R_MILP | 7s | 31s | 61s | 255s | 20m |
| R_MILP_TAG | 7s | 33s | 56s | 300s | 18m |
| ROSAME_24 | 12s | 61s | 109s | 460s | 37m |

### task 43 — gripper, mask=0.1, noise=0.1
*COMPLETED · elapsed 06:51:07 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 37s | 94s | 125s | 359s | 25m |
| PISAM(none) | 38s | 87s | 125s | 388s | 22m |
| R_MILP | 37s | 61s | 85s | 307s | 19m |
| R_MILP_TAG | 8s | 31s | 56s | 269s | 18m |
| ROSAME_24 | 10s | 49s | 92s | 470s | 32m |

### task 44 — gripper, mask=0.1, noise=0.2
*COMPLETED · elapsed 06:48:42 · 25/25 folds*

| arm | L=10 | L=50 | L=100 | L=500 | L=2000 |
|---|---|---|---|---|---|
| PISAM(init) | 42s | 94s | 122s | 358s | 24m |
| PISAM(none) | 42s | 101s | 127s | 363s | 25m |
| R_MILP | 65s | 108s | 140s | 546s | 35m |
| R_MILP_TAG | 9s | 32s | 56s | 256s | 19m |
| ROSAME_24 | 12s | 55s | 91s | 451s | 30m |
