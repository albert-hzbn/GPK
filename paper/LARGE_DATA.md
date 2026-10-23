# Large simulation fields

The figures use 72 snapshot fields (28.85 GB in total) that are too large for GitHub, whose limit is 100 MB per file. They are not in this repository. Everything else the figure scripts read is in `paper/data/`.

To get these fields, either:

* **regenerate them** with the command listed for each folder below (runs write to `output/`, where the figure scripts find them automatically), or
* **use an archived copy**, if one is listed in the main README, and point the scripts to it with `GPK_DATA=/path/to/data`.

The SHA-256 checksums identify the exact files used for the manuscript figures.

| Folder | Files | Size | Figures | Regenerate with |
|---|---:|---:|---|---|
| `alcu_validation/precipitation/three_dimensional_mesh_fine` | 4 | 78 MB | Suppl. Figs S9, S10 | `python paper/run_simulations.py --only alcu` |
| `alcu_validation/precipitation/three_dimensional_mesh_ultrafine` | 4 | 184 MB | Suppl. Figs S9, S10 | `python paper/run_simulations.py --only alcu` |
| `alcu_validation/precipitation/three_dimensional_ultrafine` | 1 | 56 MB | Suppl. Figs S9, S10 | `python paper/run_simulations.py --only alcu` |
| `fig7_large/dendrite_2d` | 2 | 159 MB | Fig. 9 (dendrite panel) | `python paper/run_simulations.py --only gallery2d` |
| `fig8_large/dendrite_3d_512` | 4 | 22.55 GB | Fig. 10 | `python paper/run_simulations.py --only gallery3d` |
| `fig8_large/multiphase_3d` | 2 | 1.28 GB | Fig. 10 | `python paper/run_simulations.py --only gallery3d` |
| `fig8_large/recrystallisation_3d` | 2 | 1.28 GB | Fig. 10 | `python paper/run_simulations.py --only gallery3d` |
| `manuscript_large_domain/anisotropic_2d_connected` | 2 | 92 MB | Fig. 9 | `python paper/run_simulations.py --only gallery2d` |
| `manuscript_large_domain/isotropic_2d_connected` | 2 | 92 MB | Fig. 9 | `python paper/run_simulations.py --only gallery2d` |
| `manuscript_large_domain/multiphase_2d` | 2 | 90 MB | Fig. 9 | `python paper/run_simulations.py --only gallery2d` |
| `manuscript_large_domain/recrystallisation_2d` | 2 | 91 MB | Fig. 9 | `python paper/run_simulations.py --only gallery2d` |
| `manuscript_smallgrain_connected/coarse_anisotropic_3d` | 1 | 274 MB | Fig. 10 | `python paper/run_simulations.py --only gallery3d` |
| `manuscript_smallgrain_connected/coarse_isotropic_3d` | 1 | 274 MB | Fig. 10 | `python paper/run_simulations.py --only gallery3d` |
| `manuscript_smallgrain_connected/fine_anisotropic_3d` | 1 | 650 MB | Fig. 10 | `python paper/run_simulations.py --only gallery3d` |
| `manuscript_smallgrain_connected/fine_isotropic_3d` | 1 | 651 MB | Fig. 10 | `python paper/run_simulations.py --only gallery3d` |
| `validation_grain_growth_dense_2d` | 25 | 509 MB | Suppl. Fig. S4 | `python paper/run_simulations.py --only grain_rex` |
| `validation_rex_current_3d` | 16 | 541 MB | Suppl. Fig. S5 | `python paper/run_simulations.py --only grain_rex` |

## Checksums

### `alcu_validation/precipitation/three_dimensional_mesh_fine`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_000.npz` | 13,558,623 | `4f24aab96bcd409c5b08637ef6be59d842fa8ee45fe606273f6ad61c996ffc34` |
| `snap_001.npz` | 18,420,340 | `e177f38d33575e726686f4e121f26483e28999fcf0e59858ab00bf6349094d82` |
| `snap_002.npz` | 21,850,091 | `24741d6632d4aac01df386ce6b820532490fccd853d96c6e0c638d488b0e9482` |
| `snap_003.npz` | 24,649,831 | `641e9061a33558472030c9a3346a2fd21972c0ce7c1b089e533cd3a128010a01` |

### `alcu_validation/precipitation/three_dimensional_mesh_ultrafine`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_000.npz` | 32,078,960 | `7fb83809fa6ea3bee7f260c18b2a68b785459be4dbca331a998115932e813d6a` |
| `snap_001.npz` | 42,942,769 | `4af1356b39cb1d6eaa055529089fdf7d82841fed10b8361cf3fd89c3ee7c26f7` |
| `snap_002.npz` | 51,062,091 | `17082ff369a588858c03c8c5c999c65aacbabff880b4e7c7494eef2c20d7308c` |
| `snap_003.npz` | 57,910,159 | `d49788ef661a6bec9c356ad804ae9a12e304cfad118da5e65a3900d1e61b45eb` |

### `alcu_validation/precipitation/three_dimensional_ultrafine`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_003.npz` | 55,538,729 | `9c4d181e0d6141a3a1550d32b49ddc77f261d0e0c0c9e4d3598f023102920d41` |

### `fig7_large/dendrite_2d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 79,691,937 | `9e81174bc07f7711e662fe3b24b02912c8332a9205c2f63d321f802ad0850896` |
| `snap_0016.csv` | 79,691,941 | `98db9babe4025d5421e4589652526d43c1b34f92605e5be0424368d953e22002` |

### `fig8_large/dendrite_3d_512`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 10,200,547,489 | `3f2467679030b743e8ca2cd0502a000cba213b3f678613e6501d516c287dd083` |
| `snap_0000.npz` | 1,073,742,896 | `84c9d71807807b6bf900e0a0e667d0d1a643e96f23cf75a9259415e4de8cb67e` |
| `snap_0012.csv` | 10,200,547,493 | `dc76bd66a6c13abd13c4aabb3613c6d8a2b8acbddbfef25865ccf1b06765b05e` |
| `snap_0012.npz` | 1,073,742,904 | `55d26e1c87e1348e548c56735b4786e656300adce850da92b8bf523401fdb455` |

### `fig8_large/multiphase_3d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 637,534,371 | `1feadb9efd450bb55aee1613c22e9a8107b165b05c1f73e7f2b51c713f3fc7de` |
| `snap_0001.csv` | 637,534,375 | `58469d3d9a8dee4b6dfddabb782304a9783758c5baee5f55f0ebbc2c2279bf3a` |

### `fig8_large/recrystallisation_3d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 637,761,576 | `cb94f39ebc1388bdbed9e9f7f6d9c2fc5bf88e15fcef260534127f915a01fa4d` |
| `snap_0001.csv` | 644,976,355 | `bc78cc9913190d20fe4e46c80322d356c8ec8902094a9e7020f491a3033da933` |

### `manuscript_large_domain/anisotropic_2d_connected`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 45,827,320 | `ccbd97017229da7806fef1ca580f1c5db251681d6513142bc73a2a78e8093f35` |
| `snap_0001.csv` | 45,816,053 | `6917a18ae2c4b6770662d7d2d742ccb181652de842006bc61f73f2cb6ebd3fa0` |

### `manuscript_large_domain/isotropic_2d_connected`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 45,832,465 | `7c3cca9991f065f68a4ccb4bd14545076075c87c2ed91d56ebee426913d786e9` |
| `snap_0001.csv` | 45,840,655 | `229b833918450fa9ecebb9313d039a616f54de19d0844e550e3281c5739870f7` |

### `manuscript_large_domain/multiphase_2d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 44,826,786 | `ba0dde4f7e807ba7c2ad7e6525c5f4057b3922bdeb77e79053924153c8c1bce0` |
| `snap_0001.csv` | 44,826,790 | `2ed4a8532a1368bed45a953b3986341dae44e2bc0abb356be93b310966e538c3` |

### `manuscript_large_domain/recrystallisation_2d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 44,854,616 | `37ed301a4b3b24305810ce9db62742d1b142bab339448c9f47ff5b92ea754d1e` |
| `snap_0001.csv` | 45,664,023 | `b5e24e5b8053ae5966d3a011fa44d9f9f51d3c5959d52229c400932e1052c70e` |

### `manuscript_smallgrain_connected/coarse_anisotropic_3d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 274,394,825 | `6c083179c94e59fa170c61e82e99d197411046cf581f9819a0d445ba505838c3` |

### `manuscript_smallgrain_connected/coarse_isotropic_3d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 274,456,842 | `857369a98ceaa9849d4020afa98a8416d303778f5d956f412f23710be6966e4e` |

### `manuscript_smallgrain_connected/fine_anisotropic_3d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0001.csv` | 650,317,935 | `3b934a2c0391fba6fe22b8d2293edc60f80e1d8bfb5c3ff3cdefca2c5fd0b8be` |

### `manuscript_smallgrain_connected/fine_isotropic_3d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0001.csv` | 650,757,067 | `af62d05d78cd46eeca44636a4dd5df8f04224190241f27017971d6e957f44fd8` |

### `validation_grain_growth_dense_2d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 20,356,071 | `8025b5d1a8f1b453aa9ef999db7ef33b443ac72c2ea9202dfe466d4dbfe25539` |
| `snap_0001.csv` | 20,351,832 | `841994017b64038d249c27816764629e9526cfccf34dc928ca241decaa9c544e` |
| `snap_0002.csv` | 20,349,408 | `d4e1b6f65d81e0a1f7d62bfecc0f6c5c1cdb5e43cc7d6c27cf34c8914a422976` |
| `snap_0003.csv` | 20,348,587 | `60a86ffa0ab76dc4045f0636ef2a69ea25b4ef33293bd9bf672a459928413ff4` |
| `snap_0004.csv` | 20,347,518 | `3b9f71948188bb72fb010f85e19639f430e15c2012952c384667fabca43f0f6f` |
| `snap_0005.csv` | 20,345,706 | `2c79774091ae35dc66b8746568b255f504c6d8268ad7abea1fe3ef45ad418b17` |
| `snap_0006.csv` | 20,345,193 | `6013a03d026e918e71f19c984571648d915159d53bac4d00ce9c8c337d3022b8` |
| `snap_0007.csv` | 20,344,217 | `23de8c10fc7bbaffb4403447d9e16e78f719f6b16214f82e6258fff525dcc327` |
| `snap_0008.csv` | 20,343,947 | `9eb13957f5d2763c641735732e9a42b37e0b4a632aab0c2f9dfbfdfe64bd9882` |
| `snap_0009.csv` | 20,343,984 | `68523e68840d3297706b0511b23551c8eb12be8e6382fa9fee6ed01febea2e70` |
| `snap_0010.csv` | 20,344,187 | `12df05f5534b574fe7ce1a6f4d01db10e31aec655d4367993a077d4200d0ada9` |
| `snap_0011.csv` | 20,345,389 | `132e7d4a153b864f9f43bf8d8c16ce88ee77b846546a64f5ce753246476ac49b` |
| `snap_0012.csv` | 20,346,781 | `d5db9c7e8612be30f65bee9834ea3fbcb8af5091e970a9da77a0ea892d6a571f` |
| `snap_0013.csv` | 20,347,043 | `3e648d1d4e6b17cb7685e9d92637610880d2fc62545ecb0a497cf0d7de7a78ad` |
| `snap_0014.csv` | 20,347,090 | `123211932520f6c25d31af5c0304054b48a9dc5858b8d971481c21d9b26aa616` |
| `snap_0015.csv` | 20,346,887 | `b40b897b3a4bebac2f7b12801a459e166d991e924b5b88bf705c0c3661edc016` |
| `snap_0016.csv` | 20,347,042 | `de14445e85de62321f003bb90f4c5bc8cc033f13db70f35d94023bbbd3994d31` |
| `snap_0017.csv` | 20,347,179 | `4919c1265e89b14a458683bf712dc6fb3a139f01014e4dd7c6b69311eb7a66ff` |
| `snap_0018.csv` | 20,347,209 | `74d36cd003d18aa13b3e2f6c8a56991009bd7621d58d0b8e59ae83de8e374518` |
| `snap_0019.csv` | 20,347,256 | `223e22c59017984381613bb385b34967e5711bf3dca19549af3ea7409e689617` |
| `snap_0020.csv` | 20,346,847 | `d57c93b6b7997e825c2e1a85823f6520ae20877cd646efe26d1f94340ff5fa94` |
| `snap_0021.csv` | 20,345,867 | `0906d30ffedf131e75393db27c0148deac2889432801c0b66f7b6f51ede3e346` |
| `snap_0022.csv` | 20,344,723 | `0c65456e9a45b59fced0b5abb382d2701b52a2f4626fe789d46aad0e1ab07432` |
| `snap_0023.csv` | 20,343,714 | `e2fc3df36252ebc3bee02f4f87998da715d392a9adf76371375ae43efba57688` |
| `snap_0024.csv` | 20,342,437 | `15caac16a4bd89372fc35c43a25539dd88dc96453bc2ade4f9ab026930ee5da4` |

### `validation_rex_current_3d`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `snap_0000.csv` | 33,623,933 | `1a12a1ba1661b962ed1a84610ce50a7f1b1325f70d8ea8232398902d5cc6a6c1` |
| `snap_0001.csv` | 33,628,045 | `c84b3eea5b4a529dbb1576ec47f6c49bd3d10b334ccc576090c312e9915ce52f` |
| `snap_0002.csv` | 33,641,002 | `a843332057e3aca3d30c635967d1e6bcd625b404d29ab48fb378726181e66779` |
| `snap_0003.csv` | 33,661,150 | `e636e2d5a7698ade380473deda0ead3cb01c9fb5b863b33a6ef67b7fdf592baa` |
| `snap_0004.csv` | 33,675,286 | `0a75a0cd08b2e8d05da7a272e0d5483a8c05aae0de2268c0e97cf6653aedd095` |
| `snap_0005.csv` | 33,705,251 | `63035edb00397967b945d1a9f21adc5bf1867ac02c02895ba48b075abe75bc93` |
| `snap_0006.csv` | 33,741,266 | `20282f35408ca2cf376591e0538081363bd6b6b2a9a9a03363b9136daa4d08ba` |
| `snap_0007.csv` | 33,779,885 | `b0ec9b41712bd5a9b47908350acb975c80f1417a6785be8276dbc4181b5a14fe` |
| `snap_0008.csv` | 33,809,335 | `766d6388a8d3940424875909402c22bb94b8da8c31c591d79df4b91fb76984ad` |
| `snap_0009.csv` | 33,845,845 | `b19c556db4d38b532c31ec0fd345d9e4f6cf4ba222aad6df5f4bc5bbbe8c6881` |
| `snap_0010.csv` | 33,875,312 | `43b569f08a17137a4401aa0fd005d4a6f08bdfdc8c0ae3f9d94af0719285fb1e` |
| `snap_0011.csv` | 33,901,972 | `392e4716cd10cdf94f593c87a596ed6151e84cbd4fece234b77296109796da92` |
| `snap_0012.csv` | 33,922,077 | `3d84f79e6f884c8541d7829b846538951189df6a0303ce3296a02d56708c0be5` |
| `snap_0013.csv` | 33,936,394 | `de9104d74381230703d5dd0da248b3ab9447b77fe462cb2ede5ae0a4f4eafc37` |
| `snap_0014.csv` | 33,944,312 | `29ca6d7a8ab57e248d0a102c29c39deef14e5b4fc2066111bdfec896d61e3b28` |
| `snap_0015.csv` | 33,948,607 | `5c5db0f35381e62bcdc5c508e07a5b7042f046a05eb87efbe686360942520c62` |
