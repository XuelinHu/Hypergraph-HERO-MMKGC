# Dataset Openness Notes

This file records the dataset status checked for the manuscript datasets. No dataset files are
downloaded by this repository.

## Summary

| Dataset | Paper Description | Public Availability | Action |
| --- | --- | --- | --- |
| DB15K | DBpedia-based MMKG benchmark with entity images/text | Public benchmark reused by MMRNS and later MMKGC repositories | Do not download automatically; provide user-side path |
| MKG-W | Wikidata-based MMKG benchmark | Public benchmark reused by MMRNS, MoMoK, C2RS, DHNS, and other MMKGC repositories | Do not download automatically; provide user-side path |
| MKG-Y | YAGO-based MMKG benchmark | Public benchmark reused by MMRNS, MoMoK, C2RS, DHNS, and other MMKGC repositories | Do not download automatically; provide user-side path |

## Confirmed Sources

- The manuscript states that DB15K, MKG-W, and MKG-Y are widely used MMKGC benchmark
  datasets and uses their official train/validation/test splits.
- MMRNS public repository: <https://github.com/quqxui/MMRNS>. Its README says the
  embeddings and raw data can be downloaded from Google Drive.
- C2RS public repository: <https://github.com/ADMIS-TONGJI/C2RS>. Its README points
  DB15K, MKG-W, and MKG-Y users to MMRNS and provides additional token downloads.
- DHNS public repository: <https://github.com/ngl567/DHNS>. Its README says it reuses
  DB15K, MKG-W, and MKG-Y along with MMRNS.
- AGE-CMSF public repository: <https://github.com/ddaipaibulaotie/AGE-CMSF>. Its README
  points DB15K, MKG-W, and MKG-Y downloads to MMRNS.
- DB15K is derived from DBpedia; MKG-W is derived from Wikidata; MKG-Y is derived from YAGO.

## License/Redistribution Caution

The benchmark data are publicly reachable and commonly reused in MMKGC research. However,
the checked repositories do not provide a clear dataset redistribution license in the visible
README text. Therefore, this project should not automatically download, mirror, or commit the
dataset files. Users should download them from the original public release and comply with
the source dataset terms.

## Practical Recommendation

For reproducibility, cite the original dataset/method repository and ask users to place the
downloaded data under:

```text
data/DB15K/
data/MKG-W/
data/MKG-Y/
```

The repository should not vendor these datasets unless the exact license and redistribution
terms of the downloaded package are checked and approved.
 