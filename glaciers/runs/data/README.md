# Initial conditions and data sources

The three NetCDF files contain the initial conditions used in the paper. `manifest.json` records their SHA-256 checksums, original experiment files, source repositories and processing.

| File | Glacier | Source and preparation |
|---|---|---|
| `aletsch.nc` | Aletsch | Model state at year 200, generated from the historical [IGM Aletsch tutorial](https://github.com/instructed-glacier-model/igm-examples/tree/43be36b2c687d27fe94a6eccb21a661c89cb9765/aletsch-basic). The pre-spin-up input is identical to the tutorial file. |
| `valais.nc` | Valais | Model state at year 200, using a crop of the 100 m SRTM-based [Alps bed](https://github.com/instructed-glacier-model/igm-core-paper-exp/tree/0f0274acb922436a92be9cd51cb8d600a43e0ebc/alps_scalability/runs), with present-day glaciers and major lakes removed. The bed uses the thickness estimates of Farinotti et al. (2019). |
| `jakobshavn.nc` | Sermeq Kujalleq | BedMachine Greenland-derived geometry at 150 m resolution (EPSG:3413), preserved from the original Jakobshavn experiment. |

The Alpine files are obtained by sampling the original DahuNet reference trajectories at year 200 and retaining the original ice mask where present. The reference trajectories used for the accuracy comparisons are generated separately by this repository.

The Sermeq file is identified by the original repository commit and checksum in the manifest. Its coordinates and surface elevations match the BedMachine Greenland grid and surface field. The exact BedMachine release and bed/thickness processing are unspecified in the archived source.

## References

- **Aletsch tutorial:** [IGM examples](https://github.com/instructed-glacier-model/igm-examples), commit `43be36b2c687d27fe94a6eccb21a661c89cb9765`, `aletsch-basic/data/input.nc`.
- **SRTM:** Farr et al. (2007), *The Shuttle Radar Topography Mission*. [doi:10.1029/2005RG000183](https://doi.org/10.1029/2005RG000183).
- **Glacier thickness:** Farinotti et al. (2019), *A consensus estimate for the ice thickness distribution of all glaciers on Earth*. [Article](https://doi.org/10.1038/s41561-019-0300-3); [dataset](https://doi.org/10.3929/ethz-b-000315707).
- **Alps bed preparation:** Jouvet et al. (2023), *Coupled climate–glacier modelling of the last glaciation in the Alps*. [doi:10.1017/jog.2023.74](https://doi.org/10.1017/jog.2023.74); Leger et al. (2025), *A data-consistent model of the last glaciation in the Alps achieved with physics-driven AI*. [doi:10.1038/s41467-025-56168-3](https://doi.org/10.1038/s41467-025-56168-3).
- **BedMachine Greenland:** [IceBridge BedMachine Greenland, IDBMG4](https://nsidc.org/data/idbmg4); Morlighem et al. (2017), *BedMachine v3: Complete bed topography and ocean bathymetry mapping of Greenland from multibeam echo sounding combined with mass conservation*. [doi:10.1002/2017GL074954](https://doi.org/10.1002/2017GL074954).

## Licensing

The repository's GPL-3.0-or-later licence applies to code. Source datasets retain their own terms. The Farinotti et al. thickness dataset is distributed under [CC BY-NC-SA 4.0](https://doi.org/10.3929/ethz-b-000315707), as specified by ETH Zurich's Research Collection.
