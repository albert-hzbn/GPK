# Reproducing the manuscript figures

This folder holds everything needed to regenerate the figures of

> A. Linda and D. Raabe, *A general sharp-interface simulation framework for microstructure evolution*, submitted to npj Computational Materials.

| Path | Contents |
|---|---|
| `inputs/` | Input decks (`.ini`) for every simulation shown in the manuscript |
| `data/` | Processed results read by the figure scripts (metrics, time series, summary tables, small snapshots) |
| `LARGE_DATA.md` | The large 3-D snapshot fields that are not in the repository: sizes, checksums and how to regenerate them |
| `calphad/` | COST-507 Al–Cu database (`COST507-modified.tdb`) and the CALPHAD tables derived from it |
| `figures/pyvista_panels/` | PyVista panel renders that the matplotlib figures embed |
| `run_simulations.py` | Re-runs the simulations, in the right order, into `output/` |
| `gpk_data.py` | Finds each result folder in `$GPK_DATA`, `output/`, `build/output/` or `paper/data/` (in that order) |
| `*.py` | Figure scripts (listed below) |

## Quick route: figures from the bundled data

From the repository root:

```bash
python -m pip install -r paper/requirements.txt
cd paper
python generate_publication_figures.py
python make_timestep_detail_figures.py
python alcu_validation_figures.py
python alcu_microstructure_figures.py
```

The figures are written to `paper/figures/` (the Al–Cu figures to `paper/`). Figures whose input is one of the large fields in [`LARGE_DATA.md`](LARGE_DATA.md) are skipped with a message; regenerate those fields first (next section).

The scripts named `render_*.py` draw the microstructure panels with ParaView's VTK. Run them with ParaView 5.13 `pvpython`, after installing PyVista into a folder that `pvpython` can see:

```bash
python -m pip install --target .pyvista_pv "pyvista>=0.47,<0.49"
pvpython paper/render_si_panels.py
pvpython paper/render_main_panels.py
pvpython paper/render_dendrite_2d_panels.py
GPK_DENDRITE_INDEX=-1 pvpython paper/render_3d_microstructures.py
```

`render_3d_microstructures.py` writes its figure to the system temporary folder (`$TEMP/gpk_3d_render/`); copy `fig_qualitative_3d_curvature_longtime.pdf` from there to `paper/figures/`.

## Full route: regenerate the simulation data

1. Build the code and run the test suite (see the main [README](../README.md)). The tests write the verification data to `build/output/`.
2. Run the production simulations:

   ```bash
   python paper/run_simulations.py --list            # show the groups and their steps
   python paper/run_simulations.py --only gallery2d  # or: gallery3d, grain_rex, alcu
   ```

   The `gallery3d` group is by far the most expensive. On the 12-thread workstation used for the manuscript, the 512³ dendrite took 13.3 h and wrote 131 GB of snapshots, and the 3-D recrystallisation run took 7.9 h. Reduce `n_snaps` in a deck to save disk space.
3. Rerun the figure scripts. They now pick up your results from `output/` and `build/output/` before the bundled data.

## Figure map

Main text:

| Figure | File | Script (function) | Data | Produced by |
|---|---|---|---|---|
| 1 | `fig1_gpk_overview.pdf` | `generate_publication_figures.py` (`overview`) | none (schematic) | — |
| 2 | `fig2_timestep_operations.pdf` | `generate_publication_figures.py` (`timestep_schematic`) | none (schematic) | — |
| 3 | `fig_velocity_subcell.pdf` | `make_timestep_detail_figures.py` | none (schematic) | — |
| 4 | `fig_tckt_bgr.pdf` | `make_timestep_detail_figures.py` | none (schematic) | — |
| 5 | `figS2_geometry_pipeline.pdf` | `generate_publication_figures.py` (`geometry_pipeline`) | computed in the script | — |
| 6 | `figS3_curvature_convention.pdf` | `generate_publication_figures.py` (`curvature_signs`) | none (schematic) | — |
| 7 | `figS4_gaussian_accuracy_final.pdf` | `generate_publication_figures.py` (`gaussian_accuracy`) | computed in the script | — |
| 8 | `fig3_foundational_validation.pdf` | `generate_publication_figures.py` (`foundational_validation`) | `validate_theory_foundations/`, `validate_extended_base_models/` | `ctest` |
| 9 | `fig_qualitative_2d_gallery.pdf` | `generate_publication_figures.py` (`qualitative_2d_gallery`) + `render_si_panels.py`, `render_main_panels.py`, `render_dendrite_2d_panels.py` | `manuscript_large_domain/`, `fig7_large/` | `run_simulations.py --only gallery2d` |
| 10 | `fig_qualitative_3d_curvature_longtime.pdf` | `render_3d_microstructures.py` | `manuscript_smallgrain_connected/`, `fig8_large/` | `run_simulations.py --only gallery3d` |
| 11 | `fig6_benchmark.pdf` | `generate_publication_figures.py` (`benchmark`) | `gpk_pf_comparison/` | `ctest` (`gpk_phase_field`) |
| 12 | `fig10_gpk_phase_field_comparison.pdf` | `generate_publication_figures.py` (`gpk_phase_field_comparison`) | `gpk_pf_comparison/` | `ctest` (`gpk_phase_field`) |
| 13 | `fig11_al4cu_material_validation.pdf` | `alcu_microstructure_figures.py` (`main_figure`) | `alcu_validation/realistic_population_3d/` | `run_simulations.py --only alcu` |

Supporting Information:

| Figure | File | Script (function) | Data | Produced by |
|---|---|---|---|---|
| S1 | `figS1_profile.pdf` | `generate_publication_figures.py` (`profile`) | computed in the script | — |
| S2 | `figS13_topology_transactions.pdf` | `generate_publication_figures.py` (`topology_transactions`) | none (schematic) | — |
| S3 | `figS5_stability.pdf` | `generate_publication_figures.py` (`stability`) | computed in the script | — |
| S4 | `figS6_grain_growth.pdf` | `generate_publication_figures.py` (`grain_growth`) | `validation_grain_growth_dense_2d/` | `run_simulations.py --only grain_rex` |
| S5 | `figS7_recrystallisation.pdf` | `generate_publication_figures.py` (`recrystallisation`) | `validation_rex_current_2d/`, `validation_rex_current_3d/` | `run_simulations.py --only grain_rex` |
| S6 | `figS9_bicrystal_validation.pdf` | `generate_publication_figures.py` (`bicrystal_validation`) | `validate_extended_base_models/` | `ctest` |
| S7 | `figS14_curved_bicrystal_validation.pdf` | `generate_publication_figures.py` (`curved_bicrystal_validation`) | `validate_extended_base_models/` | `ctest` |
| S8 | `figS10_al4cu_precipitate_targets.pdf` | `alcu_validation_figures.py` | `calphad/`, literature values in the script | `scripts/generate_alcu_calphad.py` |
| S9 | `figS11_alcu_fields_and_audits.pdf` | `alcu_microstructure_figures.py` (`si_figure`) | `alcu_validation/precipitation/` | `run_simulations.py --only alcu` |
| S10 | `figS12_alcu_mesh_convergence.pdf` | `alcu_microstructure_figures.py` (`mesh_convergence_figure`) | `alcu_validation/precipitation/` | `run_simulations.py --only alcu` |

"Data" folders are relative to the data roots searched by `gpk_data.py`. "computed in the script" means the figure evaluates analytical expressions or small calculations itself and reads no simulation output.
