# Dependency rights inventory

No third-party implementation or third-party data/atlas/font is bundled. No project LICENSE has been selected. The copied project-specific model/evaluation extensions still require author confirmation of authorship and any derivative-code rights before redistribution.

| Dependency | Local evidence reviewed | Candidate disposition |
|---|---|---|
| Hodge toolbox (`hodge-main`) | README credits Vijay D. Anand / Moo K. Chung; Hodge_decompose has copyright notice. No LICENSE found in local checkout. Local README identifies `github.com/laplcebeltrami/hodge`. | Excluded. Obtain version and redistribution permission; do not interpret public visibility as a license. |
| rest-system-id Utilities | tt_decomp and my_whitetest_multivar credit Erfan Nozari and say All rights reserved. No top-level LICENSE found in local tree. | Excluded. Needed for training/generation and legacy whiteness; permission unresolved. |
| Network-SDE-Inference / LaGNA | Local LICENSE is MIT, copyright 2025 Ting-Ting Gao, requiring retention of copyright/permission notice. | Excluded because seven-model benchmark is not bundled. If later included, preserve the exact LICENSE and inspect transitive dependencies. |
| MINDy_rfMRI_stable-main | Local LICENSE is MIT, copyright 2024 Ruiqi Chen, requiring retention of notice. Nested utilities contain additional licenses. | Excluded. Root license is not assumed to cover every nested third-party asset. |
| MATLAB + Statistics and Machine Learning Toolbox | Proprietary runtime installed locally, R2022b/12.4 used for the demo. | Not redistributed. Reviewer needs a licensed installation. |
| Brain surfaces, atlas assets, Helvetica Neue | Not needed by the synthetic demonstration. | Excluded. No licensing inference made. |

The main-figure additions are project-specific source excerpts and author-code adapters, not copies of the external libraries. Their authorship and derivative rights remain pending; absence of a license header is not proof of ownership or redistribution permission. Review/validation of a local copy must not be conflated with authorization to publish it.

## Main-figure additions

Python libraries are referenced by import and recorded versions, not vendored. Install them from their official distributions with their own license/copyright notices intact: NumPy, pandas, SciPy, threadpoolctl, Matplotlib, PyMuPDF, Pillow, nibabel, PyVista/VTK, scikit-image and BrainSpace. This inventory is NOT a legal compatibility determination. In particular, review PyMuPDF's applicable AGPL/commercial terms before public distribution or integration; this package does not supply a replacement license or library binary.

Fig5's copied project adapter and search core call external MINDy functions; no MINDy implementation is redistributed. Obtain the exact author version and retain its MIT notice plus any nested notices if separately distributing a dependency. MATLAB root validation also needs Optimization Toolbox; the small model-metric demo does not.

fsLR/Conte69 surface geometry, Schaefer and Tian atlases, canonical gradient references, locked spin indices, font files, frozen templates and derived brain images are external inputs. Their provenance, permitted use and redistribution rights must be confirmed separately. Figure scripts may fail until these assets and the licensed font are supplied; they do not download them. No unknown-license third-party implementation was quietly added.
