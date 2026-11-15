# Cell Counting methods

[中文](methods.zh.md) · [User guide](guide.en.md) · [Repository home](../README.md)

Module **1.0.0** counts accepted cell records assembled from two kinds of observation: DAPI nuclei and independently detected support-channel bodies. A record can contain a nucleus alone, a body alone, or a nucleus–body pair whose original-coordinate evidence establishes a unique association. Each record contributes one to the total when its final state is accepted and it has not been deleted.

Here, an observation is a record of a detected nucleus or body, including its mask, position, and measured evidence. A cell record links observations assigned to the same cell.

## Inputs and coordinate system

The application supplies one image series and one time point from an unsigned 8-bit or 16-bit multichannel TIFF. The request identifies a DAPI channel, a different support channel, its declared marker, and the channels and colors used for display. Multiplane data require actual X, Y, and Z physical calibration; a genuine single-plane input requires X and Y calibration and is processed with its available planar evidence. The desktop input check also requires square XY sampling for multiplane images.

All scientific masks use the original image grid. Coordinates, physical distances, and source-Z intersections remain tied to that grid. The support marker name records the user's channel declaration; eGFP, Biocytin, ALDH1L1, and custom names use the same analysis rules. Display colors and channel visibility affect the review image only.

## Find local work across the stack

The engine scans every source Z plane of the required channels. It computes exact intensity summaries on 8 × 8 and 16 × 16 XY blocks, including edge-block sample counts, and builds full-stack maximum-intensity projections for selected display channels. These summaries locate spatial concentrations and schedule work on the original pixels; their regions are not counted as cells or used as segmentation input.

DAPI localization uses the 16-pixel summaries to retain above-threshold concentration regions across Z. Large regions are partitioned into bounded computation windows while retaining all qualified foreground. Each event supplies a local Z range and an XY location. That Z range starts the investigation; original-volume recovery determines the observed object's axial extent.

## Obtain local DAPI model evidence

For each retained event, the engine takes the maximum of the original DAPI values over that event's local Z range. A real-source XY window supplies 256 × 256 pixels to the fixed model; only an image axis that is itself smaller than 256 pixels receives computational padding, which is removed before mask generation. The feature is normalized once using its measured 1st and 99th percentiles.

The fixed configuration is:

| Item | Current value and role |
|---|---|
| Cellpose package | `4.2.1.1` |
| Model | `cpsam_v2` |
| Model SHA-256 | `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667` |
| Model mode | Two-dimensional inference; `do_3D=False` |
| Primary mask thresholds | `cellprob_threshold=0.0`, `flow_threshold=0.4` |
| Weak-candidate threshold | `cellprob_threshold=-2.0`, used only when the primary masks do not cover a localization seed; new candidates require review |
| Spatial sampling | Original XY and Z sampling; no physical-size rescaling |

Primary and weak candidates use the same predicted flow and score fields. If an observed mask meets an internal model-window boundary, the engine performs a recorded check with up to 384 × 384 pixels of actual source context. It preserves the original observation and records whether the extra context resolves the boundary. The full-stack display projection does not enter this inference path.

## Recover nuclei in the original volume

The two-dimensional model masks constrain lateral identity. The engine then reads original DAPI voxels around each event and measures local signal concentration in three dimensions. It defines signal events relative to each plane's median and smooths their occupancy using a physically calibrated Gaussian bandwidth of 0.568 µm. This bandwidth sets the measurement scale; it is not an assumed nuclear diameter.

Connected signal-density regions form candidate three-dimensional regions. A region associated with one model label remains one event despite internal intensity peaks. When several model labels share a region, their measured density maxima seed a watershed separation. Disconnected regions retain distinct event identities. Regions without matching model evidence are recorded as locations for follow-up and do not produce nucleus objects.

Recovery expands its source-image window when observed density or axial-endpoint evidence requires more context. It records unresolved identity and boundary evidence for review. Overlapping nuclear observations from different windows are reconciled using their actual masks; a reciprocal overlap of at least 0.5 within the shared native region can identify a repeated observation, while competing overlap remains ambiguous. Each final nuclear observation stores a representative point and an interior anchor on every source Z plane that its mask actually intersects.

## Measure support around nuclei and discover bodies

The selected support channel has two distinct roles.

First, the engine measures fluorescence adjacent to each recovered nucleus. Physical-area-weighted outward samples record contact with the nuclear surface, consecutive nearby positive signal, and support across the observed Z planes. The current rule uses a 0.6 µm contact allowance and 1.2 µm sustained signal depth; measured contact, thick coverage, valid sampling, and Z support determine the observation's accepted, review, or excluded state. A distant disconnected signal cannot complete a consecutive run. The full decision parameters are recorded in [`support.py`](../engine/leap_cell_counting/support.py).

Second, an independent route locates concentrations in the support channel and examines their original pixels. It measures local background and signal amplitude, constructs thickness geometry, merges adjacent thickness basins where appropriate, and identifies spatially resolved compact cores. A body observation requires measured support-positive pixels associated with such a core. A core reaching a computation-window boundary requests more real-source context. The measured fluorescence mask and the geometric envelope used for association are saved separately; closing a cavity in that geometry does not create positive fluorescence voxels or a nuclear mask.

These routes use the same source-statistics pass and can run concurrently. Body detection does not use a DAPI detection or Cellpose mask as an entry requirement. The engine retains every qualified work region; a capacity error stops the run instead of exporting an incomplete count.

## Resolve cell identity globally

The engine measures each potentially associated nucleus–body pair in original coordinates. A geometric match requires all of the following:

1. The nuclear representative lies inside the body's association envelope.
2. Every voxel of the nucleus's maximal interior-distance core lies inside that envelope. This core is the set of deepest interior voxels of the observed nuclear mask, using physical distances.
3. Actual positive body pixels occur on the representative's original Z plane.

The resolver then checks the whole set of candidate relations. It joins a pair only when the correspondence is unique on both sides and has no unresolved competing relation. Partial enclosure or competition leaves separate reviewable records. If both the representative and the entire core are outside the envelope, the relation is a non-match. Whole-nucleus containment fractions are retained as measurements; centroid proximity and cross-kind mask overlap are not sufficient identity rules.

A uniquely paired nucleus and body become one cell with both source observations. Unmatched nuclei and bodies retain separate cell records. An ambiguous association makes the affected records `review`; otherwise, a record is `accepted` if at least one of its observations is accepted, `review` if none is accepted and at least one needs review, and `excluded` if all are excluded. This lets accepted nuclei and bodies count independently as well as in uniquely paired records.

The cell's per-Z anchor uses a real observation on that layer, preferring body evidence where available and retaining nuclear intersections on other layers. Thus one cell can be displayed on several Z planes while contributing only one to the total.

## Review states and saved evidence

Only non-deleted cells with `status="accepted"` contribute to the count. `review_hint="likely_astrocyte"` separates High Likelihood from the remaining review objects when the nuclear support rule passes and the boundary of the algorithm-defined XY search region is the sole remaining limitation. The hint does not accept a cell. The [user guide](guide.en.md) explains the four displayed labels and the actions that change the count.

`labels.npy` and `body-labels.npy` store separate nuclear and body labels on the source ZYX grid. A label integer is meaningful only together with its artifact ID. Cell IDs identify the logical records; the visible `display_no` is regenerated as consecutive count numbers. Manual acceptance preserves the cell ID and original masks. A manual merge receives a new cell ID and retains the parent observations. Delete and undo edit annotations without rewriting the scientific masks.

`analysis.json` records the input, model, channel selections, calibration, and label registry; `observations.json` retains source observations and measured associations; diagnostics and work ledgers retain parameters and processing evidence. Confirmed sessions preserve all review states and edit history, while the CSV and numbered PNG contain only the counted cells. See the [identity contract](prototype-d/CELL_IDENTITY.md) for exact persistence fields and the [user guide](guide.en.md) for exported columns.

## Evidence scope

The thresholds and geometric rules above describe the implemented counting method. Current real-data evaluation covers Biocytin; performance with the other selectable markers remains to be evaluated. The nucleus–body association rules still require validation against representative, manually annotated real masks. Astrocyte and Neurons identify accepted and excluded workflow categories; the module has not been independently validated as a biological cell-type classifier. A genuine single-plane input provides two-dimensional evidence only.

## Implementation and upstream sources

| Operation | Source |
|---|---|
| Source statistics and work localization | [`source_scan.py`](../engine/leap_cell_counting/source_scan.py), [`proposals.py`](../engine/leap_cell_counting/proposals.py) |
| Model identity and local inference | [`segmentation.py`](../engine/leap_cell_counting/segmentation.py), [`xy_batch.py`](../engine/leap_cell_counting/xy_batch.py) |
| Original-volume recovery and repeated observations | [`native_recovery.py`](../engine/leap_cell_counting/native_recovery.py), [`label_store.py`](../engine/leap_cell_counting/label_store.py) |
| Nuclear support and independent bodies | [`support.py`](../engine/leap_cell_counting/support.py), [`support_discovery.py`](../engine/leap_cell_counting/support_discovery.py), [`body_pipeline.py`](../engine/leap_cell_counting/body_pipeline.py) |
| Cell association and orchestration | [`identity.py`](../engine/leap_cell_counting/identity.py), [`cell_pipeline.py`](../engine/leap_cell_counting/cell_pipeline.py), [`efficient.py`](../engine/leap_cell_counting/efficient.py) |

Upstream model sources: [Cellpose](https://github.com/MouseLand/cellpose) and [official `cpsam_v2` weights](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2). Project Leap's original code uses [Apache-2.0](../LICENSE); third-party code and weights retain their own licenses and notices.
