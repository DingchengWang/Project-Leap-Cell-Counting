# Cell Counting user guide

[中文](guide.zh.md) · [Methods](methods.en.md) · [Repository home](../README.md) · [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/)

This guide describes Cell Counting **1.0.0** in **Project Leap v1.0.0**. Download the main application from the [Project Leap download page](https://dingchengwang.github.io/Project-Leap/en/downloads/). The workflow below starts inside the Project Leap application.

## Install and select an image

1. Open the Cell Counting module in Project Leap and use its install action if it is not installed. The application installs the module and its shared scientific runtime and fixed model.
2. Click **Choose image…**, or drag one `.tif` or `.tiff` file into the input area. Use an 8-bit or 16-bit multichannel fluorescence image containing DAPI and a separate support marker.
3. If the file contains multiple image series or time points, choose the intended **Image series** and **Time point**. These choices select the data for this run.
4. Check the selected **DAPI counting channel** against the image. The application tries to identify it automatically; choose the correct source channel if it is missing or wrong.
5. Select a different **Support channel**, then choose its **Support marker**: eGFP, Biocytin, ALDH1L1, or a custom name. Both selections are required. The marker records which signal you are using to assess bodies and nuclear surroundings. Current real-data evaluation is limited to Biocytin; see [Methods](methods.en.md#evidence-scope) for the evaluation scope.
6. Under **Display channels and colors**, select at least one source channel for review. Keep the channels you will need for comparing DAPI and support signal. In each channel's color text field, enter a `#RRGGBB` value or leave it empty to use the default color. You can toggle these selected channels during review.

The application automatically checks the image and calibration before running. Multiplane images need valid X/Y/Z values in micrometres and square XY sampling; true single-plane images need valid X/Y values. If the application reports missing or incompatible calibration, use a TIFF exported with the correct source metadata.

## Run and inspect the result

Click **Start automatic counting**. The application analyzes the selected image and opens the counting editor when it is ready. At this point the objects are available for review; the result has not yet been confirmed.

Use **MIP** to view the maximum-intensity projection across the stack, then move the Z slider to inspect the original planes. The interface displays Z planes starting at 1. On a single plane, a marker appears only when that object has a recorded position on the plane; MIP shows all non-deleted objects. The top-left **Total Astrocyte: N** is the number of accepted cells across the entire session, not a separate count of the current plane.

Work through the Z planes while comparing the selected channels. Check whether a proposed object corresponds to a cell, whether two records belong to the same cell, and whether an unmarked cell should be added. A single cell can appear on several planes with the same count number. Pinch to zoom, scroll or drag to pan, and press **0** to fit the image to the window.

## Interpret the four labels

| Label | Appearance | Meaning in the editor | Included in total? |
|---|---|---|---|
| Astrocyte | Green numbered circle | Accepted cell record | Yes |
| High Likelihood | Amber diamond | Nuclear support passes the acceptance rule, with the algorithm's XY search-region boundary as the only remaining limitation | Only after acceptance |
| Unrecognized | Yellow point | Other review object | Only after acceptance |
| Neurons | Purple square | Excluded object | No, unless accepted manually |

These labels describe the editor's counting states. Neurons is the excluded category; its biological identity needs to be assessed from the staining and image evidence. High Likelihood remains under review until you accept it using the actions below.

## Adjust the count

| Task | Action and result |
|---|---|
| Accept one High Likelihood object | Click it normally. The existing object immediately joins the count. |
| Select without accepting | Hold **Command** while clicking. This selects or deselects objects without changing their state; repeat to select several. |
| Accept Unrecognized or Neurons | Click to select the object, then click **Add to count**. |
| Accept several objects | Command-click the desired objects, then click **Add to count**. This accepts the selected review or excluded objects. |
| Add a missing cell | Press **A** or click **Add (A)**, then click an empty image position. The new manual point immediately counts. Press **V** or **Select (V)** to return to selection. |
| Merge records for one cell | Command-select at least two objects, then press **M** or click **Merge selected (M)**. The selection cannot contain Neurons. If every parent was accepted, the merged record counts; if any parent was under review, select the merged record and use **Add to count** when appropriate. |
| Delete an incorrect record | Select it, then use **Delete selected**, **Delete**, or **Backspace**. The object is hidden and removed from the count; deletion does not relabel it as Neurons. |
| Reverse an edit | Use **Undo** or **Redo**. The editor retains up to 100 edits in this history. |

For a manual point tied to a particular Z plane, add it while viewing that plane. A point added on MIP has no assigned Z and does not automatically appear in individual-plane views. Clicking an existing marker in Add mode activates that existing object instead of creating a second point there.

## Confirm and reopen

The editor saves a draft as you edit. To create the result files, click **Confirm & save Astrocyte count**. You can leave objects in High Likelihood or Unrecognized; only accepted, non-deleted objects enter the confirmed count. After a successful save, the editor closes and the input area is ready for a new image. If saving fails, the current review session remains open.

To continue later, open **Jobs & Results**, expand the task, and click **Reopen count** for a confirmed result or **Continue count review** for unfinished review. The editor restores the saved result and an existing draft when present. Use **Open results folder** to access a completed task's files. Confirm the current review before switching to another counting task.

Keep the original TIFF available for original-Z viewing and rerunning analysis; original-Z viewing also requires the installed counting runtime. If the file's location changes, use **Locate original image…** in the editor; the application checks the replacement before using it. Saved MIP images and count editing remain available when the original TIFF is unavailable. The session JSON refers to additional files in the task directory, so keep the task data together when retaining an editable result.

## Result files

Each confirmation creates a new revision under the task's counting directory:

```text
counting/results/
  revision-001/
    Cell_Count_Overlay.png
    Cell_Counts.csv
    Cell_Count_Session.json
  latest.json
```

| File | Contents |
|---|---|
| `Cell_Count_Overlay.png` | Full-resolution MIP with the current display-channel composition and the accepted cells numbered. It is not a screenshot of the current Z plane. |
| `Cell_Counts.csv` | One row per accepted, non-deleted cell. A zero-cell result contains only the column header. |
| `Cell_Count_Session.json` | All objects, review states, deletion and merge records, display choices, source references, and observation evidence needed to reopen with the associated task data. |
| `latest.json` | Reference to the most recently confirmed revision. |

The engine also retains analysis, label arrays, observations, and diagnostics in the task's `counting` directory. The [methods page](methods.en.md) explains their scientific roles.

### CSV columns

| Columns | Meaning |
|---|---|
| `run_id`, `revision` | Task identity and confirmed revision number. |
| `count` | Total accepted cells in this revision, repeated in each row; do not sum this column. |
| `display_no` | Consecutive number shared by the editor and PNG for this count. It is regenerated from representative Y/X ordering and is not a permanent identity. |
| `id`, `cell_id` | Stable cell-record identity; a manual merge creates a new identity. |
| `x`, `y`, `z` | Representative original-image coordinates. X/Y are in pixels; Z is a zero-based plane index. Blank Z can occur for MIP-added points or merges with different parent Z values. |
| `source` | `auto`: nuclear route; `support_body`: independent body; `dapi_support`: associated nucleus and body; `manual`: manually added point; `merged`: manual merge. |
| `status` | Saved annotation state; rows in this export are `accepted`. |
| `label_artifact_id`, `label_id` | Representative mask's artifact and local label. Interpret them together; manual records may have neither. |
| `plane_anchors`, `plane_anchor_sources` | Recorded per-Z positions and the observation or manual point that supplied each position. |
| `source_observations` | Nuclear/body observations, their mask references, original bounds, and measured evidence. |
| `support`, `association` | Nuclear support measurements and nucleus–body association records, where present. |
| `parent_ids`, `lineage`, `reference_z`, `anchor_parent_id` | Merge provenance: immediate parents, ancestry IDs, retained parent Z references, and the parent supplying the representative XY anchor. |
| `label_ids` | Legacy merged-label references; modern multi-observation records retain mask identities in `source_observations`. |
| `point_metadata` | Complete saved point record as JSON. |

Structured columns contain JSON within the CSV cell. For example, UI **Z plane 6** corresponds to source index `5`; a cell with anchors on planes `5`, `6`, and `7` still occupies one CSV row and contributes one to `count`.
