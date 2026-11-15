# Native cell observations and identity contract

The analysis uses `schema_version: 2` and
`identity_model: "multi_observation_cell_v1"`. The pure identity functions are
implemented in `engine/leap_cell_counting/identity.py`; body discovery is
implemented in `engine/leap_cell_counting/support_discovery.py`.

## Evidence and counting

DAPI observations describe nuclei. Independent Support observations describe
native three-dimensional dense body regions. A qualified body observation can
count without a successful DAPI Cellpose detection. Support does not manufacture
a nuclear mask. The same physical cell is counted once after explicit
nucleus/body association; competing associations remain reviewable.

Nucleus/body association uses actual Z evidence, nuclear containment in the
body association envelope, representative/core containment and competing
observations. Cross-kind mask IoU and centroid distance alone are not identity
rules. The approved geometric candidate requires the nuclear representative
and its entire maximal interior-distance core to lie in the body envelope,
with actual positive body pixels on the representative's original Z. Only a
unique correspondence on both sides merges. When both representative and core
are outside the envelope, peripheral contact is recorded as no match; partial
enclosure or competition requires review. Whole-nucleus containment fractions
remain descriptive rather than imposing an arbitrary percentage cutoff.

The selected diffuse Support channel is processed according to its pixels and
calibration. eGFP, Biocytin, ALDH1L1 and custom strings identify the declared
marker; they do not select hidden algorithms or name-based gates. Current real
validation data are Biocytin; other markers are not thereby validated.

## Label artifacts

`analysis.label_artifacts` has two fixed entries:

```json
{
  "dapi_nuclei": {
    "path": "labels.npy", "kind": "nucleus", "axes": "ZYX",
    "origin_zyx": [0, 0, 0], "shape_zyx": [300, 4096, 4096]
  },
  "support_bodies": {
    "path": "body-labels.npy", "kind": "body", "axes": "ZYX",
    "origin_zyx": [0, 0, 0], "shape_zyx": [300, 4096, 4096]
  }
}
```

The shape shown is an example. Each file uses original source-grid integer
labels, with zero as background. A label integer is scoped to its artifact.
Paths are relative to the analysis or session JSON that owns the registry and
must be rebased when that JSON is exported. `analysis.labels_path` remains
`labels.npy` for the nuclear artifact and existing history readers.

## Cell and observation records

`point.id` and `point.cell_id` are the same globally unique, opaque cell ID.
Clients must not parse their prefixes. `point.label_artifact_id` and
`point.label_id` identify a representative observation, not a global mask ID.
Status is `accepted`, `review` or `excluded`; only accepted logical cells count.
The UI displays accepted cells as **Astrocyte**, review cells carrying
`review_hint: "likely_astrocyte"` as **High Likelihood**, other review cells as
**Unrecognized**, and excluded cells as **Neurons**. High Likelihood and
Unrecognized do not overlap. Unprocessed review objects may remain in a saved
result. A normal click accepts an existing High Likelihood cell; Unrecognized
and Neurons require selection followed by **Add to count**. Acceptance preserves
the existing cell ID and source masks. Command-click selects objects for bulk
actions.

An optional `point.review_hint: "likely_astrocyte"` marks a review
nuclear observation whose original Support decision is accepted and whose only
remaining limitation is the artificial XY search boundary. The engine omits
the hint when other identity or model-boundary evidence is unresolved, when the
cell's association is ambiguous, or when its associated body remains unresolved.
This display hint does not increase the automatic count. Search-boundary
geometry stays recorded separately from identity evidence, while its conservative
count-state mapping preserves the accepted v4 automatic-count policy. No
`cell_type` scientific output field is added.

`point.source` preserves `auto`, `manual` and `merged`, and adds
`support_body` (body only) and `dapi_support` (associated nucleus and body).

Each member of `point.source_observations` contains:

- `id`: stable observation ID, unique within the analysis;
- `kind`: `nucleus` or `body`;
- `artifact_id` and `label_id`: the artifact-scoped actual mask identity;
- `bounds_zyx`: original-coordinate half-open `[[z0,z1],[y0,y1],[x0,x1]]`;
- `plane_anchors`: actual source-Z strings mapped to actual mask `[x,y]` points;
- `representative_xyz`, `decision`, `reasons`, and `metrics`.

`point.plane_anchors` chooses a real observation anchor on each observed layer.
`point.plane_anchor_sources` records its source, for example:

```json
{"20": {"kind": "observation", "observation_id": "support-body-8"}}
```

The referenced observation must belong to that cell and have the exact same
anchor on the same Z. Manual anchors may instead identify their manual parent
point. Legacy and manual points need not invent mask observations.

`point.association.status` is `unmatched`, `one_to_one` or `ambiguous`, with
explicit relation evidence and competing observation IDs. An unmatched accepted
body remains accepted. A main-approved explicit one-to-one relation produces
one cell containing both observations; a global development-validation label
does not by itself define every cell's review status.

## Editing and persistence

Merge preserves complete observation payloads, deduplicated by
`(artifact_id, label_id)`, and retains the actual chosen anchor provenance.
The same key with conflicting payloads is an error, not a silent scientific
merge. A manually merged cell receives a new opaque ID while source observation
IDs stay stable. Delete/exclude and undo change cell annotation state without
rewriting scientific mask files. Session/export/reopen preserves the registry,
observations, association and per-Z anchor sources.

Official schema1 historical counts and normal manual editing remain readable.
Independent body observations enter the accepted, review or excluded cell state
directly.

## Engine execution

Both routes use one native source-statistics pass. Body work retains every
qualified computation region and uses at most two queued/running ROI jobs in
the shared four-worker CPU pool. DAPI uses the single shared model and at most
four pending native-recovery jobs. Workers obtain source views only when they
execute. Body cores touching a computation boundary request actual-source
context expansion; an explicit memory-capacity failure cannot publish a partial
count. Cancellation or either route's failure joins both routes before the
reader closes.

`body-labels.npy` contains measured positive Support voxels. Local closed-cavity
envelopes and cores are separate geometry artifacts under `body-observations/`.
Their observation metadata specifies `geometry_artifact_id: "support_bodies"`
and `geometry_path_relative_to: "label_artifact_directory"`: resolve
`geometry_path` beside that registry artifact, even when a session JSON moves.
`observations.json` retains both source observations and all measured association
relations. Timings report the concurrent routes' wall time separately; their
overlapping component times must not be summed.
