# Research Data Preparation Design

**Date:** 2026-09-10
**Status:** Approved for implementation planning

## 1. Purpose

Prepare a reproducible data foundation for the final paper on joint UAV trajectory, relay selection, and bandwidth allocation. The pipeline must cover all three public-data roles required by the research direction while avoiding unnecessary bulk downloads:

- SNDlib supplies published network instances and demand information.
- Topology Zoo supplies diverse relay/backhaul graph structures and geographic coordinates.
- RescueNet supplies damage-region geometry for the final case study.
- Seeded generators supply the synthetic alerts, failures, channels, and UAV scenarios needed to reproduce Huang-style simulations.

Completeness means that every experiment in the paper can be reconstructed from scripts, manifests, configuration, and pinned public sources. It does not mean downloading every available image when the experiment uses only a declared subset.

## 2. Scope

### 2.1 Included

- Pin and download the complete Topology Zoo repository.
- Discover and download official SNDlib instances that expose topology and demand data in a format supported by the parser.
- Download the RescueNet segmentation label descriptor and validation archive.
- Build deterministic inventories for all downloaded sources.
- Select a fixed RescueNet case-study subset of 30 image-mask pairs.
- Normalize eligible network graphs and create reproducible scenario inputs.
- Record source URLs, versions, licenses, sizes, checksums, selection decisions, and exclusion reasons.
- Supply `pilot`, `paper`, and optional `full-rescuenet` acquisition profiles.
- Test acquisition, integrity checks, parsing, filtering, pairing, and deterministic scenario generation.

### 2.2 Excluded from the default paper profile

- RescueNet training and test archives. The project does not train a segmentation model, so these archives do not contribute to the planned experiments.
- Redistribution of raw third-party data through Git.
- Silent use of an unofficial mirror when a primary source fails.
- Model optimization, baseline implementation, and experiment execution. They consume this pipeline but are separate implementation phases.

The optional `full-rescuenet` profile may fetch all official segmentation archives without changing the processing interface. It is not required to reproduce the planned paper.

## 3. Considered Approaches

### 3.1 Script and manifest workflow — selected

Small downloader and processing commands use checked-in configuration and manifests. Raw files remain outside Git, while provenance and transformation rules remain reviewable. This approach has the least operational overhead and fits a course reproducibility package.

### 3.2 Data Version Control

DVC would provide content-addressed dataset versions and remote storage integration. It was rejected for the first version because it introduces another service and dependency without improving the reproducibility of immutable public downloads enough to justify the cost.

### 3.3 Manual downloads

Manual acquisition is initially quick but does not reliably preserve exact URLs, checksums, interrupted-download behavior, or exclusion decisions. It was rejected as the primary workflow.

## 4. Repository Layout

```text
data/
  raw/
    sndlib/
    topology-zoo/
    rescuenet/
  manifests/
    sources.json
    topology_inventory.csv
    sndlib_inventory.csv
    rescuenet_inventory.csv
    rescuenet_selection.csv
  processed/
    networks/
    scenarios/
configs/
  data/
    pilot.yaml
    paper.yaml
    full-rescuenet.yaml
scripts/
  data/
    download.py
    verify.py
    inventory.py
    preprocess.py
src/
  data/
    acquisition.py
    manifests.py
    topology_zoo.py
    sndlib.py
    rescuenet.py
    scenarios.py
tests/
  data/
```

`data/raw/` and large generated artifacts under `data/processed/` are ignored by Git. Manifests, configurations, scripts, source modules, tests, and small test fixtures are committed.

The command-line scripts are thin entry points. Source-specific behavior lives in focused modules under `src/data/` so it can be tested without invoking a real network download.

## 5. Acquisition Profiles

### 5.1 Pilot

The `pilot` profile downloads the Topology Zoo repository, a small supported SNDlib instance set, the RescueNet descriptor, and source metadata. It is intended to validate parsers and scenario generation before the larger archive transfer.

### 5.2 Paper

The `paper` profile downloads:

- the complete Topology Zoo repository pinned to commit `e278b1bdaafea5dac33883bf9c97401db4cd7347`;
- every official SNDlib instance listed in the committed paper-source manifest that contains both a usable graph and demand records;
- RescueNet's label descriptor and the 2,373,908,074-byte semantic-segmentation validation archive, version 1, file ID `40582916`.

This profile is the reproducibility contract for the final experiments.

### 5.3 Full RescueNet

The optional `full-rescuenet` profile adds the version-1 training and test segmentation archives. Their current official sizes are 18,699,171,789 and 2,387,196,841 bytes. The pipeline records the metadata returned by Figshare rather than treating these values as permanent constants.

## 6. Source-Specific Selection Rules

### 6.1 Topology Zoo

The downloader fetches the pinned commit directly with bounded Git history and checks it out. Inventory processing reads GraphML files without modifying them. A topology is eligible for the core benchmark when it:

- has 15 through 40 nodes after parsing;
- has valid latitude and longitude for every retained node;
- is connected when interpreted as an undirected backbone;
- can be converted to a simple graph using a documented parallel-edge rule;
- contains no non-finite coordinates.

All eligible graphs remain in the inventory. The inventory includes raw node and edge counts, coordinate completeness, connectivity, parallel-edge status, and either `eligible` or a specific exclusion reason. Development and evaluation selections are made by sorted topology ID followed by a fixed seed, so filesystem enumeration order cannot change the split.

### 6.2 SNDlib

Acquisition begins from the official SNDlib service and records the catalogue response used for discovery. The discovery result is materialized as an exact, reviewable source list in the paper-source manifest; later paper runs consume that list instead of silently expanding when the upstream catalogue changes. An instance enters the list when its official record exposes nodes, links, and demands supported by the parser. Original identifiers, units, and values are retained in raw form.

The normalized inventory records graph size, demand count, coordinate availability, units declared by the source, and parser status. Capacity-design or module fields are not interpreted as operational link capacity unless the source semantics establish that meaning. Demand-to-alert conversion is a later deterministic transformation whose rule and seed are stored in the scenario configuration; converted alerts are never labelled as original SNDlib demand.

If the primary service is unavailable, the run fails with a source-specific error and preserves its partial-download metadata. An unofficial mirror is not substituted automatically.

### 6.3 RescueNet

The paper profile fetches the official descriptor and validation ZIP because Figshare distributes the validation set as one archive. After checksum verification and safe extraction, the inventory pairs images and masks by canonical relative identifier and rejects duplicates or missing partners.

The case-study subset contains exactly 30 pairs. Selection is deterministic:

1. Compute damage-class pixel counts from each valid mask using the official descriptor.
2. Group images by the damage-class presence vector.
3. Sort within each group by canonical relative identifier.
4. Use the configured seed to select pairs in round-robin order across nonempty groups until 30 are selected.
5. If fewer than 30 valid pairs exist, stop with an error rather than duplicate images.

The selection manifest records each chosen pair, source archive, relevant class counts, selection seed, and rank. The pipeline does not infer victims, deadlines, channel measurements, altitude, or no-fly zones from the masks.

## 7. Manifest Contract

`sources.json` is the acquisition ledger. Each artifact record contains:

- stable source ID and dataset family;
- official landing page and exact download URL;
- retrieval timestamp in UTC;
- upstream version, revision, commit, DOI, or file ID;
- filename and byte size;
- upstream checksum when supplied;
- locally computed SHA-256;
- license name, license URL, and the page from which the license claim was obtained;
- acquisition profile and status.

Inventories and selections use CSV for easy inspection and analysis. Every processed network or scenario includes the relevant raw SHA-256 values and a hash of its processing configuration, allowing a result to be traced back to inputs and transformation rules.

If different official surfaces state different licenses, the manifest retains both claims and marks the artifact `license-review-required`. Local research processing may continue, but redistribution packaging excludes the raw artifact until the discrepancy is resolved.

## 8. Processing and Data Flow

```text
Pinned config
    -> acquire to temporary partial files
    -> validate size and upstream checksum
    -> atomically publish immutable raw artifacts
    -> compute SHA-256 and write source ledger
    -> parse source-specific inventory
    -> apply deterministic eligibility and selection rules
    -> normalize graphs, coordinates, demands, and target points
    -> generate seeded scenarios
    -> validate schema and invariants
    -> publish processed artifacts and provenance hashes
```

Geographic coordinates are projected to a metric coordinate system before normalization. Each selected network is uniformly scaled into the configured square region without independently stretching axes. The scale, projection inputs, bounding box, and resulting coordinates are recorded.

Scenario generation receives normalized data plus an explicit seed. It produces topology, relay candidates, rescue center, alerts, failures, channel configuration, UAV parameters, and development/evaluation split metadata. Random-number streams are named by concern so adding a new random feature does not silently change existing topology-failure or workload draws.

## 9. Failure Handling and Safety

- Downloads use partial files and resume when the server supports byte ranges.
- A valid existing artifact is reused; an invalid artifact is quarantined rather than overwritten silently.
- Final publication into `data/raw/` is atomic.
- ZIP extraction rejects absolute paths, parent traversal, symlinks, and entries outside the target directory.
- HTTP failure, incorrect size, checksum mismatch, unsupported schema, missing coordinates, disconnected graphs, image-mask mismatch, and insufficient RescueNet pairs produce distinct errors.
- A failed source does not cause another source or mirror to be used without an explicit configuration change.
- Raw artifacts are treated as immutable. Reprocessing writes versioned or content-addressed outputs rather than modifying raw files.
- Logs avoid embedding credentials or private URLs.

## 10. Testing Strategy

Unit tests use small local fixtures and mocked HTTP responses. They cover:

- clean download, interrupted download and resume, retryable failure, and non-retryable failure;
- size, MD5 when provided upstream, and SHA-256 verification;
- idempotent repeated execution;
- quarantine behavior for corrupt existing files;
- safe ZIP extraction;
- GraphML multiedge conversion, coordinate validation, and connectedness filtering;
- SNDlib parsing with units preserved and unsupported fields reported;
- RescueNet image-mask pairing, class counting, deterministic stratified selection, and the fewer-than-30 failure;
- deterministic scenario output for the same seed and changed output for a different seed;
- provenance hashes linking processed records to raw artifacts and configuration.

An integration smoke test runs the full pipeline against tiny fixtures without internet access. An opt-in live-source test checks source metadata and a small download but is excluded from the default test suite to avoid making normal tests depend on network availability.

## 11. User Interface

The workflow exposes one command for each lifecycle stage and one aggregate command documented in the repository README. The aggregate paper workflow performs acquisition, verification, inventory, selection, and preprocessing in order. Each stage can be rerun independently.

Commands support a profile, data root, and dry-run flag. Dry-run reports expected sources, sizes known from metadata, and actions without transferring files. Verification can run independently after acquisition. Machine-readable logs accompany concise terminal progress.

## 12. Acceptance Criteria

The data-preparation work is complete when:

1. A clean checkout can use documented commands to reconstruct the paper dataset from official sources.
2. Topology Zoo is pinned and every parsed topology has an eligibility decision with a reason.
3. Every supported official SNDlib instance has an inventory record, with original units retained.
4. RescueNet validation data pass integrity checks and produce the same 30-pair manifest for the configured seed.
5. Every raw artifact has provenance and a locally computed SHA-256.
6. Every processed artifact identifies its raw inputs, configuration hash, and seed.
7. Repeated runs are idempotent and do not damage valid downloads.
8. Default tests run offline and pass; opt-in source checks report current upstream drift separately.
9. Raw and large generated data are excluded from Git.
10. The README distinguishes the lightweight pilot, reproducible paper, and optional full-RescueNet profiles.

## 13. Implementation Boundary

This design prepares data only. Evaluator logic, optimization algorithms, baselines, statistical analysis, and report results will be designed and implemented in their own phases after the data contract is stable. The normalized scenario schema is the interface between this pipeline and those later components.
