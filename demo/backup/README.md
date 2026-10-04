# PMDB dashboard backup video

`pmdb-dashboard-demo.mp4` is an H.264 recording of the dashboard's six-view scripted walkthrough.

Regenerate it from the repository root:

1. Start the dashboard: `PATH=~/.local/node/bin:$PATH node demo/server.mjs --port 8080` (binds 127.0.0.1; set `HOST=0.0.0.0` to expose on the LAN)
2. Record it: `python demo/record_backup.py`

The pre-existing `outputs/clips/01_confound_flip.mp4` … `04_verdict_reject.mp4` pitch clips are a second fallback.

## Director's cut

`pmdb-cut.mp4` (~78 s, 1600×900, silent, captioned) is the same story as `pmdb-submission.mp4` with a camera that
zooms onto the panel being discussed, one short caption per beat and a chapter strip (trap → finding → proof → calls → QC).
Captions are rendered from the same facts and invariants as the submission video.

    python demo/record_cut.py            # dashboard must be running; add --allow-unreviewed gnn,patch_mil
                                         # only after checking that those outputs/ dirs don't change the story
