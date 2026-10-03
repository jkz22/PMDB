# Held-back test sites (unlabelled)

Source: organiser Google Drive folder `Hackathon-Polaron-test`
(https://drive.google.com/drive/folders/1pZa4Q0kSpxeAHi4rPHzMBKq3fOPaGeCd), downloaded 2026-10-03.

Kept separate from `data/` on purpose: these sites carry **no batch label** and must
never enter training or `list_sites()` (which only scans `data/Batch_*`). Treat as read-only.

| site | detectors | shape (H, W, 3) |
|---|---|---|
| 3e122cbj | BSE, Inlens, ETD | 2316 × 6996 |
| fn0mhxef | BSE, Inlens, ETD | 2048 × 7000 |
| xrv9xvzb | BSE, Inlens, ETD | 2088 × 7000 |

Same format as `data/`: uint8, R == G == B, 25 nm/px.

## Task framing (organiser clarification)

> Treat batch 3 as the baseline, it's what's been "promised" by the supplier. Batch 1 and 2
> arrived subsequently, and we're trying to tell if they are different. They are not explicitly
> better or worse than the batch 3 baseline, but they show the types of variation we need your
> models to pick up on.
>
> The judging criteria reflects this; can you identify what's different about the batches, and
> thus categorise the held back samples correctly. If you can, this implies unknown batch N could
> be categorised accurately as in or out of distribution - helping manufacturers make critical
> decisions about when to accept and reject a batch!

Deliverable per held-back site: **always** a batch assignment (Batch_1 / Batch_2 / Batch_3),
plus a confidence and an explanation of how it differs from the Batch 3 baseline. Low
confidence is acceptable; abstaining is not.
