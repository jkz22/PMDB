# Stale outputs

`site_explanations.csv` and `batch_directions.csv` were produced before the Devin review fixes on #106
(net KPI direction per site; unexplained PCs carry no KPI label). 23 of 40 justifications in
`site_explanations.csv` state the same KPI as both higher and lower. Regenerating needs the classifier and PC
scores, which come from the Modal embeddings (not committed). Rerun:

    modal run modal_patch_mil.py --mode explain

then `--mode explain-test` for `test_*.csv`. Delete this file after regenerating.
