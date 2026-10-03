# PMDB
Hackathon

## Data loading

```python
from pmdb.io import list_sites, load_site

manifest = list_sites()
site = load_site("Batch_1", "4ih2ggld", resolution="half")
raw = load_site("Batch_1", "4ih2ggld", resolution="half", normalise="none")
```
