# Data

The full UAV vibration dataset is not stored in this repository.

Source:

- Figshare DOI: `10.6084/m9.figshare.28765640`

The Streamlit deployment can use a lightweight public demo subset under `data/demo/`.

Generate it from the original local `.xlsx` recordings with:

```bash
python scripts/make_demo_data.py --source data_full --output data/demo --windows 30
```

Do not use the demo subset to claim independent model performance; it exists only to make the public application reproducible and lightweight.
