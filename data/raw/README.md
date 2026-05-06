# Raw Data

Raw Hugging Face dataset downloads are generated locally and must not be
committed. Use `scripts/download_datasets.sh` or the build scripts to recreate
them.

Large data should live outside the source checkout when running on shared
machines, for example `/data2/antonio/MariChatmen/data/raw`.

## Spanish Wikipedia

When the Spanish Wikipedia dump is available, place it under the artifact root,
not in git:

```text
/data2/antonio/MariChatmen/data/raw/wikipedia/eswiki/20260501/
  eswiki-20260501-pages-articles-multistream.xml.bz2
```

Source:

```text
https://dumps.wikimedia.org/eswiki/20260501/
```

Process it with:

```bash
uv run python -m marichatmen.data.build_wikipedia_cpt \
  --dump_file /data2/antonio/MariChatmen/data/raw/wikipedia/eswiki/20260501/eswiki-20260501-pages-articles-multistream.xml.bz2 \
  --out_dir /data2/antonio/MariChatmen/data/processed/cpt_wikipedia_eswiki_20260501 \
  --n_train 100000 \
  --n_valid 5000 \
  --n_probe 1000 \
  --andaluh_ratio 0.9
```

Licence note: Wikimedia text is reused under CC BY-SA 4.0 and GFDL. Keep the
generated manifest with any derived CPT dataset.
