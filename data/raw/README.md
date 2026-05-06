# Raw Data

Raw Hugging Face dataset downloads are generated locally and must not be
committed. Use `scripts/download_datasets.sh` or the build scripts to recreate
them.

Large data should live outside the source checkout when running on shared
machines, for example `/data2/antonio/MariChatmen/data/raw`.
