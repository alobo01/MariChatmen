# Blog Publication TODO

Status snapshot: 2026-05-06.

The blog is **not publication-ready yet**. The draft is useful, but it still
must be kept as an interim project log until the 4B Qwen-Andaluh quality run
finishes and passes the release gates. The 0.8B results are systems validation
only.

## 0. Current Truth To Preserve

- Do not present the 0.8B adapter as a quality model.
- Do not present MariChatmen persona as ready.
- Qwen-Andaluh and MariChatmen must remain separate in the narrative.
- Persona training is frozen for new quality work until Qwen-Andaluh passes:

```text
MARI-AAS > 80
Spanish leak rate < 7%
semantic drift penalty < 0.08
instruction following > 0.80
45/50 manual samples clearly Andaluh
40/50 manual samples semantically correct
0 critical safety failures
```

- The next headline model is `Qwen/Qwen3.5-4B-Base`, not 0.8B.
- Spanish Wikipedia `eswiki` 2026-05-01 is being integrated as CPT data and
  must be cited as CC BY-SA 4.0/GFDL text, not plain CC BY 4.0.
- The report should stay technical but readable: every metric section needs at
  least one concrete example, such as a Spanish -> Andaluh sentence, a
  high-saving tokenizer token, or a chosen/rejected ORPO pair.
- The written target is `sevillian_ce`, favouring `ç` output rather than the
  plainer `s` variant.
- Be explicit that the current runs have **not** used all available data. The
  repository now has enough raw/source material for much larger Andalûh runs,
  but completed runs are still subset runs:
  - `base_long` on Conway: 20k CPT rows, 20k SFT rows, 10k ORPO pairs.
  - Wikipedia quick subset: 4k CPT train rows, 400 valid rows, 200 probe rows.
  - Persona file split: 9k persona SFT train rows, 1.8k persona ORPO train
    pairs, 1.5k GRPO prompts.
  These are useful engineering milestones, not the final 50M-100M-token 4B CPT
  corpus.
- Add a section on the planned MariChatmen self-verified reward. The key point:
  deterministic MARI-AAS/MARI-PAS is not enough, because the model can learn
  persona keywords without answering correctly.

## 0.1 Process The Wikipedia Dump

When the dump is copied to Conway, expected location:

```text
/data2/antonio/MariChatmen/data/raw/wikipedia/eswiki/20260501/eswiki-20260501-pages-articles-multistream.xml.bz2
```

Process command:

```bash
cd /home/antonio/MariChatmen
UV_PROJECT_ENVIRONMENT=/data2/antonio/MariChatmen/.venv \
UV_CACHE_DIR=/data2/antonio/MariChatmen/.uv_cache \
MCM_ARTIFACT_ROOT=/data2/antonio/MariChatmen \
uv run python -m marichatmen.data.build_wikipedia_cpt \
  --dump_file /data2/antonio/MariChatmen/data/raw/wikipedia/eswiki/20260501/eswiki-20260501-pages-articles-multistream.xml.bz2 \
  --out_dir /data2/antonio/MariChatmen/data/processed/cpt_wikipedia_eswiki_20260501 \
  --n_train 100000 \
  --n_valid 5000 \
  --n_probe 1000 \
  --andaluh_ratio 0.9
```

Check that this exists before training:

```text
/data2/antonio/MariChatmen/data/processed/cpt_wikipedia_eswiki_20260501/manifest.json
```

Citation to use:

```text
Spanish Wikipedia contributors, "eswiki dump 20260501",
Wikimedia Dumps, https://dumps.wikimedia.org/eswiki/20260501/.
Text reused under CC BY-SA 4.0 and GFDL; transformed to Andaluh EPA for CPT.
```

## 1. Download The Latest Conway Results

Remote host:

```bash
antonio@conway.cs.ox.ac.uk
```

Remote artifact root:

```bash
/data2/antonio/MariChatmen
```

Main run tag:

```bash
overnight_smart_base_long_20260505_233037
```

Current blog bundle found on Conway:

```bash
/data2/antonio/MariChatmen/reports/blogpost_bundle/overnight_smart_base_long_20260505_233037_interim_20260506
```

Download it locally:

```bash
mkdir -p blog/conway-results/overnight_smart_base_long_20260505_233037

rsync -avz \
  antonio@conway.cs.ox.ac.uk:/data2/antonio/MariChatmen/reports/blogpost_bundle/overnight_smart_base_long_20260505_233037_interim_20260506/ \
  blog/conway-results/overnight_smart_base_long_20260505_233037/
```

Also fetch the raw logs, samples, and plots, but **not model weights**:

```bash
mkdir -p blog/conway-results/raw

rsync -avz \
  --include='*/' \
  --include='*.json' \
  --include='*.jsonl' \
  --include='*.csv' \
  --include='*.md' \
  --include='*.svg' \
  --include='*.txt' \
  --exclude='*' \
  antonio@conway.cs.ox.ac.uk:/data2/antonio/MariChatmen/reports/ \
  blog/conway-results/raw/reports/
```

Check what final adapters exist:

```bash
ssh antonio@conway.cs.ox.ac.uk \
  'find /data2/antonio/MariChatmen/outputs -maxdepth 3 -type d -name final_adapter -print | sort'
```

## 2. Current Remote Artifact Status

Confirmed completed adapter directories:

```text
/data2/antonio/MariChatmen/outputs/qwen_andaluh_08b_cpt/final_adapter
/data2/antonio/MariChatmen/outputs/qwen_andaluh_08b_sft/final_adapter
/data2/antonio/MariChatmen/outputs/qwen_andaluh_08b_orpo/final_adapter
/data2/antonio/MariChatmen/outputs/marichatmen_08b_persona_file_sft_overnight_smart_base_long_20260505_233037/final_adapter
/data2/antonio/MariChatmen/outputs/marichatmen_08b_persona_file_orpo_overnight_smart_base_long_20260505_233037/final_adapter
/data2/antonio/MariChatmen/outputs/marichatmen_08b_persona_file_grpo_overnight_smart_base_long_20260505_233037/final_adapter
/data2/antonio/MariChatmen/outputs/qwen_andaluh_2b_cpt_overnight_smart_base_long_20260505_233037/final_adapter
```

Historical/interim partial checkpoints were found for these:

```text
/data2/antonio/MariChatmen/outputs/qwen_andaluh_2b_sft_overnight_smart_base_long_20260505_233037/checkpoint-550
/data2/antonio/MariChatmen/outputs/qwen_andaluh_2b_sft_overnight_smart_base_long_20260505_233037/checkpoint-600
/data2/antonio/MariChatmen/outputs/qwen_andaluh_4b_cpt_overnight_smart_base_long_20260505_233037/checkpoint-50
/data2/antonio/MariChatmen/outputs/qwen_andaluh_4b_cpt_overnight_smart_base_long_20260505_233037/checkpoint-100
```

Before publication, do not treat the old 2B SFT and old 4B CPT results as
headline results. The new target is a quality 4B-Base run with expanded
tokenizer and longer training.

Inspect run logs here:

```bash
ssh antonio@conway.cs.ox.ac.uk \
  'ls -lh /data2/antonio/MariChatmen/reports/overnight_logs/overnight_smart_base_long_20260505_233037 && tail -200 /data2/antonio/MariChatmen/reports/overnight_logs/overnight_smart_base_long_20260505_233037/driver.log'
```

## 3. Update The Blog Results

Files to update after downloading the final artifacts:

```text
blog/marichatmen-blog-repo/data/results.json
blog/marichatmen-blog-repo/post.md
blog/marichatmen-blog-repo/widgets.md
blog/marichatmen-blog-repo/examples.md
blog/marichatmen-blog-repo/assets/
```

Replace interim claims with final ones:

- final 4B Qwen-Andaluh CPT/SFT/ORPO losses
- final 4B Qwen-Andaluh MARI-AAS and Spanish leak rate
- Wikipedia CPT token count and manifest path
- top tokenizer-expanded Andaluh tokens with frequency, base-piece count, and
  saving score
- example transformation table using `ç`, e.g. `Sevilla sabe hacer cosas
  bonitas` -> `Çebiya çabe açêh coçâ bonitâ`
- ORPO reward accuracy, reward margins, log-odds ratio, and NLL by rejection class
- persona GRPO reward components only if persona is resumed after gates pass
- bootstrap 95% confidence intervals over evaluation prompts
- decoding settings for every reported generation table
- clipped-completion rate and termination reason summary
- final benchmark table
- final 2B baseline status, if useful
- final 4B quality status
- GPU utilisation plots
- best and worst qualitative examples
- clear statement of whether each model is release-ready
- explicit data-utilisation table: data available, data processed, data used by
  each completed run, and what is still unused
- self-verified reward design for MariChatmen:
  - deterministic checks: accent, persona, repetition, hostility, alcohol safety
  - verifier check: did the answer actually address the prompt?
  - evidence requirement: verifier must identify short spans supporting its score
  - reward agreement: reward only when verifier and deterministic metrics agree

Do not publish a MariChatmen persona article until the base Qwen-Andaluh gates
have actually passed.

Use the downloaded plots from:

```text
blog/conway-results/overnight_smart_base_long_20260505_233037/plots/
```

Do not leave the article saying "interim" if the final results have been
downloaded and checked.

## 4. Upload Hugging Face Models And Add Links

The persona seed dataset has been uploaded:

```text
MariChatmen Persona dataset:
https://huggingface.co/datasets/MariChatmen/MariChatmen-Persona

Latest dataset commit:
https://huggingface.co/datasets/MariChatmen/MariChatmen-Persona/commit/1e5d859b336bbe147882703b89c8c96a2fca6036
```

I did **not** find any uploaded public Hugging Face model links recorded in the
repo or via public search for `MariChatmen` / `Qwen-Andaluh` on 2026-05-06.
So the links below are placeholders until the adapters are actually pushed.

Fill these before publication:

```text
Qwen-Andaluh 0.8B adapter:
https://huggingface.co/MariChatmen/qwen-andaluh-0.8b-lora

Qwen-Andaluh 0.8B upload commit:
https://huggingface.co/MariChatmen/qwen-andaluh-0.8b-lora/commit/932d680d82ea0f0dd1da7cf9c5609d42df705158

MariChatmen 0.8B adapter:
https://huggingface.co/MariChatmen/marichatmen-0.8b-lora

MariChatmen 0.8B upload commit:
https://huggingface.co/MariChatmen/marichatmen-0.8b-lora/commit/838878e7a351f46c8a3f44a05de908bb249ae4c7

Qwen-Andaluh 2B adapter:
https://huggingface.co/<namespace>/qwen-andaluh-2b-lora

Qwen-Andaluh 4B adapter:
https://huggingface.co/<namespace>/qwen-andaluh-4b-lora

Optional MariChatmen 2B adapter:
https://huggingface.co/<namespace>/marichatmen-2b-lora

Optional MariChatmen 4B adapter:
https://huggingface.co/<namespace>/marichatmen-4b-lora
```

Example upload command for an adapter:

```bash
huggingface-cli repo create <namespace>/marichatmen-0.8b-lora \
  --type model \
  --private

huggingface-cli upload <namespace>/marichatmen-0.8b-lora \
  /data2/antonio/MariChatmen/outputs/marichatmen_08b_persona_file_grpo_overnight_smart_base_long_20260505_233037/final_adapter \
  . \
  --repo-type model
```

After upload, add the real links to:

```text
blog/marichatmen-blog-repo/post.md
blog/marichatmen-blog-repo/sources.md
README.md
```

## 5. Add Model Cards Before Publishing Links

Each Hugging Face model page should include:

- base model ID
- adapter type: LoRA / QLoRA adapter
- training stages included
- system prompt policy
- datasets used
- licence notes
- evaluation table
- warning that MariChatmen is fictional and Sevillian-leaning
- warning that it does not represent all Andalusian varieties
- sample prompts and outputs
- known failures: Spanish leakage, truncation, low persona score, missing
  province flourish if still true

Do **not** upload:

- raw Menuda Noche material
- unreviewed private examples
- raw Hugging Face datasets
- merged models before adapter review and licence audit

## 6. Publication Checklist

Before publishing:

- [ ] Download the latest Conway result bundle.
- [ ] Confirm whether 2B SFT reached a final adapter or only checkpoints.
- [ ] Confirm whether 4B reached a final adapter or only CPT checkpoints.
- [ ] Update `results.json` with final numbers.
- [ ] Add ORPO/GRPO diagnostic metrics, not only loss curves.
- [ ] Add bootstrap confidence intervals for MARI-AAS, MARI-PAS, MARI-TOTAL and leak rates.
- [ ] Add decoding settings and seed next to each reported metric table.
- [ ] Replace `not exported yet` and `not bootstrapped yet` placeholders in the draft.
- [ ] Add validation split notes: in-distribution, held-out prompt family, human/gold, adversarial, and safety sets.
- [ ] Add human calibration if available: MARI metric Spearman correlation with blind pairwise preferences.
- [ ] Replace placeholder SVGs with real plots where useful.
- [ ] Add final sample outputs for Base, SFT, ORPO, and GRPO.
- [ ] Add a data-utilisation table so the reader can see that current runs are
      subset runs, not "all the data".
- [ ] Implement and calibrate the Mari self-verified reward before serious
      persona GRPO.
- [x] Push persona seed dataset to Hugging Face.
- [x] Push 0.8B Qwen-Andaluh and MariChatmen adapters to Hugging Face.
- [ ] Push remaining 2B/4B adapter models to Hugging Face if release-worthy.
- [ ] Replace remaining `<namespace>` Hugging Face placeholders with real links.
- [ ] Add model cards and licence notes to every HF repo.
- [ ] Re-read the article and remove any stale "interim" wording.
- [ ] Check that no private data or large artifacts are committed.
- [ ] Run `bash scripts/run_tests.sh`.

## 7. Final Expected Blog Links Section

Add a section like this to the final post once the uploads exist:

```markdown
## Models

- Qwen-Andaluh 0.8B LoRA adapter: <REAL_HF_URL>
- MariChatmen 0.8B LoRA adapter: <REAL_HF_URL>
- Qwen-Andaluh 2B LoRA adapter: <REAL_HF_URL_OR_OMIT_IF_NOT_READY>
- Qwen-Andaluh 4B LoRA adapter: <REAL_HF_URL_OR_OMIT_IF_NOT_READY>
```

If a model is not uploaded or not good enough, omit it rather than linking a
weak or incomplete checkpoint.
