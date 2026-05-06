# Blog Publication TODO

Status snapshot: 2026-05-06.

The blog is **not publication-ready yet**. The draft is useful, but it still
mixes final 0.8B results with interim 2B/4B material. Before publishing, update
the article with the final remote artifacts and the Hugging Face model links.

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

Only partial checkpoints were found for these:

```text
/data2/antonio/MariChatmen/outputs/qwen_andaluh_2b_sft_overnight_smart_base_long_20260505_233037/checkpoint-550
/data2/antonio/MariChatmen/outputs/qwen_andaluh_2b_sft_overnight_smart_base_long_20260505_233037/checkpoint-600
/data2/antonio/MariChatmen/outputs/qwen_andaluh_4b_cpt_overnight_smart_base_long_20260505_233037/checkpoint-50
/data2/antonio/MariChatmen/outputs/qwen_andaluh_4b_cpt_overnight_smart_base_long_20260505_233037/checkpoint-100
```

Before publication, decide whether the 2B SFT and 4B CPT results are final
enough to report. If they are not final, keep them out of the headline results
and describe them as ongoing or omit them entirely.

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

- final 0.8B Qwen-Andaluh CPT/SFT/ORPO losses
- final 0.8B MariChatmen SFT/ORPO/GRPO losses and rewards
- ORPO reward accuracy, reward margins, log-odds ratio, and NLL by rejection class
- GRPO reward components, entropy, KL, clipped ratio, and zero-std reward groups
- bootstrap 95% confidence intervals over evaluation prompts
- decoding settings for every reported generation table
- clipped-completion rate and termination reason summary
- final benchmark table
- final 2B status
- final 4B status
- GPU utilisation plots
- best and worst qualitative examples
- clear statement of whether each model is release-ready

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
