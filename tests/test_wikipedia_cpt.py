import json

from marichatmen.data.filter_cpt import filter_rows, parse_args as parse_filter_args
from marichatmen.data.build_wikipedia_cpt import build, parse_args
from marichatmen.data.build_wikipedia_cpt import clean_wikitext


def test_build_wikipedia_cpt_from_xml(tmp_path):
    dump = tmp_path / "eswiki-mini.xml"
    long_text = " ".join(["El modelo aprende de los datos de entrenamiento."] * 80)
    dump.write_text(
        f"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Aprendizaje automático</title>
    <ns>0</ns>
    <revision>
      <text xml:space="preserve">{long_text}</text>
    </revision>
  </page>
  <page>
    <title>Redirección</title>
    <ns>0</ns>
    <redirect title="Otro" />
    <revision><text>Texto ignorado.</text></revision>
  </page>
</mediawiki>
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    args = parse_args(
        [
            "--dump_file",
            str(dump),
            "--out_dir",
            str(out_dir),
            "--n_train",
            "1",
            "--n_valid",
            "0",
            "--n_probe",
            "1",
            "--min_words",
            "10",
            "--max_words",
            "1000",
            "--andaluh_ratio",
            "1.0",
        ]
    )

    build(args)

    rows = [json.loads(line) for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["metadata"]["source_license"] == "CC-BY-SA-4.0/GFDL"
    assert rows[0]["metadata"]["source_article_title"] == "Aprendizaje automático"
    assert rows[0]["metadata"]["transformation"] == "andaluh_epa_sevillian_ce"
    assert manifest["source_url"] == "https://dumps.wikimedia.org/eswiki/20260501/"


def test_filter_cpt_preserves_and_reports_split_view_metadata(tmp_path):
    src = tmp_path / "cpt.jsonl"
    long_text = " ".join(["Er modelo aprende bien de loh datoh limpioh."] * 20)
    src.write_text(
        "\n".join(
            [
                json.dumps({"text": long_text, "metadata": {"split_view": "andaluh"}}, ensure_ascii=False),
                json.dumps({"text": long_text, "metadata": {"split_view": "spanish"}}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "filtered.jsonl"
    manifest = tmp_path / "manifest.json"

    filter_rows(
        parse_filter_args(
            [
                "--input",
                str(src),
                "--output",
                str(out),
                "--manifest",
                str(manifest),
                "--min_words",
                "10",
                "--max_numeric_tokens",
                "8",
                "--max_repetition_rate",
                "1.0",
            ]
        )
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert {row["metadata"]["split_view"] for row in rows} == {"andaluh", "spanish"}
    assert data["written_split_view_counts"] == {"andaluh": 1, "spanish": 1}


def test_filter_cpt_rejects_chat_templates_and_wiki_artifacts(tmp_path):
    src = tmp_path / "cpt.jsonl"
    clean_text = " ".join(["Er modelo aprende bien de loh datoh limpioh."] * 20)
    chat_text = "<|im_start|>system\nEres un asistente<|im_end|>\n" + clean_text
    wiki_text = "tumb|rîtt|150px|Imagen rota\n\n" + clean_text
    src.write_text(
        "\n".join(
            [
                json.dumps({"text": clean_text, "metadata": {"split_view": "andaluh"}}, ensure_ascii=False),
                json.dumps({"text": chat_text, "metadata": {"split_view": "andaluh"}}, ensure_ascii=False),
                json.dumps({"text": wiki_text, "metadata": {"split_view": "andaluh"}}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "filtered.jsonl"
    manifest = tmp_path / "manifest.json"

    filter_rows(
        parse_filter_args(
            [
                "--input",
                str(src),
                "--output",
                str(out),
                "--manifest",
                str(manifest),
                "--min_words",
                "10",
                "--max_repetition_rate",
                "1.0",
            ]
        )
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["text"] == clean_text
    assert data["rejected"]["chat_template"] == 1
    assert data["rejected"]["wiki_markup"] == 1


def test_clean_wikitext_drops_file_thumb_pipes():
    raw = """
[[Archivo:Ejemplo.jpg|thumb|right|150px|Pie de foto]]

La validación cruzada divide los datos en partes para entrenar y probar.
"""

    cleaned = clean_wikitext(raw)

    assert "thumb" not in cleaned.lower()
    assert "150px" not in cleaned
    assert "validación cruzada" in cleaned
