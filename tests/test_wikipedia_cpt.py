import json

from marichatmen.data.build_wikipedia_cpt import build, parse_args


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
