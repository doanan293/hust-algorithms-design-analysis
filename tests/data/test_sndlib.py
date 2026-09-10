from pathlib import Path

from data.sndlib import inventory_sndlib, parse_sndlib_xml


FIXTURE = Path("tests/data/fixtures/sndlib/mini.xml")


def test_parser_preserves_ids_values_and_units():
    network = parse_sndlib_xml(FIXTURE)
    assert [node.node_id for node in network.nodes] == ["A", "B"]
    assert network.links[0].source == "A"
    assert network.links[0].target == "B"
    assert network.demands[0].source == "A"
    assert network.demands[0].target == "B"
    assert network.demands[0].value == 12.5
    assert network.demand_unit == "MBITPERSEC"


def test_normalized_demand_is_marked_derived():
    network = parse_sndlib_xml(FIXTURE)
    payload = network.to_normalized_dict()
    assert payload["demands"][0]["provenance"] == "derived-from-sndlib-demand"
    assert payload["demands"][0]["original_value"] == "12.5"


def test_inventory_records_dangling_endpoint(tmp_path: Path):
    text = FIXTURE.read_text(encoding="utf-8").replace("<target>B</target>", "<target>missing</target>")
    path = tmp_path / "dangling.xml"
    path.write_text(text, encoding="utf-8")
    record = inventory_sndlib(path)
    assert record.parser_status == "invalid"
    assert "dangling endpoint" in record.error


def test_inventory_records_missing_demand_value(tmp_path: Path):
    text = FIXTURE.read_text(encoding="utf-8").replace("<demandValue>12.5</demandValue>", "")
    path = tmp_path / "missing-value.xml"
    path.write_text(text, encoding="utf-8")
    record = inventory_sndlib(path)
    assert record.parser_status == "invalid"
    assert "missing demandValue" in record.error
