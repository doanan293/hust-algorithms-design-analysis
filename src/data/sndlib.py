from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import xml.etree.ElementTree as ET


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if _local(child.tag) == name]


def _text(element: ET.Element, name: str, path: Path) -> str:
    match = next((item for item in element.iter() if _local(item.tag) == name), None)
    if match is None or match.text is None or not match.text.strip():
        raise ValueError(f"{path}: missing {name}")
    return match.text.strip()


def _optional_text(root: ET.Element, element_name: str, attribute: str) -> str | None:
    match = next((item for item in root.iter() if _local(item.tag) == element_name), None)
    return None if match is None else match.attrib.get(attribute)


@dataclass(frozen=True)
class SndlibNode:
    node_id: str
    x: Decimal | None
    y: Decimal | None


@dataclass(frozen=True)
class SndlibLink:
    link_id: str
    source: str
    target: str


@dataclass(frozen=True)
class SndlibDemand:
    demand_id: str
    source: str
    target: str
    value: Decimal


@dataclass(frozen=True)
class SndlibNetwork:
    network_id: str
    nodes: tuple[SndlibNode, ...]
    links: tuple[SndlibLink, ...]
    demands: tuple[SndlibDemand, ...]
    demand_unit: str | None
    capacity_unit: str | None

    def to_normalized_dict(self) -> dict[str, object]:
        return {
            "network_id": self.network_id,
            "nodes": [
                {"id": node.node_id, "x": str(node.x), "y": str(node.y)}
                for node in self.nodes
            ],
            "links": [
                {"id": link.link_id, "source": link.source, "target": link.target}
                for link in self.links
            ],
            "demand_unit": self.demand_unit,
            "capacity_unit": self.capacity_unit,
            "demands": [
                {
                    "id": demand.demand_id,
                    "source": demand.source,
                    "target": demand.target,
                    "original_value": str(demand.value),
                    "original_unit": self.demand_unit,
                    "provenance": "derived-from-sndlib-demand",
                }
                for demand in self.demands
            ],
        }


@dataclass(frozen=True)
class SndlibRecord:
    network_id: str
    node_count: int
    link_count: int
    demand_count: int
    coordinate_count: int
    demand_unit: str | None
    capacity_unit: str | None
    parser_status: str
    error: str


def _parse_link(element: ET.Element, path: Path) -> SndlibLink:
    return SndlibLink(
        link_id=str(element.attrib["id"]),
        source=_text(element, "source", path),
        target=_text(element, "target", path),
    )


def _parse_demand(element: ET.Element, path: Path) -> SndlibDemand:
    try:
        value = Decimal(_text(element, "demandValue", path))
    except InvalidOperation as error:
        raise ValueError(f"{path}: invalid demandValue") from error
    return SndlibDemand(
        demand_id=str(element.attrib["id"]),
        source=_text(element, "source", path),
        target=_text(element, "target", path),
        value=value,
    )


def parse_sndlib_xml(path: Path) -> SndlibNetwork:
    root = ET.parse(path).getroot()
    nodes = tuple(
        sorted(
            (
                SndlibNode(
                    node_id=str(item.attrib["id"]),
                    x=Decimal(_text(item, "x", path)),
                    y=Decimal(_text(item, "y", path)),
                )
                for item in _children(root, "node")
            ),
            key=lambda item: item.node_id,
        )
    )
    links = tuple(
        sorted((_parse_link(item, path) for item in _children(root, "link")), key=lambda item: item.link_id)
    )
    demands = tuple(
        sorted(
            (_parse_demand(item, path) for item in _children(root, "demand")),
            key=lambda item: item.demand_id,
        )
    )
    node_ids = {node.node_id for node in nodes}
    for item in (*links, *demands):
        if item.source not in node_ids or item.target not in node_ids:
            raise ValueError(f"{path}: dangling endpoint in {item}")
    unit = next((item.text.strip() for item in _children(root, "unit") if item.text), None)
    return SndlibNetwork(
        network_id=path.stem,
        nodes=nodes,
        links=links,
        demands=demands,
        demand_unit=unit,
        capacity_unit=None,
    )


def inventory_sndlib(path: Path) -> SndlibRecord:
    try:
        network = parse_sndlib_xml(path)
    except (ET.ParseError, KeyError, InvalidOperation, ValueError) as error:
        return SndlibRecord(path.stem, 0, 0, 0, 0, None, None, "invalid", str(error))
    return SndlibRecord(
        network_id=network.network_id,
        node_count=len(network.nodes),
        link_count=len(network.links),
        demand_count=len(network.demands),
        coordinate_count=sum(node.x is not None and node.y is not None for node in network.nodes),
        demand_unit=network.demand_unit,
        capacity_unit=network.capacity_unit,
        parser_status="valid",
        error="",
    )
