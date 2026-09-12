from pathlib import Path

from PIL import Image


def write_dji_image(path: Path, size=(40, 30), camera="FC220", latitude=(29.0, 56.0, 36.0), longitude=(85.0, 24.0, 0.0), fields=None):
    """JPEG with EXIF GPS (north, west) and DJI XMP attributes; latitude=None leaves out the GPS block."""
    exif = Image.Exif()
    exif[272] = camera
    if latitude is not None:
        exif[0x8825] = {1: "N", 2: latitude, 3: "W", 4: longitude}
    attributes = " ".join(f'drone-dji:{key}="{value}"' for key, value in (fields or {}).items())
    xmp = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" {attributes}/></rdf:RDF></x:xmpmeta>'
    ).encode("utf-8")
    Image.new("RGB", size, "white").save(path, exif=exif, xmp=xmp)
    return path
