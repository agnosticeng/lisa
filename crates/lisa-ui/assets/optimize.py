"""Losslessly recompress exported PNG/ICNS assets using the Python standard library."""

from pathlib import Path
import struct
import zlib


def chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data))
    )


def pixel_rows(scanlines, width, height, channels):
    """Undo PNG's reversible row filters without color conversion."""
    stride = width * channels
    if len(scanlines) != height * (stride + 1):
        raise ValueError("Invalid PNG scanline size")
    previous = bytearray(stride)
    rows = []
    for offset in range(0, len(scanlines), stride + 1):
        kind = scanlines[offset]
        row = bytearray(scanlines[offset + 1 : offset + stride + 1])
        if kind not in range(5):
            raise ValueError("Invalid PNG filter")
        for i in range(stride):
            left = row[i - channels] if i >= channels else 0
            above = previous[i]
            corner = previous[i - channels] if i >= channels else 0
            if kind == 1:
                predictor = left
            elif kind == 2:
                predictor = above
            elif kind == 3:
                predictor = (left + above) // 2
            elif kind == 4:
                p = left + above - corner
                a, b, c = abs(p - left), abs(p - above), abs(p - corner)
                predictor = left if a <= b and a <= c else above if b <= c else corner
            else:
                predictor = 0
            row[i] = (row[i] + predictor) & 255
        rows.append(bytes(row))
        previous = row
    return rows


def compact_scanlines(original, header):
    compressed = zlib.compress(original, 9)
    width, height, bits, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", header
    )
    if bits != 8 or color not in (2, 6) or compression or filtering or interlace:
        return compressed
    channels = 4 if color == 6 else 3
    rows = pixel_rows(original, width, height, channels)
    previous = bytes(width * channels)
    up, sub = bytearray(), bytearray()
    for row in rows:
        up.append(2)
        up.extend((a - b) & 255 for a, b in zip(row, previous))
        sub.append(1)
        sub.extend(row[:channels])
        sub.extend((a - b) & 255 for a, b in zip(row[channels:], row[:-channels]))
        previous = row
    for scanlines in (up, sub):
        candidate = zlib.compress(scanlines, 9)
        if len(candidate) < len(compressed):
            if pixel_rows(zlib.decompress(candidate), width, height, channels) != rows:
                raise ValueError("PNG refiltering changed pixels")
            compressed = candidate
    return compressed


def png(data):
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data
    chunks = []
    offset = 8
    while offset < len(data):
        size, kind = struct.unpack_from(">I4s", data, offset)
        payload = data[offset + 8 : offset + 8 + size]
        checksum = struct.unpack_from(">I", data, offset + 8 + size)[0]
        if checksum != zlib.crc32(kind + payload):
            raise ValueError("Invalid PNG checksum")
        chunks.append((kind, payload))
        offset += size + 12
    original = b"".join(payload for kind, payload in chunks if kind == b"IDAT")
    # Keep exact pixels, color profiles and metadata. Try reversible filters
    # suited to the flat icon artwork before selecting the smallest encoding.
    header = next(payload for kind, payload in chunks if kind == b"IHDR")
    compressed = compact_scanlines(zlib.decompress(original), header)
    if len(compressed) >= len(original):
        return data
    output = bytearray(data[:8])
    written = False
    for kind, payload in chunks:
        if kind == b"IDAT":
            if written:
                continue
            payload = compressed
            written = True
        output.extend(chunk(kind, payload))
    return bytes(output)


def icns(data):
    magic, size = struct.unpack_from(">4sI", data)
    if magic != b"icns" or size != len(data):
        raise ValueError("Invalid ICNS header")
    output = bytearray()
    offset = 8
    while offset < len(data):
        kind, size = struct.unpack_from(">4sI", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError("Invalid ICNS element")
        payload = png(data[offset + 8 : offset + size])
        output.extend(struct.pack(">4sI", kind, len(payload) + 8) + payload)
        offset += size
    return struct.pack(">4sI", b"icns", len(output) + 8) + output


if __name__ == "__main__":
    assets = Path(__file__).resolve().parent
    for name, encode in [("Lisa.png", png), ("Lisa.icns", icns)]:
        path = assets / name
        original = path.read_bytes()
        compact = encode(original)
        if len(compact) < len(original):
            path.write_bytes(compact)
        print(f"{name}: {len(original)} -> {min(len(original), len(compact))} bytes")
