from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from resiliparse.parse.html import HTMLTree


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    # Decode to Unicode — try UTF-8 first, fall back to detected encoding
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        enc = detect_encoding(html_bytes) or "latin-1"
        html_str = html_bytes.decode(enc, errors="replace")

    tree = HTMLTree.parse(html_str)
    if tree is None:
        return None
    return extract_plain_text(
        tree,
        preserve_formatting=True,
        main_content=False,
        list_bullets=True,
        alt_texts=False,
        links=False,
    )
