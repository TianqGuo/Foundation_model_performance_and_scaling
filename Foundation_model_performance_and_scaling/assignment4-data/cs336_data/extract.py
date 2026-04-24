from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    tree = HTMLTree.parse_from_bytes(html_bytes)
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