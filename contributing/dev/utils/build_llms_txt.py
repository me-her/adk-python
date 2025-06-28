#!/usr/bin/env python3
"""
build_llms_txt.py – produce llms.txt and llms-full.txt
                   – skips ```java``` blocks
                   – README can be next to docs/ or inside docs/
"""
from __future__ import annotations
import argparse, re, sys, textwrap
from pathlib import Path
from typing import List, Tuple

RE_JAVA = re.compile(r"```java[ \t\r\n][\s\S]*?```", re.I | re.M)

def strip_java(md: str) -> str:
    return RE_JAVA.sub("", md)

def first_heading(md: str) -> str | None:
    for line in md.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None

def md_to_text(md: str) -> str:
    import markdown, bs4
    html = markdown.markdown(md, extensions=["fenced_code", "tables", "attr_list"])
    return bs4.BeautifulSoup(html, "html.parser").get_text("\n")

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding(model).encode(text))
    except Exception:
        return len(text.split())

# ---------- index (llms.txt) ----------
def build_index(docs: Path) -> str:
    # Locate README
    for cand in (docs / "README.md", docs.parent / "README.md"):
        if cand.exists():
            readme = cand.read_text(encoding="utf-8")
            break
    else:
        sys.exit("README.md not found in docs/ or its parent")

    title = first_heading(readme) or "Documentation"
    summary = md_to_text(readme).split("\n\n")[0]
    lines = [f"# {title}", "", f"> {summary}", ""]

    primary: List[Tuple[str, str]] = []
    secondary: List[Tuple[str, str]] = []

    for md in sorted(docs.rglob("*.md")):
        rel = md.relative_to(docs)
        url = f"https://google.github.io/adk-docs/{rel}".replace(" ", "%20")
        h = first_heading(strip_java(md.read_text(encoding="utf-8"))) or rel.stem
        (secondary if "sample" in rel.parts or "tutorial" in rel.parts else primary).append((h, url))

    def emit(name: str, items: List[Tuple[str, str]]):
        nonlocal lines
        if items:
            lines.append(f"## {name}")
            lines += [f"- [{h}]({u})" for h, u in items]
            lines.append("")

    emit("Documentation", primary)
    emit("Optional",     secondary)
    return "\n".join(lines)

# ---------- full corpus ----------
def build_full(docs: Path) -> str:
    out = []
    for md in sorted(docs.rglob("*.md")):
        out.append(strip_java(md.read_text(encoding="utf-8")))
    return "\n\n".join(out)

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate llms.txt / llms-full.txt",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", required=True, type=Path)
    ap.add_argument("--out-root", default=Path("."), type=Path)
    ap.add_argument("--index-limit", type=int, default=50_000)
    ap.add_argument("--full-limit",  type=int, default=500_000)
    args = ap.parse_args()

    idx, full = build_index(args.docs_dir), build_full(args.docs_dir)
    if (tok := count_tokens(idx))  > args.index_limit: sys.exit(f"Index too big: {tok:,}")
    if (tok := count_tokens(full)) > args.full_limit:  sys.exit(f"Full text too big: {tok:,}")

    (args.out_root / "llms.txt").write_text(idx,   encoding="utf-8")
    (args.out_root / "llms-full.txt").write_text(full, encoding="utf-8")
    print("✅ Generated llms.txt and llms-full.txt successfully")
    print(f"llms.txt tokens: {count_tokens(idx)}")
    print(f"llms-full.txt tokens: {count_tokens(full)}")

if __name__ == "__main__":
    main()
