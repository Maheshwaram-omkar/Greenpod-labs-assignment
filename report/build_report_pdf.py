"""Render the editable assignment report as a compact multi-page PDF."""
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def render(source: Path, destination: Path):
    lines = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            line = line.lstrip("# ").upper() if line.startswith("# ") else line.lstrip("# ")
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", line)
        line = line.replace("`", "")
        if line.startswith("|---") or line.startswith("| ---"):
            continue
        if line.startswith("|"):
            line = "  " + "   |   ".join(cell.strip() for cell in line.strip("|").split("|"))
        chunks = textwrap.wrap(line, width=98, subsequent_indent="  ") if line else [""]
        lines.extend(chunks)
    destination.parent.mkdir(parents=True, exist_ok=True)
    page_lines = 47
    with PdfPages(destination) as pdf:
        for start in range(0, len(lines), page_lines):
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            ax = fig.add_axes([0.09, 0.07, 0.82, 0.86]); ax.axis("off")
            ax.text(0, 1, "\n".join(lines[start:start + page_lines]), va="top", ha="left",
                    fontsize=9.2, family="DejaVu Sans", linespacing=1.35, color="#202820")
            ax.text(1, -0.035, f"Greenpod Labs | Assignment 2 | {start // page_lines + 1}",
                    ha="right", va="top", fontsize=7.5, color="#66705f", transform=ax.transAxes)
            pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    render(base / "Assignment2_Avocado_Report.md", base / "Assignment2_Avocado_Report.pdf")
