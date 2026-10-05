"""Source integrity checks. This does not rerun training or validate model quality."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "ai/notebooks/banking77.ipynb").read_text())
assert notebook["nbformat"] == 4
code_cells = 0
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    code_cells += 1
    source = "".join(cell["source"])
    # Colab magics are supported in the notebook, not parsed as Python.
    if not any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
        ast.parse(source)
    assert not any(out.get("output_type") == "error" for out in cell.get("outputs", []))
labels = json.loads((ROOT / "web/backend/label_mapping.json").read_text())["id2label"]
assert len(labels) == 77 and set(labels) == {str(i) for i in range(77)}
metrics = json.loads((ROOT / "ai/results/metrics.json").read_text())
assert metrics["primary"]["rows"] == 3080
assert 0 <= metrics["primary"]["accuracy"] <= 1
for path in (ROOT / "web/backend").glob("*.py"):
    ast.parse(path.read_text())
print(f"Notebook: {code_cells} code cells; 77 labels; saved metrics and API syntax checked.")
