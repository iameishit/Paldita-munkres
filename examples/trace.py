"""Watch the algorithm work, and save the steps as a web page."""

import pathlib
import tempfile

from munkres import solve

trace = solve([[4, 1, 3], [2, 0, 5], [3, 2, 2]], trace=True).trace
print(trace.to_text())
page = pathlib.Path(tempfile.gettempdir()) / "munkres-trace.html"
page.write_text(trace.to_html(), encoding="utf-8")
print("HTML written to", page)
