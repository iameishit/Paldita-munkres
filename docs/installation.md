# Installation

```bash
pip install munkres            # once 2.0 is on PyPI
pip install munkres==2.0.0rc1 # the release candidate
pip install git+https://github.com/iameishit/Paldita-munkres   # straight from GitHub
```

Requires Python 3.10 or newer. There are no runtime dependencies. numpy and pandas are optional:
install them only if you want to pass arrays or DataFrames.

Check the install:

```bash
python -m munkres      # runs a built-in self-check
munkres --version
```

## From source

```bash
git clone https://github.com/iameishit/Paldita-munkres
cd Paldita-munkres
pip install -e .
```
