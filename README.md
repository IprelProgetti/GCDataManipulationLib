# GCDataManipulationLib
Data manipulation library to easily move from/to narrow/wide table representations. 

**Current version:** gcornilib v1.0.1

**What's new?**

* multi input and multi output column selection
* configurable training and predicting time steps
* support for multi-variable and multi-time re-scale to the original variable range
* automatic label encoder/decoder

## Usage

### Install

Create your python virtual environment, activate it, then type:

```
git clone https://github.com/horns-g/GCDataManipulationLib.git
cd GCDataManipulationLib
python setup.py install
```

### Use

Import gcornilib from within your python file and use it:

```
from gcornilib.DataManipulation import MLPrePostProcessing as dm2
```

### Run example

```
cd gcornilib/Examples
python lstm.py | main.py
```