import numpy as np
import jedi
import pandas as pd


def foo(x, y=0):
    """Test docstring"""
    return x


print(jedi.Script('import numpy').completions())
print(jedi.Script('import pandas').completions())
print(jedi.Script('import cv2').completions())
