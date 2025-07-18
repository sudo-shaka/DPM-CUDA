#!/bin/bash

#if this build failed, and issue I ran into was the setup.py was not including the include path. to fix you can put the includes in .venv/include/ "cp ./include/* ./venv/bin/include/"

python3 -m venv .venv
source .venv/bin/activate

pip install setuptools
pip install torch 
pip install torchvision
pip install progressbar2
pip install matplotlib
pip install pandas
pip install imageio

python setup.py build_ext --inplace || (echo "Build failed." && exit);

f=$(python test3D.py | awk '{print $1}')
python plottext.py $f 2 output/
