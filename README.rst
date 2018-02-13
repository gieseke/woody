woody
=====

A Python library for constructing very large random forests. The basic idea is to use "top trees" built for a small random subset of the data and to use these top trees to distribute all the training instances to the top trees' leaves. For each leaf, one or more bottom trees are built. For the bottom trees, woody resorts to pure C code that follows the random forest construction scheme provided by the `Scikit-Learn <http://scikit-learn.org/stable/>`_.

Dependencies
------------

The woody package is tested under Python 2.7. See the requirements.txt for the packages that need to be installed.

Further, `Swig <http://www.swig.org>`_, `setuptools <https://pypi.python.org/pypi/setuptools>`_, and a working C/C++ compiler need to be available. 

Quickstart
----------

To install the package from the sources, first get the current development release via::

  git clone git@gitlab.com:fgieseke/woody.git

Afterwards, install a virtual environment via virtualenv. Go to the root of the woody package and type::

    mkdir .venv
    cd .venv
    virtualenv woody
    source woody/bin/activate
    cd ..
    pip install -r requirements

Next, you can install the package locally (development) via::

  python setup.py clean
  python setup.py develop

To run all the experiments, you also need to manually install::

  git clone https://github.com/tgsmith61591/skutil
  cd skutil
  python setup.py install

Experiments
-----------

To run the experiments, simply run the launch.py file in the corresponding subdirectory. The associated run files will automatically download the datasets needed (in case this phase is interrupted, please delete the incomplete data files in the corresponding directory under woody/data). For instance::

  cd experiments/small_data
  python launch.py 

Disclaimer
----------

The source code is published under the GNU General Public License (GPLv3). The authors are not responsible for any implications that stem from the use of this software.

