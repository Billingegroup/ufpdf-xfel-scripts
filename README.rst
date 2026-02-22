|Icon| |title|_
===============

.. |title| replace:: bglk-euxfel
.. _title: https://billingegroup.github.io/bglk-euxfel

.. |Icon| image:: https://avatars.githubusercontent.com/billingegroup
        :target: https://billingegroup.github.io/bglk-euxfel
        :height: 100px

|PyPI| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/billingegroup/bglk-euxfel/actions/workflows/matrix-and-codecov-on-merge-to-main.yml/badge.svg
        :target: https://github.com/billingegroup/bglk-euxfel/actions/workflows/matrix-and-codecov-on-merge-to-main.yml

.. |Codecov| image:: https://codecov.io/gh/billingegroup/bglk-euxfel/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/billingegroup/bglk-euxfel

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/bglk-euxfel
        :target: https://anaconda.org/conda-forge/bglk-euxfel

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff
        :target: https://github.com/billingegroup/bglk-euxfel/pulls

.. |PyPI| image:: https://img.shields.io/pypi/v/bglk-euxfel
        :target: https://pypi.org/project/bglk-euxfel/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/bglk-euxfel
        :target: https://pypi.org/project/bglk-euxfel/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/billingegroup/bglk-euxfel/issues

Scripts and code for XFEL experiments to do ultrafast PDF measurements

Analysis scripts for assessing ultrafast pump probe powder diffraction and PDF measurements.

Code for assessing figures of merit from ultrafast pump-probed data from the EuXFEL and LCLS.  This was first written
for the experiment in March 2025 (EuXFEL) and February 2026 (LCLS)

We are working on a cli but there are also Jupyter scripts in `src/bglk_lcls/morph_LCLS_pyfai.ipynb` `src/bglk_euxfel/scripts`
Functions are in `src/bglk_euxfel/functions.py` and plotters in `src/bglk_euxfel/plotters.py`

to use the CLI please use the installation instructions below.

To use the Jupyter notebooks, please use the instructions in the `Getting Started` section

For more information about the `bglk_euxfel` library, please consult our `online documentation <https://billingegroup.github.io/bglk_euxfel>`_.

For more information about the bglk-euxfel library, please consult our `online documentation <https://billingegroup.github.io/bglk-euxfel>`_.

Citation
--------

If you use bglk-euxfel in a scientific publication, we would like you to cite this package as

        bglk-euxfel Package, https://github.com/billingegroup/bglk-euxfel

Installation
------------

The preferred method is to use `Miniconda Python
<https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_
and install from the "conda-forge" channel of Conda packages.

To add "conda-forge" to the conda channels, run the following in a terminal. ::

        conda config --add channels conda-forge

We want to install our packages in a suitable conda environment.
The following creates and activates a new environment named ``bglk-euxfel_env`` ::

        conda create -n bglk-euxfel_env bglk-euxfel python=3.13
        conda activate bglk-euxfel_env

The output should print the latest version displayed on the badges above.

If the above does not work, you can use ``pip`` to download and install the latest release from
`Python Package Index <https://pypi.python.org>`_.
To install using ``pip`` into your ``bglk-euxfel_env`` environment, type ::

        pip install bglk-euxfel

If you prefer to install from sources, after installing the dependencies, obtain the source archive from
`GitHub <https://github.com/billingegroup/bglk-euxfel/>`_. Once installed, ``cd`` into your ``bglk-euxfel`` directory
and run the following ::

        pip install .

This package also provides command-line utilities. To check the software has been installed correctly, type ::

        bglk-euxfel --version

You can also type the following command to verify the installation. ::

        python -c "import bglk_euxfel; print(bglk_euxfel.__version__)"

To view the basic usage and available commands, type ::

        bglk-euxfel -h

Getting Started
---------------

You may consult our `online documentation <https://billingegroup.github.io/bglk-euxfel>`_ for tutorials and API references.

## Installation

to use the CLI please use the installation instructions above.

To use the Jupyter notebooks, please use the following instructions:

We assume that you have miniconda or conda installed

One time setup:

1. clone the repository or download and unzip the code.
2. move to the top level folder of the repository. It will be called `bglk_euxfel` and contain the `pyproject.toml` file
3. create a python 3.13 conda environment that you will work in, e.g.,::

        conda create -n bglk_lcls python=3.13

4. activate the environment::

        conda activate bglk_lcls

5. install all the dependencies::

        conda install --file requirements/conda.txt
        pip install .

6. You also need to pip install PDFGetX from the .whl that you can find in `mfxl1044925/scratch/PDFGetX`, this requires
first setting up a symbolic link from your home directory to the proposal data directory in the LCLS S3DF:
    1. Create a working folder in your S3DF home directory using the New Folder on the homepage and name it something
    distinguishable (e.g., /mfx_ions/)
    2. Move into that directory::

        cd /mfx_ions/

    3. Create a symbolic link to the proposal folder, you will change the rest of the second directory in this command,
    but keep /mfxl1044925 at the end::

        ln -s /sdf/data/lcls/ds/mfx/mfxl1044925 /sdf/home/l/lkitsu/mfx_ions/mfxl1044925

7. if you will be uploading any code edits and making push requests to the repository install and do some more things::

        conda install pre-commit
        pre-commit install

8. Make the `bgkl_lcls` environment to be available as a kernel in jupyter::

        python -m ipykernel install --user --name=bgkl_lcls

For LCLS:

In order to set up the environment in your S3DF account follow these steps:
1.  Open an interactive terminal session: S3DF Main Page -> Open in Terminal (drop-down, not just the open terminal
    button directory) -> (select) PSAna Interactive -> There will be some security-related prompt, just reply ‘yes'
2.  In the PSAna interactive session, create the conda environment with required packages following the procedure
    explained above

To run the code in S3DF:

1. log in to your computer and start a Jupyter notebook
2. open the jupyter notebook under bgkl_lcls
3. edit any user-settable parameters in the second cell, such as the run number you want to work on, the q-range you
want to use for the normalization and the q-range you want to compute the figure of merit over.
4. Change the path to the input data folder as well as check the other paths in case yours are different.
4. run the notebook. The safest way to do it is using the double-chevron that restarts the kernel and runs all the cels
5. Good luck!


For EuXFEL:

Testing your installation by running the example in this repo

1. look for the `example` directory in the `doc` directory. You should see a folder called `input_data`
2. make a copy of the `figure_of_merit_assessment.ipynb` and paste it into the `example` directory.
3. in your terminal move to that same directory, e.g., `cd doc/example`
4. with your `bglk_euxfel` environment activated type `jupyter notebook &`
5. Jupyter should open in your browser
6. navigate to the `figure_of_merit_assessment.ipynb` and double-click on it to open it
7. make sure that the kernel selected is `bglk_euxfel` by selecting it in the dropdown menu
8. adjust the user-defined entries the second cell appropriately and run the notebook. You can do this by running each cell separately, but better is
9. it should run and give plots that look nice without crashing
10. after doing this, you can safely delete that ipynb

Downloading the data from Maxwell cluster to do a full analysis:

1. log in to [Jupyter Hub](https://max-jhub.desy.de) on the Maxwell cluster using your DESY credentials and one time authentication factor. See this [link](https://it.desy.de/services/mfa/external_people/index_eng.html) for instructions
2. choose suitable partition on Maxwell from the upper drop-down menu and spawn your job. You will automatically have a folder named GPFS
3. navigate to `gpfs/exfel/exp/\<beamline>/\<cycle>/\<proposal_number>/usr/\<code_name>`, for example `gpfs/exfel/exp/FXE/202501/p008015/usr/data_for_morph`. Here you will find a ipynb file \<file_name.ipynb> and a folder \<folder_name>, for example `get_data_for_morphing.ipynb` and folder name 'save_raw_integrated_Data'. The folder is where all the azimuthal integrated 2D detector images, geometry and mask corrected, will be saved for each run. This includes the raw scattering intensities for each delay scan, sorted between laser on and laser off, scattering vector q, and delay scan motor positions
4. open the ipynb
5. edit the parameters in the second cell, see example below

```
proposal = 8015     # number of the proposal
max_run = 190       # the code will try and process all scans from any numbered zero up to max_run
overwrite = False   # set to true if you want to reanalyze previously analyzed data, otherwise set to False for faster response
```

6. run the notebook
7. Download the folder with the data from the EuXFEL cluster and copy it to a folder named 'input_data'. This folder must reside in the same folder as the ipynb for the morphing `figure_of_merit_assessment.ipynb`

To run the code:

1. log in to your computer and start a terminal
2. activate the conda environment `conda activate bglk_euxfel`
3. move to the directory with the ipynb in it that you want to work on
4. start jupyter `jupyter lab`
5. open the ipynb
6. edit any user-settable parameters in the second cell, such as the run number you want to work on, the q-range you want to use for the normalization and the q-range you want to compute the figure of merit over.
7. run the notebook. The safest way to do it is using the double-chevron that restarts the kernel and runs all the cels
8. Good luck!

Support and Contribute
----------------------

`Diffpy user group <https://groups.google.com/g/diffpy-users>`_ is the discussion forum for general questions and discussions about the use of bglk_euxfel. Please join the bglk_euxfel users community by joining the Google group. The bglk_euxfel project welcomes your expertise and enthusiasm!

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/billingegroup/bglk_euxfel/issues>`_ and/or `submit a fix as a PR <https://github.com/billingegroup/bglk_euxfel/pulls>`_. You can also post it to the `Diffpy user group <https://groups.google.com/g/diffpy-users>`_.

Feel free to fork the project and contribute. To install bglk_euxfel
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/billingegroup/bglk_euxfel/blob/main/CODE_OF_CONDUCT.rst>`_.

Contact
-------

For more information on bglk_euxfel please visit the project `web-page <https://billingegroup.github.io/>`_ or email Prof. Simon J. L. Billinge at sb2896@columbia.edu.


Support and Contribute
----------------------

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/billingegroup/bglk-euxfel/issues>`_ and/or `submit a fix as a PR <https://github.com/billingegroup/bglk-euxfel/pulls>`_.

Feel free to fork the project and contribute. To install bglk-euxfel
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/billingegroup/bglk-euxfel/blob/main/CODE-OF-CONDUCT.rst>`_.

Contact
-------

For more information on bglk-euxfel please visit the project `web-page <https://billingegroup.github.io/>`_ or email the maintainers ``Luis Kitsu(luki2322@colorado.edu) and Simon Billinge(sbillinge@ucsb.edu)``.

Acknowledgements
----------------

``bglk-euxfel`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.
