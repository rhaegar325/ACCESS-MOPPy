Changelog
=========

This CHANGELOG documents only key changes between versions. For a full description
of all changes see https://github.com/ACCESS-NRI/ACCESS-MOPPy/releases

moppy-v1.1.0b (2026-04-24)
---------------------------

**Bug Fixes & Extended Variable Support**

* Numerous bug fixes across atmosphere, ocean, sea-ice, and land components
* Extended variable support: ``tran``, ``hfgeou``, ``msftbarot``, ``sftof``, ``zfull``,
  ``landCoverFrac``, ``tsl``, ``gpp``, ``cl``, ``siconc``, ``hfds``, ``zg``, ``so``,
  ``sos``, ``tasmax``, ``tasmin``, and more
* Add support for CMIP6, CMIP6Plus, and CMIP7 controlled vocabularies simultaneously
* ILAMB workflow: batch processing and softlink generator for evaluation of historical
  runs (see documentation and ``Tutorial_CMORise_ILAMB_Variables.ipynb``)
* Re-enable Python 3.13 support
* Improved unit test coverage for derivation modules
* Documentation improvements

moppy-v1.0.0 (2025-10-27)
--------------------------

**Major Rebranding Release**

* **BREAKING CHANGE**: Rebranded from ACCESS-MOPPeR to access_moppy
* **Package name**: Changed from ``access_mopper`` to ``access_moppy``
* **New versioning**: Reset to v1.0.0 with new tag prefix ``moppy-v``
* **Import changes**: All imports now use ``from access_moppy import ...``
* **Installation**: Now install via ``pip install access_moppy``

This release marks the official rebranding of the package while maintaining
all existing functionality. Please update your imports and installation
commands accordingly.
