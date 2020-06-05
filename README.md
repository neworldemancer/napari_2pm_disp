# napari_2pm_disp
Display utility for fast visualiation after 2PM imaging (works with LaVision TrimScope OEM-TIFF).


# Setup
1. Install anaconda
2. Setup conda environment with the included definition: 
* Open anaconda prompt
* Call ```conda env create -f environment.yml``` to create environment
3. Configure path to Anaconda in `anaconda_path_cfg.bat`
4. Verify installation works by running `displaydataset.bat`
5. For fast display of last acquired dataset configure `__DS_ROOT` variable in `displaydataset.py`
6. Place shortcuts to `displaydataset.bat` and `displaylast.bat` in a user-friendly place.
