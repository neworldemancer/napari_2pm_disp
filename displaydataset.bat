@echo off
call C:\Anaconda3\Scripts\activate.bat C:\Anaconda3

call conda activate napari_disp
set spath=%~dp0
pushd %spath%
python  %spath%/displaydataset.py 

popd