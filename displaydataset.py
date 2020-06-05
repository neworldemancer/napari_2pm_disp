#!/usr/bin/env python
# coding: utf-8

import skimage.io
import numpy as np
import napari
import glob
import os
from time import sleep
from aicsimageio import AICSImage
import wx
import argparse
    
__DS_ROOT = r'd:\TrimScope_Data\*'

def get_tiles_files_and_ofs(ds):
    metadata = ds.metadata
    if isinstance(metadata, str):
        has_tiles = False
    else:
        tiles_sections = metadata.to_xml().split('TileConfiguration TileConfiguration="')
        has_tiles = True
        if len(tiles_sections)==1:
            has_tiles = False

    if has_tiles:
        tiles_section = tiles_sections[1].split('"')
        if len(tiles_section) == 0:
            has_tiles = False

    if has_tiles:
        tile_desc = tiles_section[0].split('  ')
        if len(tile_desc) <2:
            has_tiles = False
        else:
            tile_desc = tile_desc[1:]

    if has_tiles:
        tile_files_all = []
        tile_ofs_all = []
        for td in tile_desc:
            fname, smth, ofs = td.split(';')
            tile_files_all.append(fname)
            
            ofs = [float(v.strip()) for v in ofs.strip('()').split(',')]
            
            tile_ofs_all.append(ofs)

        #for fname, ofs in zip(tile_files_all, tile_ofs_all):
        #    print(fname, ofs)

        n_tile_max=100
        n_x = 1
        n_y = 1
        str_tmpl = '[%02d x %02d]'
        for nt in range(n_tile_max):
            str_x = str_tmpl % (0, nt)
            str_y = str_tmpl % (nt, 0)

            x_in_map = [str_x in fname for fname in tile_files_all]
            y_in_map = [str_y in fname for fname in tile_files_all]

            any_x = np.any(x_in_map)
            any_y = np.any(y_in_map)

            if any_x:
                n_x = nt+1
            if any_y:
                n_y = nt+1
            if not (any_x or any_y):
                break

        #print(f'number of tiles: {n_x} x {n_y}')

        tile_files = []
        tile_ofs = []
        for iy in range(n_y):
            for ix in range(n_x):
                str_yx = str_tmpl % (iy, ix)
                tile_idxs = [idx for idx, fname in enumerate(tile_files_all) if str_yx in fname]
                fnames = [tile_files_all[idx] for idx in tile_idxs]
                ofs = tile_ofs_all[tile_idxs[0]]
                if len(ofs)==4:
                    ofs = (ofs[0], ofs[1], ofs[3])
                tile_files.append(fnames)
                tile_ofs.append(ofs)
        #for fname, ofs in zip(tile_files, tile_ofs):
        #    print(fname, ofs)

    if not has_tiles:
        #print('no tiles')
        return [1,1], [],[]
    else:
        return [n_x, n_y], tile_files, tile_ofs
        
def read_dataset_from_seed_file(file):
    im = skimage.io.imread(file)
    
    sh = im.shape
    
    dims_dict = {3: 'ZYX',
                 4: 'CZYX',
                 5: 'TCZYX'    
                }
    dims = dims_dict.get(len(sh), 'ZYX')
    
    inf = {}
    data_aics = AICSImage(file, known_dims=dims)
    inf['res'] = data_aics.get_physical_pixel_size()
    
    tiles_nx_ny, tiles_files, tiles_ofs = get_tiles_files_and_ofs(data_aics)
    # print(tiles_ofs)
    # print(inf, im.shape)
    # print(data_aics.data)
    inf['n_xy_tiles'] = tiles_nx_ny
    if tiles_nx_ny == [1,1]:
        inf['tiles'] = [data_aics.data[0, 0].copy()]  # first position and time
        inf['ofs'] = [(0,0,0)]
    else:
        tiles = []
        prfx = os.path.dirname(os.path.realpath(file))
        for tile_names in tiles_files:
            tile_name = os.path.join(prfx, tile_names[0])
            data_aics = AICSImage(tile_name, known_dims=dims)
            tile_data = data_aics.data[0, 0].copy()
            #data_aics.close()
            #del data_aics
            tiles.append(tile_data)
        inf['tiles'] = tiles
        inf['ofs'] = tiles_ofs
    return inf
    

def get_path(wildcard, latest_dir):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    defaultDir=latest_dir or ''
    dialog = wx.FileDialog(parent=None,
                           message='Select any file from the dataset you want to display:',
                           wildcard=wildcard,
                           defaultDir=defaultDir,
                           style=style)
                           
    print(defaultDir)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

def get_tif_path(latest_dir=None):
    p = get_path('OME-TIFF files (*.tif;*.tiff)|*.tif;*.tiff', latest_dir)
    return p

    
def show_ds(ds_inf):
    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=3)

        chn = ['blue', 'green', 'red', 'far red']
        cols = ['blue', 'green', 'red', 'gray']

        scale=ds_inf['res'][::-1]

        n_tile = len(ds_inf['tiles'])
        for tile_idx, (tile,ofs) in enumerate(zip(ds_inf['tiles'], ds_inf['ofs'])):
            print(f'loading tile {tile_idx}')
            ofs = ofs[::-1]
            #print(ofs, scale)
            for ch, im in enumerate(tile):
                if ch>=4:
                    continue
                chname = chn[ch]
                
                h, b = np.histogram(im.flatten(), 100)
                peak = b[np.argmax(h)]
                clim = [peak, np.round(im.max() * 0.35)]
                # add im as a layer
                viewer.add_image(im,
                                 name=chname + (('_'+str(tile_idx)) if n_tile>1 else ''),
                                 scale=scale,
                                 contrast_limits=clim,
                                 blending='additive',
                                 colormap=cols[ch],
                                 translate=ofs,
                                 gamma=0.85)
                                 
def load_show_ds():
    list_of_files = glob.glob(__DS_ROOT)
    latest_expepriment = max(list_of_files, key=os.path.getctime)

    latest_file = get_tif_path(latest_expepriment+'\\')
    
    print('loading from file', latest_file)
   
    inf = read_dataset_from_seed_file(latest_file)
    
    show_ds(ds_inf=inf)
    
    
def last_show_ds():
    list_of_files = glob.glob(__DS_ROOT)
    latest_expepriment = max(list_of_files, key=os.path.getctime)
    print('parsing experiment', latest_expepriment)

    list_of_files = glob.glob(latest_expepriment+'\*')
    latest_dataset = max(list_of_files, key=os.path.getctime)
    print('parsing dataset', latest_dataset)


    list_of_files = glob.glob(latest_dataset+'\*.tif')
    latest_file = max(list_of_files, key=os.path.getctime)
    
    print('loading from file', latest_file)


    inf = read_dataset_from_seed_file(latest_file)
    
    show_ds(ds_inf=inf)

def display():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--last", help="show last dataset",
                        action="store_true")
    args = parser.parse_args()
    if args.last:
        last_show_ds()
    else:
        load_show_ds()

if __name__ == "__main__": 
    display()