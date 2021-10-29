#!/usr/bin/env python
# coding: utf-8

import skimage.io
import numpy as np
import napari
import locale
import glob
import os
from time import sleep
from aicsimageio import AICSImage
import wx
import re
import argparse
from PIL import Image
from magicgui import magicgui
import pickle
    
__DS_ROOT = r'd:\TrimScope_Data\*'
_V = None
_D = None
_P = None
_N_tiles = 1
_N_read = 0

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

def try_get_resolution(file, guess_res=(0.7, 0.7, 2)):
    try:
        im = skimage.io.imread(file)
        sh = im.shape
        
        dims_dict = {3: 'ZYX',
                     4: 'CZYX',
                     5: 'TCZYX'    
                    }
        dims = dims_dict.get(len(sh), 'ZYX')
        
        data_aics = AICSImage(file, known_dims=dims)
        resolution = data_aics.get_physical_pixel_size()
        
        del im
        del data_aics
    except:
        resolution = guess_res
    return resolution
    
def read_dataset_from_seed_file(file, load_time):
    im = skimage.io.imread(file)
    title = os.path.basename(os.path.dirname(file))
    
    sh = im.shape
    
    dims_dict = {3: 'ZYX',
                 4: 'CZYX',
                 5: 'TCZYX'    
                }
    dims = dims_dict.get(len(sh), 'ZYX')
    
    inf = {}
    data_aics = AICSImage(file, known_dims=dims)
    inf['res'] = data_aics.get_physical_pixel_size()
    #print(inf['res'])
    tiles_nx_ny, tiles_files, tiles_ofs = get_tiles_files_and_ofs(data_aics)
    # print(tiles_ofs)
    # print(inf, im.shape)
    # print(data_aics.data.shape)
    inf['n_xy_tiles'] = tiles_nx_ny
    inf['title'] = title
    
    if tiles_nx_ny == [1,1]:
        
        if load_time:
            tile_data = data_aics.data[0, :].transpose(1, 0, 2, 3, 4).copy()  # first position
        else:
            tile_data = data_aics.data[0, 0].copy()  # first position and time
        inf['tiles'] = [tile_data]  # first position and time
        inf['ofs'] = [(0,0,0)]
    else:
        tiles = []
        prfx = os.path.dirname(os.path.realpath(file))
        for tile_names in tiles_files:
            tile_name = os.path.join(prfx, tile_names[0])
            data_aics = AICSImage(tile_name, known_dims=dims)
            
            if load_time:
                tile_data = data_aics.data[0, :].transpose(1, 0, 2, 3, 4).copy()  # first position
            else:
                tile_data = data_aics.data[0, 0].copy()  # first position and time

            tiles.append(tile_data)
        inf['tiles'] = tiles
        inf['ofs'] = tiles_ofs
    return inf
    
    
def get_f_inf(filename):
    re_res = re.findall('(.*)_Doc(.*)_PMT - PMT \[(.*)\] _C(.*)_Time Time(.*)\.ome.tif', filename)
    if len(re_res):
        timestamp, doc_id, color_desc, ch_id, t_id = re_res[0]
        doc_id, ch_id, t_id = int(doc_id), int(ch_id), int(t_id)
        return timestamp, doc_id, color_desc, ch_id, t_id
    return None

def read_mp_tiff(path):
    """
    Args:
        path (str) : path to the images, e.g. `/path/to/stacks/img.png`

    Returns:
        image (np.ndarray): image, DHWC
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)
    
def read_dataset_from_dir(fdir, load_time=True, prev_d=None, guess_res=(0.7, 0.7, 2)):
    global _P
    global _N_read
    _P = fdir
    # list files
    
    title = os.path.basename(fdir)
    p = glob.glob(os.path.join(fdir, '*.ome.tif'))
    
    im = read_mp_tiff(p[0])
    sh = list(im.shape)
    print(sh, im.dtype)
    
    f_infs = []
    
    if prev_d is not None:
        p_fnames = prev_d['fnames']
        p = [pi for pi in p if pi not in p_fnames]
    else:
        _N_read = 0
        
    for f_p in p:
        fn = os.path.basename(f_p)
        f_inf = get_f_inf(fn)
        f_infs.append(f_inf)
        
    if len(f_infs) == 0:
        return prev_d
        
    # find n available t & n_ch
    n_tb = np.min([f_inf[4] for f_inf in f_infs])
    n_te = np.max([f_inf[4] for f_inf in f_infs])
    n_t = n_te-n_tb+1
    #print(n_t)
    if n_t == 0:
        return prev_d
        
    n_c = np.max([f_inf[3] for f_inf in f_infs]) + 1
    #print(n_t, n_c)
    
    # extend:
    if prev_d is None:
        n_t_split = int(np.ceil(n_t/_N_tiles))
        data_all = np.zeros(shape=[n_c, n_t_split * _N_tiles]+sh, dtype=im.dtype)
        
    else:
        n_tot_prev = _N_read
        n_tot_new = n_tot_prev + n_t
        
        data_all_old = prev_d['data_all'][0]
        
        n_t_split_old = data_all_old.shape[1] // _N_tiles
        n_t_split_new = int(np.ceil(n_tot_new/_N_tiles))
        
        delta_n_t = n_t_split_new - n_t_split_old
        
        data = np.zeros(shape=[n_c, delta_n_t * _N_tiles]+sh, dtype=im.dtype)
        data_all = np.concatenate((data_all_old, data), axis=1)
    
    ts0 = f_infs[0][0]
    
    fnames = []
    for f_p, f_inf in zip(p, f_infs):
        timestamp, doc_id, color_desc, ch_id, t_id = f_inf
        assert(timestamp == ts0)
        fnames.append(f_p)
        im = read_mp_tiff(f_p)
        print(f_p)
        data_all[ch_id, _N_read + t_id-n_tb] = im
    
    _N_read += n_t
        
    data_view = data_all.view()
    n_t_tot = data_all.shape[1]
    data_view.shape = ([n_c, n_t_tot//_N_tiles, _N_tiles]+sh)
    
    #print(data.shape)
    
    # R&D on positions ('tiles') separate axis, every n-th of acquired images:
    # data->rename to data_all, and store in dict, make padded to #positions, 
    # keep track of prev true size in d - for concatenation
    # data_view = data_all.view()
    # data_view.shape = ((n_ch, n_time, n_tile, d, h, w)) ## will rise error upon wrong shapes
    # use data_view in place of data as before, re-reshape upon tiles info setting and reloading
    
    #.reshape((1, 2, 2, 61, 512, 512))
    #print(data.shape)
    
    if prev_d is None:
        d = {
             'res': guess_res,
             'n_xy_tiles': [1, 1],
             'tiles': [data_view],
             'data_all': [data_all],
             'ofs': [(0,0,0)],
             'fnames': fnames,
             'title': title
            }
    else:
        #data_p = prev_d['tiles'][0]
        res = prev_d['res']
        title = prev_d['title']
        #data = np.concatenate((data_p, data), axis=1)
        
        p_fnames = prev_d['fnames']
        fnames = p_fnames+fnames
        d = {
             'res': res,
             'n_xy_tiles': [1, 1],
             'tiles': [data_view],
             'data_all': [data_all],
             'ofs': [(0,0,0)],
             'fnames': fnames,
             'title': title
            }
        
    #print(d['res'])
    return d
            

def get_path_open(wildcard, latest_dir):
    app = wx.App(None)
    #loc = wx.Locale(wx.LANGUAGE_ENGLISH_UK)
    
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    defaultDir=latest_dir or ''
    
    dialog = wx.FileDialog(parent=None,
                           message='Select any file from the dataset you want to display:',
                           wildcard=wildcard,
                           defaultDir=defaultDir,
                           style=style)
    
    #print(defaultDir)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

def get_path_save(wildcard, latest_dir):
    app = wx.App(None)
    #loc = wx.Locale(wx.LANGUAGE_ENGLISH_UK)
    
    style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
    defaultDir=latest_dir or ''
    
    dialog = wx.FileDialog(parent=None,
                           message='Select any file from the dataset you want to display:',
                           wildcard=wildcard,
                           defaultDir=defaultDir,
                           style=style)
    
    #print(defaultDir)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

def get_tif_path(latest_dir=None):
    p = get_path_open('OME-TIFF files (*.tif;*.tiff)|*.tif;*.tiff', latest_dir)
    return p

@magicgui(call_button="Tiles")
def set_n_tiles(n_tiles=1):
    global _N_tiles
    
    if n_tiles != _N_tiles and n_tiles>0:
        _N_tiles = n_tiles
        
        data_all_old = _D['data_all'][0][:, :_N_read]
        
        n_c, n_t, *sh = list(data_all_old.shape)
        
        n_t_split_new = int(np.ceil(_N_read/_N_tiles))
        
        n_extra = n_t_split_new*_N_tiles - _N_read
        if n_extra>0:
            data = np.zeros(shape=[n_c, n_extra]+sh, dtype=data_all_old.dtype)
            data_all = np.concatenate((data_all_old, data), axis=1)
        else:
            data_all = data_all_old
            
        data_view = data_all.view()
        n_t_tot = data_all.shape[1]
        data_view.shape = ([n_c, n_t_tot//_N_tiles, _N_tiles]+sh)
        print(data_view.shape, data_all.shape)
        _D['data_all'] = [data_all]
        _D['tiles'] = [data_view]
        
        i = 0
        for tile_idx, (tile,ofs) in enumerate(zip(_D['tiles'], _D['ofs'])):
            print(f'loading tile {tile_idx}')
            ofs = ofs[::-1]
            #print(ofs, tile.shape)
            for ch, im in enumerate(tile):
                if ch>=5:
                    continue
                l = _V.layers[i]
                l.data = im
                l.refresh()
                i += 1
                print('r', i)

@magicgui(call_button="reload")
def reload_data():
    try:
        print('reloading', _V)
        global _D
        print('r ok')
        _D = read_dataset_from_dir(_P, True, _D)
        
        for i, tile in enumerate(_D['tiles']):
            sh = list(tile.shape)
            #print(sh) #[4, 4, 136, 512, 512]
            _D['tiles'][i] = tile
            n_ch = len(tile)
            break
            
        i = 0
        for tile_idx, (tile,ofs) in enumerate(zip(_D['tiles'], _D['ofs'])):
            print(f'loading tile {tile_idx}')
            ofs = ofs[::-1]
            #print(ofs, scale)
            for ch, im in enumerate(tile):
                if ch>=5:
                    continue
                l = _V.layers[i]
                l.data = im
                l.refresh()
                i += 1
                print('r', i)
        return 'ok!'
    except ValueError:
        print('ve')
        return "ValueError!"
        
@magicgui(call_button="SaveParams")
def save_disp_params():
    try:
        limits_dict = {}
        layers = [l for l in _V.layers if isinstance(l, napari.layers.image.image.Image)]
        for ch, l in enumerate(layers):
            if ch>=5:
                continue
            #print(l.name, l.contrast_limits)
            limits_dict[ch] = {
                'contrast_limits': l.contrast_limits,
                'visible': l.visible
            }
        
        fname = get_path_save('DisplayLimitsConfig (*.dlc)|*.dlc', None)
        with open(fname, 'wb') as f:
            pickle.dump(limits_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return 'ok!'
    except ValueError:
        return "ValueError!"
        
@magicgui(call_button="LoadParams")
def load_disp_params():
    try:
        fname = get_path_open('DisplayLimitsConfig (*.dlc)|*.dlc', None)
        with open(fname, 'rb') as f:
            limits_dict = pickle.load(f)
        print(limits_dict)
        
        layers = [l for l in _V.layers if isinstance(l, napari.layers.image.image.Image)]
        for ch, l in enumerate(layers):
            if ch>=5 or ch not in limits_dict:
                continue
            
            params = limits_dict[ch]
            l.contrast_limits = params['contrast_limits']
            l.visible = params['visible']
            
            l.refresh()
        return 'ok!'
    except ValueError:
        return "ValueError!"
        
@magicgui(call_button='ChangeRes', layout='horizontal')
def change_resolution(res_xy=0.7, res_z=2.):
    try:
        res = (res_xy, res_xy, res_z)
        _D['res'] = res
        
        scale = list(res[::-1])
        
        layers = [l for l in _V.layers if isinstance(l, napari.layers.image.image.Image)]
        for ch, l in enumerate(layers):
            l.scale=scale
            
            l.refresh()
        return 'ok!'
    except ValueError:
        return "ValueError!"
        

@magicgui(layout='horizontal', view_option={
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "choices": [("top", 0), ("left", 1), ("front", 2), ("bot", 3), ("right", 4), ("back", 5)],
        },
        auto_call=True
    )
def view_direction(view_option=0):
    try:
        angles={
            0: (0,0,90),
            1: (-90,-90,-90),
            2: (0,0,180),
            3: (0,0,-90),
            4: (-90,90,90),
            5: (-180,0,0)
        }
        a = angles[view_option]
        _V.camera.angles = a
        return 'ok!'
    except ValueError:
        return "ValueError!"
       
def show_ds(ds_inf, reload=False):
    title = ds_inf.get('title', None)
    
    viewer = napari.Viewer(ndisplay=3, title=title)
    
    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True
    viewer.scale_bar.unit = 'um'
	
    #viewer.window.qt_viewer.scale_bar.color= 'black'
    global _V
    _V = viewer
    global _D
    _D = ds_inf
    chn = ['blue', 'green', 'red', 'far red', 'far far red']
    cols = ['blue', 'green', 'red', 'gray', 'magma']

    scale = list(ds_inf['res'][::-1])
    
    #uncomment for true orientation & better scale
    #scale[0] = min(scale[0], 4)
    
    #scale[1] = scale[1]*0.99
    #scale[2] = scale[2]*1.09
    print(scale)

    n_tile = len(ds_inf['tiles'])
    if n_tile>1:
        scale[1] = -scale[1]
        
    n_ch = 0

    for tile_idx, (tile,ofs) in enumerate(zip(ds_inf['tiles'], ds_inf['ofs'])):
        n_ch = len(tile)
        break
    n_ch = min(n_ch, 5)

    peak_map = {}
    clim_map = {}
    for ch in range(n_ch):
        ch_ims = []
        for tile_idx, (tile,ofs) in enumerate(zip(ds_inf['tiles'], ds_inf['ofs'])):
            ch_ims.append(tile[ch])
            
        im = np.array(ch_ims).flatten()
        h, b = np.histogram(im, 100)
        peak = b[np.argmax(h)]
        clim = [peak, np.round(im.max() * 0.35)]
        
        peak_map[ch] = peak
        clim_map[ch] = clim

    for tile_idx, (tile,ofs) in enumerate(zip(ds_inf['tiles'], ds_inf['ofs'])):
        print(f'loading tile {tile_idx}')
        ofs = ofs[::-1]
        #print(ofs, scale)
        for ch, im in enumerate(tile):
            if ch>=5:
                continue
            chname = chn[ch]
            
            #h, b = np.histogram(im.flatten(), 100)
            #peak = b[np.argmax(h)]
            #clim = [peak, np.round(im.max() * 0.35)]
            peak = peak_map[ch]
            clim = clim_map[ch]
            
            # add im as a layer
            ndim = len(im.shape)
            ddim = ndim-3
            tscale = scale if ndim==3 else ([1]*ddim+list(scale)) 
            tofs = ofs if ndim==3 else ([0]*ddim+list(ofs)) 
            viewer.add_image(im,
                             name=chname + (('_'+str(tile_idx)) if n_tile>1 else ''),
                             scale=tscale,
                             contrast_limits=clim,
                             blending='additive',
                             colormap=cols[ch],
                             interpolation='nearest',
                             translate=tofs,
                             gamma=0.85)
                             
    if reload:
        viewer.window.add_dock_widget(reload_data)
        viewer.window.add_dock_widget(set_n_tiles)
        viewer.window.add_dock_widget(change_resolution)
        viewer.window.add_dock_widget(save_disp_params)
        viewer.window.add_dock_widget(load_disp_params)
        viewer.window.add_dock_widget(view_direction)
            

    napari.run()

def load_show_ds():
    list_of_files = glob.glob(__DS_ROOT)
    if list_of_files:
        latest_expepriment = max(list_of_files, key=os.path.getctime)
        start_dir = latest_expepriment+'\\'
    else:
        start_dir = None
    
    latest_file = get_tif_path(start_dir)
    
    print('loading from file', latest_file)
   
    inf = read_dataset_from_seed_file(latest_file, load_time=True)
    #print(inf['tiles'][0].shape)
    show_ds(ds_inf=inf)
    
def load_show_nmi():
    list_of_files = glob.glob(__DS_ROOT)
    if list_of_files:
        latest_expepriment = max(list_of_files, key=os.path.getctime)
        start_dir = latest_expepriment+'\\'
    else:
        start_dir = None

    latest_file = get_tif_path(start_dir)
    
    print('loading from file', latest_file)
    
    guess_res = try_get_resolution(latest_file)
    inf = read_dataset_from_dir(os.path.dirname(latest_file), load_time=True, guess_res=guess_res)
    
    show_ds(ds_inf=inf, reload=True)
    
    
def last_show_ds():
    list_of_files = glob.glob(__DS_ROOT)
    if not list_of_files:
        print(f'Root directory {__DS_ROOT} is empty or not found. Wrong path?')
        return
    latest_expepriment = max(list_of_files, key=os.path.getctime)
    print('parsing experiment', latest_expepriment)

    list_of_files = glob.glob(latest_expepriment+'\*')
    latest_dataset = max(list_of_files, key=os.path.getctime)
    print('parsing dataset', latest_dataset)


    list_of_files = glob.glob(latest_dataset+'\*.tif')
    latest_file = min(list_of_files, key=os.path.getctime)
    
    print('loading from file', latest_file)


    inf = read_dataset_from_seed_file(latest_file, load_time=False)
    
    show_ds(ds_inf=inf)

def display():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--last", help="show last dataset",
                        action="store_true")
    parser.add_argument("-n", "--nometainfo", help="load dataset w/o metainfo",
                        action="store_true")
    args = parser.parse_args()
    if args.last:
        last_show_ds()
    elif args.nometainfo:
        load_show_nmi()
    else:
        load_show_ds()

if __name__ == "__main__": 
    #locale.setlocale(locale.LC_ALL, 'en-US')
    locale.setlocale(locale.LC_ALL, '')
    display()
    