import os.path as osp
import subprocess
import itertools
import pathlib

import re
import numpy as np 

# from 

def split_las_pointcloud(filename, num_subclouds, output_dir):
    '''Split a pointcloud stored in a las file into split * split blocks in x and y, 
        where split = num_subclouds//num_subclouds.

        Uses LAZtools
    '''

    split = int(num_subclouds**0.5)

    print('Splitting {} by {}x{}...'.format(filename, split, split))

    path = pathlib.Path(filename)

    try:
        info = _exe_command(['lasinfo', '-no_check', str(path)])
    except FileNotFoundError as e:
        print(e)
        print("Using LAZtools failed. Are they installed and on PATH?")

    minX, minY = map(float, re.search(r'min x y z:\s*(\S*)\s(\S*)', info).groups())
    maxX, maxY = map(float, re.search(r'max x y z:\s*(\S*)\s(\S*)', info).groups())
    sizeX = maxX - minX
    sizeY = maxY - minY

    subtileTopLefts = itertools.product(
        np.linspace(minX, maxX, num=split, endpoint=False),
        np.linspace(minY, maxY, num=split, endpoint=False)
    )

    subtiles = [((tlx, tly), (tlx + sizeX/split, tly + sizeY/split)) for tlx, tly in subtileTopLefts]
    print('Computes subtiles: ', subtiles)

    _exe_command(['lasindex', '-dont_reindex', '-i', str(path)])

    for i, ((tlx, tly), (brx, bry)) in enumerate(subtiles):
        output_filename = output_dir + (path.stem + '_{}.LAZ'.format(i))

        if osp.exists(output_filename):
            print('Split {} already exists. Skipping'.format(output_filename))
        else:
            _exe_command(['las2las', '-i', str(path), 
                '-o', output_filename,
                '-inside', *map(str, [tlx, tly, brx, bry]),
            ])



def _exe_command(cmd):
    print('>>>>> Running: \n' + ' '.join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = result.stdout if len(result.stdout) != 0 else result.stderr
    result = result.decode()
    print('>>>>> Result: \n', result)
    return result

