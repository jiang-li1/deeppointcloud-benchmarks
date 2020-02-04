import json
import subprocess
import itertools
import pathlib 

import pdal
import re
import numpy as np

datadir = '/home/tristan/Downloads/'
datafile = 'C_38FN1.LAZ'

path = pathlib.Path(datadir + datafile)

split = 4 #number of divisions in x and y - number of output tiles will be split^2

def exe_command(cmd):
    print('>>>>> Running: \n' + ' '.join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = result.stdout if len(result.stdout) != 0 else result.stderr
    result = result.decode()
    print('>>>>> Result: \n', result)
    return result

# pipeline = [
#     {
#         "type": "readers.las",
#         "filename": datadir + '/' + datafile
#     },
#     {
#         "type": "filters.info"
#     }
# ]

# jsonStr = json.dumps(pipeline)
# print(jsonStr)
# pdalPipeline = pdal.Pipeline(jsonStr)
# pdalPipeline.validate()
# pdalPipeline.execute()

result = exe_command(['lasinfo', '-no_check', datadir + datafile])

minX, minY = map(float, re.search(r'min x y z:\s*(\S*)\s(\S*)', result).groups())
maxX, maxY = map(float, re.search(r'max x y z:\s*(\S*)\s(\S*)', result).groups())
sizeX = maxX - minX
sizeY = maxY - minY

subtileTopLefts = itertools.product(
    np.linspace(minX, maxX, num=split, endpoint=False),
    np.linspace(minY, maxY, num=split, endpoint=False)
)

subtiles = [((tlx, tly), (tlx + sizeX/split, tly + sizeY/split)) for tlx, tly in subtileTopLefts]
print('Computes subtiles: ', subtiles)

exe_command(['lasindex', '-dont_reindex', '-i', datadir + datafile])

for i, ((tlx, tly), (brx, bry)) in enumerate(subtiles):
    exe_command(['las2las', '-i', str(path), 
        '-o', str(path.with_name(path.stem + '_{}.LAZ'.format(i))),
        '-inside', *map(str, [tlx, tly, brx, bry]),
    ])



