
import matplotlib.pyplot as plt
import numpy as np

from pcd_utils import PointCloud, AHNPointCloud
import pcd_utils

datapath = '/home/tristan/data/'
# cloud = '37EN2_11_section.laz'
cloud = 'adriaan_tiles/37EN2_11.LAZ'

# pointcloud = PointCloud(datapath + cloud)

pcd = AHNPointCloud.from_cloud(datapath + cloud)

# bins = np.logspace(np.log10(1), np.log10(1e5), 100)
# plt.hist(pointcloud.arr['Intensity'], bins=bins)
# plt.yscale('log', nonposy='clip')
# plt.xscale('log')

# pointcloud.arr = pcd_utils.get_log_clip_intensity(pointcloud.arr)

# intensity = pointcloud.arr['Intensity']

# plt.hist(intensity, bins=50)

# plt.hist(pointcloud.arr['Classification'], bins = range(0, 27))

# plt.yscale('log', nonposy='clip')

# plt.show()

def clas_boxplot(pcd, field):

    clasClouds = pcd.split_to_classes()

    fieldVecs = [getattr(p, field) for c, p in clasClouds.items()]

    plt.boxplot(fieldVecs)

    plt.xticks(list(range(1, len(pcd.clasNames) + 1)), pcd.clasNames)
    plt.ylabel(field)
    plt.title(pcd.name)

    plt.show(block=False)

def features_correlation(pcd : AHNPointCloud):

    df = pcd.to_dataframe()
    corrMat = df.corr()

    plt.matshow(corrMat)

    plt.xticks(range(df.shape[1]), df.columns, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns)
    plt.colorbar()
    plt.title(pcd.name)
    plt.show()