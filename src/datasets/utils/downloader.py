import os.path as osp
import os
import errno
import urllib.request

from tqdm import tqdm 

#from https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/download.py
def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print('Using existing file:', filename)
        return path

    if log:
        print('Downloading:', url)
        print('To file:', path)

    _makedirs(folder)
    # data = urllib.request.urlopen(url)

    # with open(path, 'wb') as f:
    #     f.write(data.read())

    _download_url_helper(url, path)

    return path


#from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads/15645088
class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_url_helper(url, output_path):
    with _DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def _makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e