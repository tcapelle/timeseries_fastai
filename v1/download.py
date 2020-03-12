from pathlib import Path
from typing import Union
import os
import shutil
import requests
from fastprogress.fastprogress import progress_bar
import zipfile


PathOrStr = Union[Path,str]
LOCAL_PATH = Path.cwd()
DATA_PATH = Path.home()/'tmp'
UCR_LINK = 'http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip'

## From fastai

def ifnone(a, b): return b if a is None else a
def url2name(url): return url.split('/')[-1].split('.')[0]

def url2path(url, ext='.zip'):
    "Change `url` to a path."
    name = url2name(url)
    return DATA_PATH/(name+ext)

def datapath4file(filename, ext:str='.zip'):
    local_path = LOCAL_PATH/'data'/filename
    return local_path

def download_data(url:str, fname:PathOrStr, data:bool=True, ext:str='.tgz') -> Path:
    "Download `url` to destination `fname`."
    fname = Path(fname)
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists():
        print(f'Downloading {url}')
        download_url(url, fname)
    return fname

def unzip_data(url:str=UCR_LINK, fname:PathOrStr=None, dest:PathOrStr=None, force_download=False) -> Path:
    "Download `url` to `fname` if `dest` doesn't exist, and un-zip to folder `dest`."
    fname = Path(ifnone(fname, url2path(url)))
    dest = LOCAL_PATH/'Univariate_arff' if dest is None else Path(dest)
    fname = Path(ifnone(fname, url2path(url)))
    if force_download:
        print(f"A new version of the dataset is available.")
        if fname.exists(): os.remove(fname)
        if dest.exists(): shutil.rmtree(dest)
    if not dest.exists():
        fname = download_data(url, fname=fname)
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall(dest.parent)
    else: print(f'Files present in : {dest}')
    return dest

def download_url(url:str, dest:str, overwrite:bool=False,
                 show_progress=True, chunk_size=1024*1024, timeout=4, retries=5)->None:
    "Download `url` to `dest` unless it exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite: return

    s = requests.Session()
    s.mount('http://',requests.adapters.HTTPAdapter(max_retries=retries))
    u = s.get(url, stream=True, timeout=timeout)
    try: file_size = int(u.headers["Content-Length"])
    except: show_progress = False

    with open(dest, 'wb') as f:
        nbytes = 0
        if show_progress: pbar = progress_bar(range(file_size), auto_update=False, leave=False)
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                if show_progress: pbar.update(nbytes)
                f.write(chunk)
        except requests.exceptions.ConnectionError as e:
            print(f'Try downloading your file manually from {url}')
            import sys;sys.exit(1)