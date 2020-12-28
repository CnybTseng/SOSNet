import os
import glob

paths = sorted(glob.glob(os.path.join('./', '*.tar')))
for path in paths:
    _, name_with_suffix = os.path.split(path)
    name, _ = os.path.splitext(name_with_suffix)
    subdir = os.path.join('.', name)
    os.makedirs(subdir)
    print('uncompress {} to {}'.format(path, subdir))
    os.system('tar xvf {} -C {}'.format(path, subdir))