import os
import os.path as osp
import ycm_core

flags = [
  '-std=c++17',
  '-I' + osp.join(osp.realpath(osp.dirname(__file__)), 'src')
]

SOURCE_EXTENSIONS = ['.cpp', '.cxx', '.cc', '.c']

def FlagsForFile(filename, **kwargs):
    return {
        'flags': flags,
        'do_cache': True
    }

