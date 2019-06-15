import os
import os.path as osp
import ycm_core

flags = [
  '-std=c++17',
  '-Wall',
  '-Wextra',
  '-Werror',
  '-Wno-long-long',
  '-Wno-variadic-macros',
  '-fexceptions',
  '-ferror-limit=10000',
  '-DNDEBUG',
  '-std=c99',
  '-xc',
  '-I' + osp.join(osp.realpath(osp.dirname(__file__)), 'src')
]

SOURCE_EXTENSIONS = ['.cpp', '.cxx', '.cc', '.c']

def FlagsForFile(filename, **kwargs):
    return {
        'flags': flags,
        'do_cache': True
    }

