from setuptools import setup

setup(name='deepfetal',
      version='0.1',
      description='Tools for deep fetal phenotyping',
      #url='http://github.com/...',
      author='Lucilio Cordero-Grande',
      author_email='lucilio.cordero@upm.es',
      license='MIT',
      packages=['deepfetal', 'deepfetal.arch', 'deepfetal.build', 'deepfetal.lay', 'deepfetal.meth', 'deepfetal.opt', 'deepfetal.unit'],
      zip_safe=False)
