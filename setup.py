from setuptools import setup, find_packages
from os import path

# read the contents of README.rst
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sora-astro',
    packages=find_packages(),
    package_data={'sora': ['data/*']},
    version='0.3',
    license='MIT',
    description='Stellar Occultation Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='SORA Team',
    author_email='rio.occteam@gmail.com',
    url='https://github.com/riogroup/SORA',
    keywords=['science', 'astronomy', 'occultation'],
    install_requires=[
        'numpy>=1.21',
        'pyerfa>=2.0',
        'astropy>=4.3.1',
        'astroquery>=0.4.6',
        'spiceypy>=5.1.1',
        'matplotlib>=3.5.3',
        'scipy>=1.7.1',
        'requests',
        'tqdm>=4.64',
        'shapely>=1.8.2',
    ],
    python_requires=">=3.7, <4",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/riogroup/SORA/issues',
        'Documentation': 'https://sora.readthedocs.io/',
    },
)
