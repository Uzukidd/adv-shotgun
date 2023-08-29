from setuptools import setup, find_packages
import os


current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''
    
try:
    with open(os.path.join(current_directory, 'advshotgun/VERSION'), encoding='utf-8') as f:
        version = f.read()
except Exception:
    version = ''

setup(
    name="adv-shotgun",
    packages=find_packages('.'), 
    version=version,
    license='MIT',
    description='',
    long_description = long_description,
    long_description_context_type = 'text/markdown',
    author='', 
    author_email='',     
    url='',
    download_url='',
    keywords=[],
    install_requires=[],
    classifiers=[]  
)