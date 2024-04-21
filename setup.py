from setuptools import setup, find_packages

setup(
    name = 'eindex-callum',
    version  =  '0.1.1',
    packages = find_packages(),
    install_requires = [
        'torch',
        'einops'
    ],
    author = 'Callum McDougall',
    author_email = 'cal.s.mcdougall@gmail.com',
    description = 'My interpretation of einops-like indexing',
    url = 'https://github.com/callummcdougall/eindex',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
