from setuptools import setup, find_packages

setup(
    name='eindex',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'einops'  # Add this line
    ],
    author='Callum McDougall',
    author_email='cal.s.mcdougall@gmail.com',
    description='My interpretation of indexing with einops-like pattern notation',
    url='https://github.com/callummcdougall/eindex',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
