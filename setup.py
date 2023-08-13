from setuptools import setup, find_packages

setup(
    name='eindex',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch'
    ],
    author='Callum McDougall',
    author_email='cal.s.mcdougall@gmail.com',  # Replace with your email
    description='My interpretation of indexing with einops-like pattern notation',  # Replace with your project's description
    url='https://github.com/callummcdougall/eindex',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
