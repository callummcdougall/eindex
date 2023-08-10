from setuptools import setup, find_packages

setup(
    name='eindex',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'jaxtyping',
        'einops',
        'numpy',
    ],
    author='Callum McDougall',
    author_email='cal.s.mcdougall@gmail.com',  # Replace with your email
    description='A brief description of your project',  # Replace with your project's description
    url='https://github.com/callummcdougall/eindex',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)