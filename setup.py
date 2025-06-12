from setuptools import setup, find_packages

setup(
    name="transaction-classifier-ml",
    version="0.1",
    packages=find_packages(include=['src', 'src.*', 'config', 'config.*']),
    package_dir={'': '.'},
    install_requires=[
        'tensorflow>=2.11',
        'scikit-learn',
        'pandas',
        'numpy',
        'jupyter',
        'matplotlib',
        'seaborn'
    ],
    python_requires='>=3.8,<3.12',
) 