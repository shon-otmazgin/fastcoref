from setuptools import setup, find_packages

setup(
    name='fastcoref',
    version='1.2',
    license='MIT',
    author="Shon Otmazgin, Arie Cattan, Yoav Goldberg",
    author_email='shon711@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/shon-otmazgin/fastcoref',
    install_requires=[
        'tqdm>=4.64.0',
        'numpy>=1.21.6',
        'scipy>=1.7.3',
        'pandas>=1.3.5',
        'spacy>=3.0.6',
        'torch>=1.10.0',
        'transformers>=4.11.3',
        'datasets>=2.1.0',
        'wandb>=0.12.15',
      ],

)
