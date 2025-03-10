from setuptools import setup, find_packages

setup(
    name='markdown-indexer',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A module for indexing Markdown text using embeddings and storage backends.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'markdown',
        'transformers',
        'faiss-cpu',  # or 'faiss-gpu' if you want GPU support
        'azure-search-documents',
        'numpy',
        'pandas'  # if you are handling tables
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)