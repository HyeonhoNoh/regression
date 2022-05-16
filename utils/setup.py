from setuptools import setup, find_packages

setup(name='utils',
        py_modules=['utils'],
        install_requires=[
            'torch',
            'numpy',
            'scipy',
            'gym',
            'matlab',
            'pylab-sdk'
            ],
        description="802.11ax HDQN",
        author="Hyeonho Noh",)
