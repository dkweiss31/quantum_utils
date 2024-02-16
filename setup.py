from setuptools import setup

setup(
    name='quantum_utils',
    version='0.1',
    packages=["quantum_utils", ],
    description='commonly used utilities, all in one place',
    long_description=open('README.md').read(),
    author='Daniel Weiss',
    author_email='daniel.kamrath.weiss@gmail.com',
    url='https://github.com/dkweiss31/quantum_utils',
    install_requires=["numpy", "scipy", "h5py", "pathos", "typing"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
