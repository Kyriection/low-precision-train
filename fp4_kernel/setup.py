from setuptools import setup
import pkg_resources, os, sys
from setuptools.command.install import install
from setuptools.command.develop import develop

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        if not 'torch' in installed_packages:
            os.system("pip install -y --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        if not 'torch' in installed_packages:
            os.system("pip install -y --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")

setup(
    name="FP4 Kernels",
    version="1.0.0",
    author="Bas Maat",
    author_email="",
    description="For running true FP4 layers",
    url="",
    packages=["fp4_torch_kernel"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',  # Minimum Python version
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand
    }
)