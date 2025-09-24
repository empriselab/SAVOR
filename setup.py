"""
Setup script for SAVOR: Skill Affordance Learning from Visuo-Haptic Perception
for Robot-Assisted Bite Acquisition
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="savor",
    version="1.0.0",
    author="Zhanxin Wu",
    author_email="zw754@cornell.edu",
    description="Skill Affordance Learning from Visuo-Haptic Perception for Robot-Assisted Bite Acquisition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://emprise.cs.cornell.edu/savor/",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "savor-train=train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)

