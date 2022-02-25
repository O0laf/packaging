import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="face_align_doyeon",
    version="0.0.1",
    author="Doyeon Kim",
    author_email="doyeon@innerverz.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires = [
    'numpy',
    'opencv-python',
    'dlib',
    'facenet-pytorch',
    'face-alignment',
    'scipy',
    'torchvision',
    'importlib-resources'],
    include_package_data=True
)