import setuptools

PACKAGE_NAME = "MotifBoost"

setuptools.setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    description="Repertoire Classification for Small Data",
    long_description=open("README.md").read().strip(),
    long_description_content_type="text/markdown",
    author="hmirin",
    url="https://github.com/hmirin/MotifBoost",
    python_requires=">=3.7",
    py_modules=[PACKAGE_NAME],
    install_requires=[
        "click",
        "cloudpickle",
        "pickle5",
        "joblib",
        "mlflow",
        "tqdm",
        "lightgbm",
        "h5py==3.7.0",
        "immuneML==2.1.1",  # depends on h5py
        "pandas",
        "scipy",
        "numba",
        "numba-scipy",  # depends on scipy == 1.6.2, which doesn't work on M1 Mac
        "scipy==1.6.2",
        "numpy",
        "bitarray",
        "pyarrow",
        "catboost",
        "xgboost",
        "optuna",
    ],
    extras_require={"dev": ["nose", "coverage", "black", "isort", "numpy"]},
    zip_safe=False,
    keywords="",
    classifiers=[],
    packages=setuptools.find_packages(exclude=["contrib", "docs", "tests*"]),
    include_package_data=True,
)
