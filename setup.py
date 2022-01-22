import setuptools

PACKAGE_NAME = "MotifBoost"

setuptools.setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    description="Repertoire Classification for Small Data",
    long_description=open("README.md").read().strip(),
    long_description_content_type="text/markdown",
    author="hmirin",
    url="http://github.com/hmirin/MotifBoost",
    python_requires=">=3.9",
    py_modules=[PACKAGE_NAME],
    install_requires=[
        "click>=7",
        "cloudpickle",
        "joblib",
        "mlflow",
        "tqdm",
        "lightgbm",
        "immuneML",
        "pandas",
        "scipy",
        "numba",
        "numba-scipy",
        "numpy",
        "bitarray",
        "pyarrow",
        "catboost",
        "xgboost",
        "optuna"
    ],
    extras_require={"dev": ["nose", "coverage", "black", "isort", "numpy"]},
    zip_safe=False,
    keywords="",
    classifiers=[],
    packages=setuptools.find_packages(exclude=["contrib", "docs", "tests*"]),
    include_package_data=True,
)
