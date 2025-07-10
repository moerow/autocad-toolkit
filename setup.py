from setuptools import setup, find_packages

setup(
    name="autocad-construction-toolkit",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pyautocad>=0.2.0",
        "customtkinter>=5.2.0",
        "click>=8.0.0",
        "PyPDF2>=3.0.0",
        "pandas>=1.5.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "acad-toolkit=src.main:main",
            "acad-toolkit-cli=src.presentation.cli.cli_app:cli",
        ],
    },
)
