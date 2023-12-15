from setuptools import setup, find_packages

setup(
    name='vpfrc_text_classifier',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "transformers",
        "ipython",
        "openai"
    ],
)

