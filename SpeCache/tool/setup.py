from setuptools import setup, find_packages

setup(
    name="specache",
    version="0.1.0",
    description="Speculative KV Cache experiments for LLMs",
    author="adanilishin",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
    ],
    extras_require={
        "dev": [
            "pytest",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)