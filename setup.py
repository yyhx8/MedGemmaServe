"""MedServer — One-command MedGemma serving platform."""

from setuptools import setup, find_packages

setup(
    name="medserver",
    version="1.0.0",
    description="Self-hosted MedGemma clinical AI server with auto-install and premium web UI",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="MedGemmaServe",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "medserver": ["static/**/*", "templates/**/*"],
    },
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "sglang>=0.1.0; platform_system=='Linux'",
        "transformers>=4.48.0",
        "accelerate>=1.0.0",
        "bitsandbytes>=0.46.1",
        "huggingface-hub>=0.25.0",
        "slowapi>=0.1.9",
    ],
    entry_points={
        "console_scripts": [
            "medserver=medserver.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
    ],
)
