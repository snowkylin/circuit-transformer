from setuptools import setup

setup(
    name='circuit-transformer',
    version='1.0',
    packages=['circuit_transformer'],
    url='https://github.com/snowkylin/circuit-transformer',
    license='MIT',
    author='Xihan Li (snowkylin)',
    author_email='xihan.li@cs.ucl.ac.uk',
    description='An end-to-end Transformer model that efficiently produces logic circuits strictly equivalent to given Boolean functions.',
    install_requires=["tensorflow[and-cuda]",
                      "tf_keras",
                      "tf-models-official",
                      "npn",
                      "graphviz",
                      "nvidia-ml-py",
                      "bitarray",
                      "huggingface_hub"],
    package_data={'circuit_transformer': ['bin/*']}
)
