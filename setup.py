from setuptools import setup, find_namespace_packages


def _read(f):
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="image_dataset_converter_imgaug",
    description="Image augmentation extension for the image-dataset-converter library.",
    long_description=(
            _read('DESCRIPTION.rst') + b'\n' +
            _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/waikato-datamining/image-dataset-converter-imgaug",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    license='MIT License',
    package_dir={
        '': 'src'
    },
    packages=find_namespace_packages(where='src'),
    install_requires=[
        "image_dataset_converter>=0.0.5",
        "imgaug3",
        "matplotlib",
        "Shapely",
        "simple-mask-utils==0.0.1"
    ],
    version="0.0.8",
    author='Peter Reutemann',
    author_email='fracpete@waikato.ac.nz',
    entry_points={
        "console_scripts": [
            "idc-generate-regions=idc.imgaug.tool.generate_regions:sys_main",
            "idc-combine-sub-images=idc.imgaug.tool.combine_sub_images:sys_main",
        ],
        "class_lister": [
            "idc.imgaug=idc.imgaug.class_lister",
        ],
    },
)
