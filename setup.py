import setuptools


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    with open(filename) as f:
        required = f.read().splitlines()
    return required


setuptools.setup(
    name="neural_de",
    packages=setuptools.find_packages(),
    version="1.1",
    author="Nelson Fernandez Pinto",
    author_email="nelson.fernandez-pinto@irt-systemx.fr",
    description="Image enhancement library: corruption (noise, meteorological...), removal",
    include_package_data=True,
    license="TODO",
    install_requires=parse_requirements('requirements_python39.txt'),
    entry_points={
        'console_scripts': [
            'neural_de = neural_de.main:main',
        ]},
)

