from setuptools import setup, find_packages
from pathlib import Path

def get_requirements(file_path: str = 'requirements.txt'):
    requirements = []
    has_hypen_e_dot = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line == '-e .':
                has_hypen_e_dot = True
                continue
            requirements.append(line)
    return requirements, has_hypen_e_dot

requirements, has_hypen_e_dot = get_requirements()

if has_hypen_e_dot:
    print("Detected '-e .' in requirements.txt. This is skipped from install_requires.")

setup(
    name='mlop_project',  # Ganti dengan nama proyekmu
    version='0.0.1',
    author='mfikry_rz',
    author_email='fikryyoshiko70@gmail.com',
    description='Proyek machine learning untuk MLOp',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/mfikryrz/mlop_project.git',
    packages=find_packages(where='src'),
    install_requires=requirements
)
