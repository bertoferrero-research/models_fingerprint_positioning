# Copyright 2024 Alberto Ferrero López
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name='bertoferrero-positioning-fingerprint',
    version='0.4.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow<=2.15.0',
        'scikit-learn',
        'autokeras==1.1.0',
        'keras-tuner'
    ],
    license='Apache License 2.0', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License', 
        'Operating System :: OS Independent',
    ],
)