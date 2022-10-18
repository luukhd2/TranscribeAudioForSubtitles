from setuptools import setup
# todo, install this library automatically
# python -m pip install huggingsound==0.1.5
# python -m pip install librosa==0.8.1
# python -m pip install pysoundfile==0.9.0
# python -m pip install moviepy

# WIP
# python -m pip install pydub

# IGNORE, NOT USEFUL CAUSE REQUIRES FFMPEG: python -m pip install transformers?? maybe already included somewhere
# IGNORE, NOT USEFUL CAUSE REQUIRES FFMPEG: python -m pip install pip install ffmpeg-python




# todo, specify python version and add check for it

setup(
    name='AddRussianSubtitlesToVideo',
    version='0.1.0',    
    description='Use hugginface deep learning model to transcribe Russian audio to text',
    url='https://github.com/luukhd2/AddRussianSubtitlesToVideo',
    author='Luuk HD',
    author_email='luukhd@gmail.com',
    license='GNU GENERAL PUBLIC LICENSE',
    packages=['pyexample'],
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - First version',
        'Intended Audience :: People learning Russian through videos, but want word specific subtitles.',
        'Programming Language :: Python :: 3.8',
    ],
)