from setuptools import setup, find_packages

setup(
    name='cartpole',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts':[
            'cartpole_train = cartpole.learn:main',
            'cartpole_play = cartpole.play:main'
        ]
    }
)
