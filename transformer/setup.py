from setuptools import setup, find_packages

# 定义包的元数据
setup(
    name='MyTransformer',  # 包的名称
    version='0.1.0',  # 包的版本号
    author='Zhang Chenxiao',  # 包的作者
    author_email='',  # 作者的电子邮件地址
    description='Implementation of Transformer',  # 包的简短描述
    packages=find_packages(),  # 要安装的包，自动查找所有包
    install_requires=[  # 包的依赖项
        'numpy',
        'torch'
        # 添加其他依赖项
    ],
    classifiers=[  # 用于PyPI分类的元数据
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.9',  # 指定支持的Python版本
)