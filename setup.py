#nsml: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
# 위와 같이 파일 첫 줄에 #nsml: 뒤에 사용할 docker image 를 써주면 그 이미지 위에서 아래의 라이브러리들이 설치된 환경에서 코드가 돌아갑닏.

from distutils.core import setup

setup(
    name='nia dm hackathon example',
    version='1.0',
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'imblearn',
        'smote_variants',
    ]
)

# available images
# 1.5-cuda10.1-cudnn7-runtime
# 1.6.0-cuda10.1-cudnn7-runtime
# 1.8.0-cuda11.1-cudnn8-runtime
# 1.9.1-cuda11.1-cudnn8-runtime
