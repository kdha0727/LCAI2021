# Larynx Cancer Artificial Intelligence Datathon 2021 Baseline

## Baseline Structure

- [main.py](./main.py)
  - train / inference 수행할 메인 코드
- [setup.py](./setup.py)
  - nsml docker configuration
- [.nsmlignore](./.nsmlignore)
  - nsml run은 파일을 업로드하여 가상 서버에서 진행. 
  - 올라가면 안 되는 파일의 경우 .nsmlignore에 추가하여 업로드하지 않을 수 있다.
  - .gitignore rule과 동일
- [nsml.py](./nsml.py)
  - nsml baseline에 맞는 server-side nsml package 구현체
- 기타 python 관련 파일 자유롭게 추가 가능

## About NSML
- NSML은 대회 진행용으로 이용할 수도 있고, 가상 환경으로서 이용할 수도 있다.
- 모델의 checkpoint 를 Save, Load를 쉽게 할 수 있다.
- Docker Container 를 올려놨다가 훈련이 끝나면 바로 docker exit. 

### NSML 이용방법

자세한 내용은 [여기](https://n-clair.github.io/ai-docs/_build/html/ko_KR/index.html) 를 참고하세요.

```bash
# 명칭이 'nia_dm_prelim'인 데이터셋을 사용해 세션 실행하기
$ nsml run -d nia_dm
# 메인 파일명이 'main.py'가 아닌 경우('-e' 옵션으로 entry point 지정)
# 예: nsml run -d nia_dm -e main_lightgbm.py
$ nsml run -d nia_dm -e [파일명]

# 세션 로그 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml logs -f [세션명]

# 세션 종료 후 모델 목록 및 제출하고자 하는 모델의 checkpoint 번호 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml model ls [세션명]

# 모델 제출 전 제출 코드에 문제가 없는지 점검하기('-t' 옵션)
$ nsml submit -t [세션명] [모델_checkpoint_번호]

# 모델 제출하기
# 제출 후 리더보드에서 점수 확인 가능
$ nsml submit [세션명] [모델_checkpoint_번호]

# session 멈추기
$ nsml stop [세션명]
```
- model checkpoint 이어서 훈련가능
- 모델의 저장은 nsml.save를 통해서 저장가능 (epoch 하나마다 저장하면 memory 많이 차지함)
