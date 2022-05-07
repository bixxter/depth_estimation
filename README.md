# Depth estimation using MiDaS🌌

![](br.jpg)

## 📌Практика 1: /Depth_map_Midas_v2.1

Тут используем старую модельку MiDaS_v2.1, где был CNN.

Важно скачать модельки по этой [ссылке](https://github.com/isl-org/MiDaS/releases/tag/v2_1) и положить в /models

- model-f6b98070.onnx
- model-small.onnx

## 📌Практика 2: /Depth_estimation_MiDaS_v3

Тут уже новая версия MiDaS_v3, где модельку обучили на куче других параментрах. Как итог, результаты более показательны. Ещё одно отличие - тут используются трансформеры, как в NLP.

## 📌Практика 3: /Depth_estimation_MiDaS_v3/gi_based

Тут запускается локальный сервак, с интерфейсом. Можно заливать свои фоточки

## 📜Статьи:

- MiDaS - [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](http://vladlen.info/papers/midas.pdf)
- Бустинг MiDas (как улучшали MiDaS) - [Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging](http://yaksoy.github.io/papers/CVPR21-HighResDepth.pdf)
  Demo бустинга - http://yaksoy.github.io/highresdepth/
- Попытка использовать DE - [Consistent Video Depth Estimation](https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/)

## ⚙️Deps(надо поставить):

- pip install cv2(важно)<br />
- pip install pytorch (важно)<br />
- pip install timm (важно)<br />
- pip install numpy<br />
- pip install matplotlib<br />
- pip install scikit-learn<br />

Всё👍🏻
