
▼Abstract
[2011.08785v1] PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
https://arxiv.org/abs/2011.08785v1

▼Paper
https://arxiv.org/pdf/2011.08785v1.pdf

▼Code
PaDiM-Anomaly-Detection-Localization-master: 
https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

▼日本語資料
PaDiM : 再学習不要で不良品検知を行う機械学習モデル. ailia… | by Kazuki Kyakuno | axinc | Medium
https://medium.com/axinc/padim-%E5%86%8D%E5%AD%A6%E7%BF%92%E4%B8%8D%E8%A6%81%E3%81%A7%E4%B8%8D%E8%89%AF%E5%93%81%E6%A4%9C%E7%9F%A5%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-69add653fbd3

異常検知 SPADE登場からの1年間 - Qiita
https://qiita.com/h1day/items/e65d840e557a14cf1a37

PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization | Nakatsuka Shunsuke
https://salty-vanilla.github.io/portfolio/post/padim/


▼インストール(condaも可）
python == 3.7
pytorch == 1.5
tqdm
sklearn
matplotlib
など

▼MVTEC ADデータセット(Download the whole dataset (4.9 GB))
https://www.mvtec.com/company/research/datasets/mvtec-ad



▼使いかた
main.py, original_data.py の parse_args() でデータパスの設定をする

mapin.py:
	ほぼ公式実装　trainとtestが一体

original_data.py:
	rain()		# 学習用
	test()		# 画像を使って予測
	capture()	# USBカメラで撮影して予測



▼ToDo
GUI作って号口に入れてみようかな
