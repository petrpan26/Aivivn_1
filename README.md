# Aivivn_1

Our submission for Aivivn Contest 1.

By Nhat Pham and Hoang Phan.

Install environment:
```bash
conda install python=3.6
```

Dependencies guide:

```bash
pip install -r requirements.txt
cd external_lib
chmod a+x install_lib.sh
./install_lib.sh
cd ..
```
Notebook test link:
https://colab.research.google.com/drive/1fgtIYXkXKKmZVI2w62nCI22wiVSNEQxw

Sample run command:

```bash
python -m main -m VDCNN -e ./embeddings/baomoi.model.bin --max 40000 --mix --prob
```
