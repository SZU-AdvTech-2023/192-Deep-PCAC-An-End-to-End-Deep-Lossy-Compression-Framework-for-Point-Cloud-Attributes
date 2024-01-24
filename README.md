### Requirments

tensorflow-gpu 1.13.1

open3d

nvidia-docker2

cuda10.0

python 3.6

### 预训练模型

预先训练好的模型存储在 /model/256、/model/200、/model/128、/model/64 中。请修改这些文件夹中的 "checkpoint "文件，并更改查找 ckpt 的绝对路径。(我已将 "checkpoint "文件的路径中性化，您可以直接运行它们。如果路径有问题，请修改 "checkpoint "文件中的路径）。

### 编码

python mycodec.py compress --input="./testdata/soldier_vox10_0690.ply" --ckpt_dir='./model/256/' --latent_points=256

### 解码

python mycodec.py decompress --input="./testdata/soldier_vox10_0690" --ckpt_dir='./model/256/' --latent_points=256