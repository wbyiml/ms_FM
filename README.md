# ms_task2

## 利用MindSpore实现广告推荐网络FM(Factorization Machine)

数据集criteo存放在 datas/set下：
  FM
  --datas
    --set
      --train.txt
      --test.txt
      
datas/makeDataset.py 可用于将train.txt中抽取部分数据作为训练集和验证集
<br />


训练：
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path datas/midset/train.txt --device_target GPU

评估：
CUDA_VISIBLE_DEVICES=1 python eval.py --dataset_path datas/midset/test.txt --device_target GPU --ckpt_path outputs/fm-100_439.ckpt
<br />

在train.txt 前1000000条数据中（train 900000,val 100000）,结果精度为：0.781

模型下载：https://pan.baidu.com/s/1PfgGk9DIOBkHIojcJI3q3Q

提取码：fabv
