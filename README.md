# En Face Fundus-to-RNFLT Norm

The code for the paper entitled [**Deep Learning Prediction of Personalized Peripapillary Retinal Nerve Fiber Layer Thickness Norms from Fundus Images in Glaucoma**](https://pubmed.ncbi.nlm.nih.gov/42245038/). If you have any questions, please email harvardophai@gmail.com and harvardairobotics@gmail.com.

## Requirements
Python 3.8  
tensorflow 2.4.0  
opencv-python 4.5.5

## Dataset

A sample OCT fundus image is provided at `samples/sample_oct_fundus.png`. Here are sample codes to load it:

````
from octfundus2rnflt import *

oct_fundus = cv2.resize(cv2.imread('samples/sample_oct_fundus.png', 0), (256, 256))
img = np.array([np.transpose(np.array([oct_fundus, oct_fundus, oct_fundus]), (1, 2, 0))])
mask = np.ones_like(img)

plt.imshow(oct_fundus, cmap='gray')
plt.show()
````

## Pretrained Model
The model weight "octfundus_to_rnflt_model.final.h5" is available at https://huggingface.co/datasets/harvardairobotics/Fundus2RNFLTNorm.

## Use the Model
````
from octfundus2rnflt import *

# load the pretrained model
octfundus2rnflt = PCModel(img_rows=256, img_cols=256)
octfundus2rnflt.load('checkpoint/octfundus_to_rnflt_model.final.h5', train_bn=False, lr=0.00001)

# RNFLT prediction
pred = octfundus2rnflt.model.predict([img, mask])[0][:,:,0]
plot_2dmap(pred, show_cup=True)
````

A notebook version of this example is in `inference.ipynb`.

#### RNFLT prediction example:

## Training

Prepare your own `.npz` samples. Each file should contain `oct_fundus` and `rnflt`. A metadata CSV should have columns `filename` and `train` (`1` = training, `0` = held-out).

````
python train.py \
  --data-path /path/to/npz_folder \
  --meta-csv /path/to/metatable.csv \
  --checkpoint-dir checkpoint/ \
  --model-name octfundus_to_rnflt_new2 \
  --img-rows 256 \
  --img-cols 256 \
  --batch-size 8 \
  --epochs 200 \
  --lr 0.00001 \
  --gpu 0
````

## Acknowledgement and Citation

If you find this repository useful for your research, please consider citing our [paper](https://pubmed.ncbi.nlm.nih.gov/42245038/):

```bibtex
@article{yildiz2026rnfltnorm,
  title={Deep Learning Prediction of Personalized Peripapillary Retinal Nerve Fiber Layer Thickness Norms from Fundus Images in Glaucoma},
  author={Yildiz, Elif and Zha, Lucy and Zebardast, Nazlee and Shi, Min and Wang, Mengyu},
  journal={medRxiv},
  year={2026},
  doi={10.64898/2026.05.26.26354081},
  url={https://pubmed.ncbi.nlm.nih.gov/42245038/}
}
```
