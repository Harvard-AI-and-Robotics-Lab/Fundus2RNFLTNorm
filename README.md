# OCT-Fundus-to-RNFLT

The code for predicting a circumpapillary **retinal nerve fiber layer thickness (RNFLT) map** from an **OCT fundus image**. If you have any questions, please email harvardophai@gmail.com and harvardairobotics@gmail.com.

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
The model weight "octfundus_to_rnflt_new2.final.h5" is included at `checkpoint/octfundus_to_rnflt_new2.final.h5`.

## Use the Model
````
from octfundus2rnflt import *

# load the pretrained model
octfundus2rnflt = PCModel(img_rows=256, img_cols=256)
octfundus2rnflt.load('checkpoint/octfundus_to_rnflt_new2.final.h5', train_bn=False, lr=0.00001)

# RNFLT prediction
pred = octfundus2rnflt.model.predict([img, mask])[0][:,:,0]
plot_2dmap(pred, show_cup=True)
````

A notebook version of this example is in `inference.ipynb`.

To choose a device, set `CUDA_VISIBLE_DEVICES` before the first import (`''` runs on CPU):
````
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from octfundus2rnflt import *
````

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

If you find this repository useful for your research, please consider citing our work.
