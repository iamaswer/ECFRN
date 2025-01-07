# ECFRN

Code relaese for [Ensembled Cross-Class Feature Reconstruction for
 Few-Shot Fine-Grained Image Classification](article_link).

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yaml
  conda activate ECFRN
  ```

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN), but the categories  of train, val, and test follows split.txt. And then move the processed dataset  to directory ./data.

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1KLvyo-frDFxCKBj2X1Mi7meWqtpdqHyP/view?usp=sharing)\]
- cars \[[Download Link](https://drive.google.com/file/d/18uyevIiF2YoX-c-GPb9iAs1LmERTmCUb/view?usp=sharing)\]
- dogs \[[Download Link](https://drive.google.com/file/d/17U8fuR2yqfDL5DJB10JW5F3AE8DLBShe/view?usp=sharing)\]
- iNaturalist2017 \[[Download Link](https://drive.google.com/file/d/1s0SJXE-gQMnH_Zj4D0nJpILNs3WOljKQ/view?usp=sharing)\]

After setting up few-shot datasets following the steps above, the following folders will exist in your `data_path`:
- `CUB_fewshot_cropped`: 130/20/50 classes for train/validation/test, using bounding-box cropped images as input
- `cars`: 130/17/49 classes for train/validation/test
- `dogs`: 70/20/30 classes for train/validation/test
- `meta_iNat`: 908/227 classes for train/test. <!-- Holds softlinks to images in `inat2017_84x84` -->
- `tiered_meta_iNat`: 781/354 classes for train/test, split by superclass. <!-- Holds softlinks to images in `inat2017_84x84`  -->

Under each folder, images are organized into `train`, `val`, and `test` folders. In addition, you may also find folders named `val_pre` and `test_pre`, which contain validation and testing images pre-resized to 84x84 for the sake of speed.

## Train

* To train Bi-FRN on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/ECFRN/Conv-4
  ./train.sh
  ```

* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/ECFRN/ResNet-12
  ./train.sh
  ```

## Test

```shell
    cd experiments/CUB_fewshot_cropped/ECFRN/Conv-4
    python ./test.py
    
    cd experiments/CUB_fewshot_cropped/ECFRN/ResNet-12
    python ./test.py
```

## References

Thanks to  [Davis](https://github.com/Tsingularity/FRN), [Phil](https://github.com/lucidrains/vit-pytorch),  [Yassine](https://github.com/yassouali/SCL) and [Jijie](https://github.com/PRIS-CV/Bi-FRN) for the preliminary implementations.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:

- xinrongwang@email.ncu.edu.cn
