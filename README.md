# captcha
Segmentation and recognition for captcha

Dataset for this project was taken from kaggle: https://www.kaggle.com/datasets/huthayfahodeb/captcha-dataset

On this link you can see only one code for solve this problem. This code uses pretrained resnet and mse loss?... There isn't any noise cleaning and segmentation, that i think should be for greater result.

So firstly i wanted to segment every captcha in 6 smaller images. Each small image should contain its own digit. I've applied unsupervised segmentation, especially Canny algorithm with some smoothing. It've provided me ~30% accuracy. So 30% of train images were segmented in 6 parts. But for my purpose it has to be more! I decided to learn model to expand this for more images. It turned out! ~75% of images that weren't segment into 6 parts (6 digits) were segmented by my model.

in 'unsupervised_segmentation.py' you can see how i apply smoothing and Canny algorithm after it. (result ~2400 correctly segmentated images) in 'my_models.py' you can see 'SimpleSeg' - model, based on encoder-decoder architecture in 'dataset_classes.py' you can see 'SegDataset' - Dataset for comfortable train and test with PyTorch and in 'train_seg.py' there is training loop for semantic segmentation with choosing some parameters for model

After that, i wanted to make model that will process every small image (every digit of captcha) independntly. Think, think, think... And understand that my wonderful model for semantic segmentation had learn to clean noises from captcha. So i made another model, that uses cleaned captcha to recognise digits on it with better result than without cleaning noise. I've reached 60% accuracy on all 6 digits (6 from 6 digits correctly predict).

in 'my_models.py' you can see 'RecognizerOnPretrainedSegm' - model. Uses base model to clean noises and then simple recognition architecture. in 'dataset_classes.py' you can see 'KaptchaDataset' - Dataset for comfortable train and test with PyTorch and in 'train_rec.py' there is training loop for captcha recognition with choosing some parameters for model.

Hope it can be useful for one who started his road in ML.
