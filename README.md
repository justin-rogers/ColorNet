# ColorNet
Status: archive.
Experimental project in image classification. This generates five separate models and compares their results on the cifar10 dataset.
All models are based on standard CNN architectures:
  {Red, green, blue} models: trained only on the {red, green, blue} color channel.
  All-color model: trained on all color channels.
  Fuser model (ColorNet): uses the red/green/blue monochrome models as input, fuses their predictions with a fully connected layer.
