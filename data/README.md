# Dataset Notes

The project loads CIFAR-10 using `tensorflow.keras.datasets.cifar10.load_data()`.
On first execution, Keras downloads the official dataset into its local cache.
The `CIFAR10_Dataset_Info.csv` and `cifar10_labels.txt` files are reference files;
they are not the 60,000 CIFAR-10 images.
