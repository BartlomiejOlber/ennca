1.Create Dataset object to load LR and HR images, you can use just grids of images not the entire images, say you can crop 112X112 images from LR image and referencing HR image 224X224. 

2. Build Unet-type architecture using resnet as encoder and BiFPN as decoder, you should use Pixel shuffle at the end to upsample the output to super resolution target.
You should use only Y channel from YCbCr format to output Y channel super resolution output, then use linear interpolation to the original CbCr channel inputs, to form SR YCbCr and convert it back to RGB as the final output.
