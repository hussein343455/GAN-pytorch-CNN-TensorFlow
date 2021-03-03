# create an empty tensor with the shape of the images
x=torch.empty(0,152,152,3)
# add the image into the tensor
x=torch.cat((image.reshape(1,152,152,3),x),dim=0)
