def show_tensor_images(image_tensor, num_images=25):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
show_tensor_images(data)#data is a tensor with the shape of (60000,1,28,28) or  (60000,3,28,28) 
#if the data have a shape of (60000,28,28) just add a dimension before calling the function for Example:
show_tensor_images(only_0.reshape(5923,1,28,28))
