# introduction

[toc]

***\*1.1 VGG19\****

VGG19 is a deep convolutional neural network architecture that was introduced as part of the Visual Geometry Group (VGG) project at the University of Oxford. It was proposed by Karen Simonyan and Andrew Zisserman in their paper titled "Very Deep Convolutional Networks for Large-Scale Image Recognition," which was published in 2014. VGG19 is an extension of the earlier VGG16 model, and the "19" in its name refers to the total number of layers, including both convolutional and fully connected layers.

VGG19 is a deep network with 19 layers, including 16 convolutional layers followed by three fully connected layers. It follows a simple and uniform architecture where each layer uses small 3x3 convolutional filters with a stride of 1, and the convolutional layers are stacked on top of each other.VGG19 uses 3x3 convolutional filters throughout the network. It avoids the need for larger filters that were commonly used in earlier networks.After each set of convolutional layers, VGG19 employs max-pooling layers with a 2x2 window and a stride of 2. These pooling layers help in downsampling the spatial dimensions, reducing the computational load and capturing more abstract features.The architecture's depth, with multiple stacked convolutional layers, allows the network to learn hierarchical features from low-level to high-level patterns in the images. This depth contributes to the network's ability to learn more complex and discriminative representations.The last three layers of VGG19 consist of fully connected layers that progressively reduce the spatial dimensions and finally produce the class probabilities or predictions.VGG19 was trained on large-scale image classification tasks such as the ImageNet dataset, which contains millions of labeled images and thousands of object classes. During training, VGG19 utilizes the cross-entropy loss and is optimized using stochastic gradient descent (SGD) with momentum.

While VGG19 and similar deep architectures have been surpassed by more recent models like ResNet, Inception, and Transformer-based networks, VGG19 remains an important milestone in the development of deep learning models for computer vision tasks. Its simplicity and effectiveness have made it a popular choice for educational purposes and as a benchmark for comparing the performance of new models.

 ![image](https://github.com/tong0410/Image-classification-and-image-style-migration/blob/main/images/1.png)

 

***\*1.2 Style Transfer\****      

Based on the above discussion, although traditional image style transfer algorithms can honestly depict some specific image styles, they have certain limitation.Therefore, completely new algorithms are needed to remove these limitations. Thus, the field of neural network image style transfer emerged.

With the development of artificial intelligence, Gatys et al. pioneered a neural network-based image style transfer technique. The core principle of the algorithm is: iteratively optimizes the image by pre-training the VGG model, the purpose is to match the high-level abstract feature distribution of the content image and the style image, and then synthesize the stylized original content image by iterative optimization of the input random noise map.

 

# Methods

***\*2. CNN Neural Network\****

The choice of Convolutional Neural Network in this report, I implement one way to optimize the objective function on the basis of Visual Geometry Group(VGG). 

I used the feature space provided by the 16 convolutional and 5 pooling layers of the 19 layer VGG Network, and I do not use any of the fully connected layers. For convenience, I choose the pretrained VGG19. I extract number 0, 5, 10, 19, 28 of this neural network which are ‘conv1 1’, ‘conv2 1’, ‘conv3 1’, ‘conv4 1’ and ‘conv5 1’ of the original VGG19 Network.

I calculate the loss of this Neural Network. I divide the total loss into two parts: content loss and style loss. 

For content loss, the soft-margin SVM is trained by solving a quadratic programming problem, which is expressed in the dual form as follows:

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps2.jpg) 

I define the squared-error loss between the two feature representations. Fij  represents the jth output value of the ith feature map of the generated images. The content loss described the content difference between the original image and the generated image.

For style loss, I use Gram Matrix to calculated the similarity of two images texture for feature extraction: 

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps3.jpg) 

To generate a texture that matches the style of a given image, I use gradient descent from a white noise image to find another image that matches the style representation of the original image. And the formula below is similar to the formula which calculates the content loss. The difference is the weight of the formula. It is much more depending on the shape of the matrix(M) and channels(N), where M is the height times the width of the feature map:

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps4.jpg) 

As for total loss, I merge content loss and style loss together. At the same time, I set two parameters. In the research, I change different The ratio between the two to achieve the best image transfer effect. (I assume that α as 1, β as 100 to train the whole Neural Network at first.)

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps5.jpg) 

where α and β are the weighting factors for content and style reconstruction respectively.

 

#  **Experiment** **and Results**

***\*3.1 Neural Style Transfer\****

After preparing the model, start testing some hyperparameters for style transfer.

(1) The number of iterations

When I start the first fifty iterations. The style loss reaches 176.005219, and the content loss reaches 35.989944. Since the content image is adjusted and enlarged to be consistent with the style image, the clarity is reduced, and the content loss is less than the style loss. When the number of iterations reaches 100, the style loss is reduced to 64.118866, which is close to one-third of the previous data. Content loss reached 35.255096, a decrease of 0.7. After the 900 times iteration, both style and content losses stabilize. But the values of both were still changing slightly, so the iterations continued to 1500 in pursuit of more accurate results.

 

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps6.jpg) 

Figure 1: the VGG19 net code

 

(2) Content weight and style weight

Different weights result in different degrees of content or style loss. The content weight takes a value of 1, which is consistent with the official website. Values greater than 1 will result in increased content loss. The value of the style weight is 1,000,000 as the optimal value. Greater or less than 1,000,000 will result in content loss or increased style loss.

(3) Results 

Input and Output Here is the input for a portrait of a man as the content image.

 

Result 1

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps7.jpg) 

Figure 2: Style transfer result 1

 

 

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps8.jpg) 

Figure 3: Style transfer loss 1

 

The file of loss and code are in attachments.

 

Another results: 

 

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps9.jpg) 

Figure 4: Style transfer result 2

 

 

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps10.jpg) 

Figure 5: Style transfer loss 2

 

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps11.jpg) 

Figure 6: Style transfer result 3

 

![img](file:///C:\Users\32018\AppData\Local\Temp\ksohtml25072\wps12.jpg) 

Figure 7: Style transfer loss 3

 

 

 

The results show that the color style is successfully transferred to the portrait. A stylized image-like distortion appears on the edges of objects in portraits. The effect is outstanding.

 

# **Conclusion** 

In this report, I created application prospect of image style transfer technology, analyzed the pre-trained VGG19 model, calculated content loss, style loss and total loss with formulas, and proved the best training times and weights with experiments. VGG19's deep convolutional layers enable the separation of content and style information from two different images.  By using the feature maps of intermediate layers, NST successfully disentangles content and style representations, allowing for the creation of novel artistic images. The utilization of pre-trained VGG19 models simplifies the NST process, making it more accessible to users with limited computational resources. Fine-tuning VGG19 for NST is not required, reducing the need for large-scale training datasets. While NST with VGG19 can be computationally intensive, advancements in optimization techniques have facilitated real-time stylization on modern hardware, making the method practical for various applications. Overall, Neural Style Transfer on VGG19 showcases the intersection of cutting-edge deep learning and artistic expression.  It has become an influential technique for generating visually stunning and unique images, while also inspiring further research into combining AI and creativity. Finally, the test examples and results are presented. I found that the results were significant for 1500 iterations with a style weight of 1,000,000. At the same time, it also tests the difference of loss when landscape picture and figure picture are used as style picture and original picture respectively, and calculates the difference.

 

