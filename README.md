# Hand written digit classification

A neural network was built for handwritten digit classification. The following are the descriptions of the neural network:

**Dataset:** Download the MNIST data (60,000 training images and 10,000 testing images) from the data source, http://yann.lecun.com/exdb/mnist/. The MNIST dataset include the 10 digit labels, from 0 to 9. Each (grayscale) image is a 28 x 28 matrix of pixels, with values between 0 and 255. Each pixel is converted to a value in the interval [0, 1] by dividing by 255. The figure below shows an example of each digit from the MINST dataset.

![image](https://user-images.githubusercontent.com/38344621/146652628-69cd897d-8eaa-49e5-9cf3-d61f979a063e.png)

**Image data structure:** Since images are 2-dimensional matrices, we first flatten them into a vector 𝐱 ∈ R784 with dimensionality 𝑑 = 28 × 28 = 784. This is done by simply concatenating all of the rows of the images to obtain one long vector.

**Output labels:** Since the output labels are categorical values that denote the digits from 0 to 9, we need to convert them into binary (numerical) vectors, using one-hot encoding. Thus, the label 0 is encoded as 𝒆1 = (1,0,0,0,0,0,0,0,0,0)𝑇 ∈ R10, the label 1 as 𝒆2 = (0,1,0,0,0,0,0,0,0,0)𝑇 ∈ R10, and so on, and finally the label 9 is encoded as 𝒆10 = (0,0,0,0,0,0,0,0,0,1)𝑇 ∈ R10

**Input and output:** Each input image vector 𝐱 has a corresponding target response vector𝐲 ∈ {𝒆1, 𝒆2, ... , 𝒆10}.

**Neural network:** Three different neural network models were used: (i) a neural network with no hidden layer, (ii) a neural network with one hidden layer with 7 neurons, and (iii) a neural network with one hidden layer with 49 neurons. In all the models, the input layer has 𝑑 = 784 and the output layer has 10 neurons.

**Training:** Each neural network was trained for 10 epochs, using learning rate (/step size) 𝛼 = 0.25.

The following plot shows the performance comparison:

![image](https://user-images.githubusercontent.com/38344621/146652763-ed2e6037-2024-4cca-99e1-928387d5c8f0.png)

