### Fast AI Learnings

Welcome to my blog on the Fast AI course. In here I will post my learnings as I go through the course.

As an introduction, the FastAI course is taught by **Jeremy Howard** - a highly experienced Australian computer scientist and one of the cofounders of FastAI. This course provides a highly hands on approach to deep learning, providing a plethora of theoretical concepts paired with implementation through their FastAI python library. Having only just scraped the surface of this course, this blog post will be primarily touching on the part 1 topics introduced.

### Neural Net Foundations

After walking through the required setup to cover the course in the *'Deployment'* section (I am using a slightly different setup harnessing docker instead of purely python and WSL), Jeremy begins to explore the foundations of `neural networks`. Fundamentally, a neural netowrk is a computing model capable of learning complex patterns and relationships for a provided set of data. A neural network can perform a variety of tasks, based on what it is designed to do. Initially, we will be focusing on classifiation - which is the application of a neural network to sorting input data (in this case images) into different catagories or classes based on the image 'content'. How this is done is that the neural network will be fed a large database of images which are presorted into classes. An example of images sorted into the CIFAR-10 classes is shown below:

![Example image set.](/images/image_set_example.PNG)

Now, when we are training our image models there are a number of different mathematical models that can be used. These image models come in a variety of different forms, represented by differening mathematical models, however we are primarily concerned with: how fast they are, how much memory they use, and how accurate they are for our application. Below shows a comparison of a variety of different training models **Jeremy** explored, where the y-axis represents the accuracy of the model and the x-axis represents the speed of the model (in seconds / sample)

![Training model speed](/images/newplot.png)

Ideally, you would want to select a model closest to the top left hand side of the of the chart - representing a fast (low time cost / sample) and highly accurate model. Changing between these models in Python can be very easy if you simply import the `PyTorch Image Models (timm)` library we can simply run:

```
timm.list_models('resnet*')
```
in order to bring up a list of the possible resnet models that can be used (simply replace this with another model name to search for other models), followed by passing this into our vision learner via:

```
learn = vision_learner(dls, 'resnet18', metrics=error_rate).to_fp16()
```
to specify the model to be used within the learner. 
