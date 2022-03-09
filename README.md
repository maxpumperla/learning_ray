# Learning Ray - Flexible Distributed Python for Data Science

Jupyter notebooks and other resources for the upcoming book "Learning Ray" (O'Reilly).

The book is organized to guide you chapter by chapter from core concepts of Ray to more sophisticated topics along the way.
The first three chapters of the book teach the basics of Ray as a distributed Python framework with practical examples.
Chapters four to ten introduce Ray's high-level libraries and show how to build applications with them.
The last two chapters give you an overview of Ray's ecosystem and show you where to go next.
Here's what you can expect from each chapter.

![Ray Layers](https://raw.githubusercontent.com/maxpumperla/learning_ray/main/notebooks/images/chapter_01/ray_layers.png)

* [_Chapter 1, An Overview of Ray_](https://github.com/maxpumperla/learning_ray/blob/main/notebooks/ch_01_overview.ipynb)
  Introduces you at a high level to all of Ray's components, how it can be used in
  machine learning and other tasks, what the Ray ecosystem currently looks like and how
  Ray as a whole fits into the landscape of distributed Python.
* [_Chapter 2, Getting Started with Ray_](https://github.com/maxpumperla/learning_ray/blob/main/notebooks/ch_02_ray_core.ipynb)
  Walks you through the foundations of the Ray project, namely its low-level API.
  It also discussed how Ray Tasks and Actors naturally extend from Python functions and classes.
  You also learn about all of Ray's system components and how they work together.
* [_Chapter 3, Building Your First Distributed Application with Ray Core_](https://github.com/maxpumperla/learning_ray/blob/main/notebooks/ch_03_core_app.ipynb)
  Gives you an introduction to distributed systems and what makes them hard.
  We'll then build a first application together and discuss how to peak behind the scenes
  and get insights from the Ray toolbox.
* [_Chapter 4, Reinforcement Learning with Ray RLlib_](https://github.com/maxpumperla/learning_ray/blob/main/notebooks/ch_04_rllib.ipynb)
  Gives you a quick introduction to reinforcement learning and shows how Ray implements
  important concepts in RLlib. After building some examples together, we'll also dive into
  more advanced topics like preprocessors, custom models, or working with offline data.
* [_Chapter 5, Hyperparameter Optimization with Ray Tune_](https://github.com/maxpumperla/learning_ray/blob/main/notebooks/ch_05_tune.ipynb)
  Covers why efficiently tuning hyperparameters is hard, how Ray Tune works conceptually,
  and how you can use it in practice for your machine learning projects.
* [_Chapter 6, Distributed Training with Ray Train_](https://github.com/maxpumperla/learning_ray/blob/main/notebooks/ch_06_train.ipynb)
  Provides you with the basics of distributed model training and shows you how to use
  RaySGD with popular frameworks such as TensorFlow or PyTorch, and how to combine it
  with Ray Tune for hyperparameter optimization.
* [_Chapter 7, Data Processing with Ray_](https://github.com/maxpumperla/learning_ray/blob/main/notebooks/ch_07_data_processing.ipynb)
  Introduces you to the Dataset abstraction of Ray and how it fits into the landscape
  of other data structures. You will also learn how to bring pandas data frames, Dask
  data structures and Apache Spark workloads to Ray.
* _Chapter 8, Working with Ray Clusters_
  This chapter is all about how you configure, launch and scale Ray clusters for your applications.
  You'll learn about Ray's cluster launcher CLI and autoscaler, as well as how to set
  up clusters in the cloud and how to deploy on Kubernetes and other cluster managers.
* _Chapter 9, Serving Models with Ray Serve_
  Introduces you to model serving with Ray, why it works well within the framework,
  and how to do single-node and cluster deployment with it.
* _Chapter 10, Building A Machine Learning Application with Ray_
  Bringing together many of the previous chapters, in this chapter we'll build a machine
  learning in Ray end-to-end, from first experimentation to cloud deployment.
* _Chapter 11, The Ray Ecosystem_
  Comments on all the interesting extensions and third party integrations that Ray has
  attracted over the years.
* _Chapter 12, Conclusion_
  To conclude, we recap what you've learned in this book, where to go next to learn more
  about Ray, and what you can expect from Ray in the years to come.
