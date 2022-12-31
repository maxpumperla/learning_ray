# Learning Ray - Flexible Distributed Python for Machine Learning

-- _Max Pumperla, Edward Oakes, Richard Liaw_

Online version of "Learning Ray" (O'Reilly).
All code and diagrams used in the book are fully open-sourced, and you can find self-contained notebooks accompanying the book here for free.
You won't get the exact same reading experience as with the printed book, but you should get a good idea if the book is for you.
If you want to support this project and  buy the book, you can e.g. get it 
[directly from O'Reilly](https://www.oreilly.com/library/view/learning-ray/9781098117214/),
or [from Amazon](https://www.amazon.com/Learning-Ray-Flexible-Distributed-Machine/dp/1098117220/).
The book will be published in May 2023, but online formats should be available before that.


![Learning Ray](https://raw.githubusercontent.com/maxpumperla/learning_ray/main/notebooks/images/learning_ray.png)


## Overview

The book is organized to guide you chapter by chapter from core concepts of Ray to more sophisticated topics along the way.
The first three chapters of the book teach the basics of Ray as a distributed Python framework with practical examples.
Chapters four to ten introduce Ray's high-level libraries and show how to build applications with them.
The last two chapters give you an overview of Ray's ecosystem and show you where to go next.
Here's what you can expect from each chapter.

* [_Chapter 1, An Overview of Ray_](./ch_01_overview)
  Introduces you at a high level to all of Ray's components, how it can be used in
  machine learning and other tasks, what the Ray ecosystem currently looks like and how
  Ray as a whole fits into the landscape of distributed Python.
* [_Chapter 2, Getting Started with Ray_](./ch_02_ray_core)
  Walks you through the foundations of the Ray project, namely its low-level API.
  It also discussed how Ray Tasks and Actors naturally extend from Python functions and classes.
  You also learn about all of Ray's system components and how they work together.
* [_Chapter 3, Building Your First Distributed Application with Ray Core_](./ch_03_core_app)
  Gives you an introduction to distributed systems and what makes them hard.
  We'll then build a first application together and discuss how to peak behind the scenes
  and get insights from the Ray toolbox.
* [_Chapter 4, Reinforcement Learning with Ray RLlib_](./ch_04_rllib)
  Gives you a quick introduction to reinforcement learning and shows how Ray implements
  important concepts in RLlib. After building some examples together, we'll also dive into
  more advanced topics like preprocessors, custom models, or working with offline data.
* [_Chapter 5, Hyperparameter Optimization with Ray Tune_](./ch_05_tune)
  Covers why efficiently tuning hyperparameters is hard, how Ray Tune works conceptually,
  and how you can use it in practice for your machine learning projects.
* [_Chapter 6, Data Processing with Ray_](./ch_06_data_processing)
  Introduces you to the Dataset abstraction of Ray and how it fits into the landscape
  of other data structures. You will also learn how to bring pandas data frames, Dask
  data structures and Apache Spark workloads to Ray.
* [_Chapter 7, Distributed Training with Ray Train_](./ch_07_train)
  Provides you with the basics of distributed model training and shows you how to use
  RaySGD with popular frameworks such as TensorFlow or PyTorch, and how to combine it
  with Ray Tune for hyperparameter optimization.
* [_Chapter 9, Serving Models with Ray Serve_](./ch_08_model_serving)
  Introduces you to model serving with Ray, why it works well within the framework,
  and how to do single-node and cluster deployment with it.
* [_Chapter 9, Working with Ray Clusters_](./ch_09_script)
  This chapter is all about how you configure, launch and scale Ray clusters for your applications.
  You'll learn about Ray's cluster launcher CLI and autoscaler, as well as how to set
  up clusters in the cloud and how to deploy on Kubernetes and other cluster managers.
* [_Chapter 10, Getting Started with the Ray AI Runtime_](./ch_10_air)
  Introduces you to Ray AIR, a unified toolkit for your ML workloads that offers many
  third party integrations for model training or accessing custom data sources.
* [_Chapter 11, Ray's Ecosystem and Beyond_](./ch_11_ecosystem)
  Gives you an overview of the many interesting extensions and
  integrations that Ray has attracted over the years. 

```python

```
