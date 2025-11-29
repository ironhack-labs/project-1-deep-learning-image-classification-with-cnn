
<h1>🧠 Deep Learning: Image Classification with CNN</h1>

<!-- Badges (images linked) -->
<p>
  <a href="https://github.com/IshuDhana/project-1-deep-learning-image-classification-with-cnn">
    <img alt="GitHub repo size" src="https://img.shields.io/badge/Repo-ImageClassify-blue" />
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue" />
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-orange" />
</p>

<!-- ===========================
     Short Project Description
     =========================== -->
<p>
  <strong>Project I</strong> — This project focuses on developing and deploying a Convolutional Neural Network (CNN) for image classification.
The model is designed to classify images from two possible datasets — CIFAR-10 or a custom 10-class animal dataset — into predefined categories.
</p>

<strong>DATA PROCESSING</strong> — 
Data loading and preprocessing (e.g., normalization, resizing, augmentation).
Create visualizations of some images, and labels..
</p>

<strong>🧠 Task Description</strong> — 
<ul>
  <li>Train and evaluate the model using CIFAR-10 or the 10-class animal dataset.</li>
  <li>Build a CNN-based image classifier.</li>
  <li>Apply transfer learning with pre-trained models (e.g., VGG16, Inception).</li>
   <li>Deploy the final model in a user-friendly interface.</li>
</ul>
</p>

<strong>🧬 Datasets</strong> — 
<ul>
  <li>We have used the following datasets:.</li><a href="[https://www.tensorflow.org/]https://www.cs.toronto.edu/~kriz/cifar.html"></a>
  <li>Option 1: CIFAR-10</li>
  <li>Images: 60,000 color images (32x32 pixels)</li>
   <li>Classes: 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)</li>
</ul>
</p>
<p>
<strong>⚙️ Key Components</strong> 
</p>

<strong>Objective:</strong> —
<ul>
  <li>Develop and deploy a deep learning image classification system using convolutional neural networks (CNNs)</li></a>
  <li>on the CIFAR-10 dataset to identify images across 10 categories such as airplanes, cars, and animals.</li>
</ul>
</p>

<strong>Models Created:</strong> —
<ul>
  <li>A custom baseline CNN built from scratch -</li></a>
  <li>A transfer learning model using VGG16 pretrained on ImageNet for improved accuracy.</li>
 <li>Additional experimental models for comparison.</li></a>
</ul>
</p>

<strong>Goals: </strong> —
<ul>
  <li>Train multiple models Used : Dataset 1</li></a>
  <li>evaluate performance metrics (accuracy, precision, recall, F1 score), and integrate the best-performing ones into an interactive platform.</li>
</ul>
</p>

<strong>Output: </strong> —
<ul>
  <li>A locally running Streamlit app that allows users to select a pre-trained model, upload an image, and receive instant classification predictions.</li>
</ul>
</p>


<strong>DATASET ON MODEL </strong> —
<ul>
  <li>Manageable size: It contains 60,000 color images (32×32 pixels), which is small enough to train models on a standard laptop without needing high-end hardware.</li>
 <li>Balanced and diverse: It includes 10 evenly distributed classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), testing your model’s ability to distinguish between different object types.</li>
 <li>Readily available: CIFAR-10 is built into TensorFlow and Keras, so it can be loaded easily with one command, reducing setup friction.</li>

</ul>
</p>

<strong> Data Preprocessing</strong> —
<ul>
  <li>Standardizing the image data — resizing, normalizing, and encoding — ensures models can learn patterns effectively and make accurate predictions on new inputs.</li></a>
  <li>Normalized pixel values by dividing all RGB pixel intensities by 255.0, converting them from the original 0–255 range to 0–1. This helps the neural network learn more efficiently and improves gradient stability during training.</li>
  <li>Converted images to numerical arrays so they could be processed by Keras.</li>
   <li>One-hot encoded the class labels (during training) so the model could output probabilities across the 10 CIFAR-10 categories.</li>
     <li>In the Demo app, the same preprocessing pipeline was used (image.load_img, image.img_to_array, normalization, and reshaping) to ensure uploaded images matched the format of your training data.
</li>
</ul>
</p>

<strong> Model Optimization</strong> —
<ul>
  <li>We experimented with different optimizers, including Adam (adaptive learning rate optimization) and Stochastic Gradient Descent (SGD), to identify which gave the best accuracy and convergence speed</li></a>
  <li>Learning rate tuning: By using adaptive optimizers like Adam (and potentially testing learning rate values), you controlled how quickly your model adjusted its weights.</li>
  <li>Dropout layers: You added Dropout after convolutional and dense layers (e.g., 0.25 and 0.5 rates) to prevent overfitting by randomly disabling neurons during training, improving model generalization.</li>
   <li>Early stopping: You implemented EarlyStopping with patience to halt training automatically when validation loss stopped improving, avoiding wasted epochs and overfitting.</li>
     <li>Transfer learning: The VGG16-based model reused pre-trained ImageNet weights, accelerating training and Should have improved accuracy compared to training from scratch
</li>
</ul>
</p>

<!-- ===========================
     Key Parameters
     =========================== -->
<h2 id="files">Parameters</h2>
<table>
  <tr><th>Parameters</th><th>Values</th></tr>
  <tr><td><code>Dataset</code></td><td>CIFAR-10</td></tr>
  <tr><td><code>Learning Rate</code></td><td>0.001</td></tr>
  <tr><td><code>Optimizer</code></td><td>Adam</td></tr>
  <tr><td><code>Epoch</code></td><td>50</td></tr>
 <tr><td><code>Batch Size</code></td><td>32</td></tr>
<tr><td><code>Accuracy Value</code></td><td>0.89</td></tr>
 <tr><td><code>Loss Value</code></td><td>0.40</td></tr>
</table>

<strong> Model Deployment </strong> —
<ul>
  <li>We used three model deployments :</li></a>
  <li><b>Gradio : </b>Utilized Gradio Interface for rapid model deployment</li>
 
  <li>Create a simple, shareable web interface for testing and showcasing the model using Gradio.</li>
  <li>Tested locally and then deployed via Google colab.</li>
   <li>Hosted the app via Gradio link created and tested in browser.</li>
   
  <li><b>Streamlit : </b>Deploy the trained image classification model through an interactive and user-friendly web interface using Streamlit.</li>
 
  <li>Loaded the saved model and preprocessing pipeline.</li>
  <li>Designed an upload interface for single/multiple image inputs.</li>
   <li>Processed images and displayed predictions with confidence levels.</li>
   <li>Hosted the app via Streamlit Local..</li>
</li>
</ul>
</p>

<strong> Transfer Learning</strong> —
<ul>
 <a> <li>Chose VGG16 because it brings strong feature extraction from large-scale ImageNet training, enabling our models to benefit from pre-learned visual patterns and improve performance on smaller or more specific datasets like our dataset in this exercise</li></a>
 
</ul>
</p>

<strong>Summary</strong> —
<ul>
  <li>In this project, we developed a Convolutional Neural Network (CNN) to classify images into defined categories using either the CIFAR-10 or Animals10 dataset</li></a>
  <li>In this project, we developed a Convolutional Neural Network (CNN) to classify images into defined categories using either the CIFAR-10 or Animals10 dataset.</li>
  <li>Training techniques matter: Optimizers, batch size, and early stopping help control overfitting and improve convergence.</li>
   <li>Finally, we documented results, insights, and deployed the best-performing model through a Flask application.</li>
   <li>Overall, this project provided hands-on experience with deep learning workflows, from data handling to model deployment, reinforcing both technical implementation and analytical evaluation skills..</li>
   
</ul>
</p>

 

<strong> Insights gained from the experimentation process.</strong> —
<ul>
  <li>Data preparation is critical: Proper normalization, augmentation, and visualization improve model robustness.</li></a>
  <li>Model design impacts accuracy: Layer selection and parameter tuning directly influence performance.</li>
 <li>Training techniques matter: Optimizers, batch size, and early stopping help control overfitting and improve convergence.
</li>
 <li>Evaluation metrics provide insight: Accuracy, precision, recall, F1-score, and confusion matrix highlight class-wise performance.</li>
 <li>Transfer learning saves time: Pre-trained models often achieve better accuracy with less data and training effort.</li>
  <li>Code clarity and documentation are essential: Well-structured code enables reproducibility and collaboration.</li>
 <li>Deployment completes the pipeline: Flask integration demonstrates practical use by allowing real-time image
predictions.</li>
</ul>
</p>



<!-- ===========================
     Features
     =========================== -->
<h2 id="features">Features</h2>
<ul>
  <li><strong>Model notebook:</strong> <code>G4_CNN_adam_model_keras.ipynb</code></li>
  <li><strong>Streamlit UI:</strong> <code>app.py</code> (finalized)</li>
  <li><strong>Gradio UI:</strong> <code>Gradio_deployment.ipynb</code> </li>
  <li><strong>Transfer learning experiments:</strong> VGG16, Inception</li>
</ul>

<!-- ===========================
     Key Files
     =========================== -->
<h2 id="files">Key Files</h2>
<table>
  <tr><th>Path</th><th>Description</th></tr>
  <tr><td><code>G4_CNN_adam_model_keras.ipynb</code></td><td>Main training & evaluation notebook</td></tr>
  <tr><td><code>app.py</code></td><td>Streamlit application for uploading images and predicting</td></tr>
  <tr><td><code>requirements.txt</code></td><td>Python dependencies for running Streamlit</td></tr>
  <tr><td><code>gradio_requirements.txt</code></td><td>Model Deployment with gradio</td></tr>
 <tr><td><code>Project_1_Deep_Learning_Image_Classification_with_CNN-2_transfer learning.ipynb</code></td><td>Transfer Learning</td></tr>
 <tr><td><code>Modeldataset2.ipynb</code></td><td>Sample model Test with Dataset_2</td></tr>
 <tr><td><code>Gradio_deployment.ipynb</code></td><td>Deploying in Gradio</td></tr>
 <tr><td><code>model_image_testing_screenshots</code></td><td>Screenshots taken while testing Model</td></tr>
 <tr><td><code>Graphs</code></td><td>Graphs and visualization</td></tr>
</table>

<!-- ===========================
     How to run (use Markdown fenced blocks for code)
     =========================== -->
<h2 id="usage">How to run</h2>

<p>Clone the repo and run the Streamlit app locally:</p>

```bash
git clone https://github.com/IshuDhana/project-1-deep-learning-image-classification-with-cnn.git
cd project-1-deep-learning-image-classification-with-cnn
pip install -r requirements.txt
streamlit run app.py
