## Zero-Shot Detection of Street Flooding in New York City
Large Vision Models (LVMs) are growing, yet untapped generators of high-quality annotated data. Here, we pair OpenAI’s Contrastive Language-Image Pretraining (CLIP) model and the in-preview GPT4-Vision model as zero-shot annotators of street flooding conditions in New York City. This project has two thrusts: one developing a new method for training high quality computer vision classifiers, and the other using said model to improve the state of the art in flood detection, which is imperative as urban flooding worsens with climate change. We also investigate correlations between crowd-sourced flooding reports from New York City 311 and flooding in our dashcam images. 

At a high level, CLIP is a multimodal model that connects text and images. The model consists of a text and image encoder, also used in generative image models like Stable Diffusion, which encodes textual and visual information into an embedding space. 

The training aim of the model is to increase the cosine similarity score of images and text which are asserted as associated, while decreasing the cosine similarity score between distant concepts. 

