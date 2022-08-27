#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import io
import cv2
import requests
import numpy as np
from IPython.display import Image, display


# In[2]:


# Commençons par mettre en place toutes ces informations:
base_url = 'http://localhost:8000'
endpoint = '/predict'
model = 'yolov3-tiny'


# In[3]:


url_with_endpoint_no_params = base_url + endpoint
url_with_endpoint_no_params


# In[4]:


full_url = url_with_endpoint_no_params + "?model=" + model
full_url


# In[5]:


def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed. False otherwise.

    Returns:
        requests.models.Response: Response from the server.
    """
    
    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response


# In[6]:


# Pour tester cette fonction, ouvrez un fichier dans votre système de fichiers et passez-le en paramètre à côté de l'URL :
with open("clock2.jpg", "rb") as image_file:
    prediction = response_from_server(full_url, image_file)


# In[7]:


# Bonne nouvelle! La demande a été acceptée. Cependant, vous n'obtenez aucune information sur les objets de l'image.
# Pour obtenir l'image avec les cadres de délimitation et les étiquettes, 
# vous devez analyser le contenu de la réponse dans un format approprié. 
# Ce processus ressemble beaucoup à la façon dont vous lisez des images brutes dans une image cv2 sur le serveur.
#-------------------------------------------------------------------------------------------------------------------------------
# Pour gérer cette étape, créons un répertoire appelé images_predicted pour enregistrer l'image dans :
dir_name = "images_predicted"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


# In[8]:


# Création de la fonction display_image_from_response
def display_image_from_response(response):
    """Display image within server's response.

    Args:
        response (requests.models.Response): The response from the server after object detection.
    """
    
    image_stream = io.BytesIO(response.content)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    filename = "image_with_objects.jpeg"
    cv2.imwrite(f'images_predicted/{filename}', image)
    display(Image(f'images_predicted/{filename}'))


# In[9]:


display_image_from_response(prediction)


# In[ ]:


# Vous êtes maintenant prêt à utiliser votre modèle de détection d'objets via votre propre client !
# Testons-le sur d'autres images :
image_files = [
    'car2.jpg',
    'clock3.jpg',
    'apples.jpg'
]

for image_file in image_files:
    with open(f"images/{image_file}", "rb") as image_file:
        prediction = response_from_server(full_url, image_file, verbose=False)
    
    display_image_from_response(prediction)

