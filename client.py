# import requests

# # Ruta de la imagen a enviar
# image_path = "C:/Users/matel/Desktop/a1.png"

# # URL del servidor Flask
# url = "http://localhost:5000/predict"

# try:
#     # Abrir la imagen y enviar la solicitud
#     with open(image_path, "rb") as file:
#         files = {"file": file}
#         response = requests.post(url, files=files)

#     # Verificar el estado de la respuesta
#     if response.status_code == 200:
#         result = response.json()
#         print("Predicted Class:", result["predicted_class"])
#         print("Probabilities:")
#         for class_name, prob in result["probabilities"].items():
#             print(f"  {class_name}: {prob:.4f}")
#     else:
#         print(f"Error: {response.status_code}")
#         print(response.json())
# except Exception as e:
#     print(f"Error while sending the request: {str(e)}")
