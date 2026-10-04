import requests  #allow your model to access hf
from PIL import Image, ImageDraw   #process the image
from config import HF_API_KEY  
#hugging face model
API_URL ="https://router.huggingface.co/hf-inference/models/facebook/detr-resnet-50"
#api key ...access the url
headers={"Authorrization":f"Bearer {HF_API_KEY}"}
#input the image path
image_path=input("Enter the image name:")
#open the image with pillow
image=Image.open(image_path).convert("RGB")
#what object to detect
wanted_object=input("What do you want to detect?").lower()
#open the binary image and find the object
with open(image_path,"rb") as file:
    image_data=file.read()
#send the image to hf
response=requests.post(API_URL,headers=headers,data=image_data)
#get response
results=response.json()
#outline the detected object
draw=ImageDraw.Draw(image)
for object in results:
    object_name=object["label"].lower()
    confidence=object["score"]
    #check if this is the object the user requested
    if object_name==wanted_object:
        #draw a rectab=ngle around it
        box=object["box"]
        x1 = box["xmin"]
        y1 = box["ymin"]
        x2 = box["xmax"]
        y2 = box["ymax"]
        draw.rectangle([x1,y1,x2,y2],outline="blue",width=4)
        draw.text((x1,y1),f"{object_name} {confidence:.0%}",fill="white")
#save the image
image.save("results.jpg")
#show the image
image.show()
print("Done! Result saved as result.jpg")

        