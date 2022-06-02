from PIL import Image
from facenet import Facenet
from Face_recognition import predict_image_for_server_2

if __name__ == "__main__":
    model = Facenet()
        
    probability = predict_image_for_server_2('img/JinShaoFei.jpg', model, 'loss')
    
    print(probability)
