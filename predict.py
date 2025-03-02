import cv2
import torch
import torchvision.transforms as transforms
from model import ISLModel


num_classes = 32  
model = ISLModel(num_classes)
model.load_state_dict(torch.load("isl_model.pth"))
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_img = transform(frame).unsqueeze(0)

    
    with torch.no_grad():
        output = model(input_img)
        prediction = output.argmax(dim=1).item()

    cv2.putText(frame, f"Prediction: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
