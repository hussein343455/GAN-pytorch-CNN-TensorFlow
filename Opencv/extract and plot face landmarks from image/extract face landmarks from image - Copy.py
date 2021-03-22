import face_alignment
import cv2
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = cv2.imread('imege.jpg')
preds = fa.get_landmarks(input)
#convert into Tensor
landmarks=torch.Tensor(preds).permute(1,2,0)
