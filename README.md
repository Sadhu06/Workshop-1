## Workshop-1

## Adding Sunglasses to Your Passport Photo Using OpenCV
## NAME : SADHANA S
## REGISTER NUMBER : 212224230234

### PROGRAM AND OUTPUT

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread(r"C:\Users\admin\Downloads\mypic.jpeg")
plt.imshow(faceImage[:,:,::-1]); plt.title("Face")
```

<img width="407" height="588" alt="image" src="https://github.com/user-attachments/assets/35f617b4-c88b-45f7-b323-c4f38a12281b" />

```
#resized_faceImage.shape
faceImage.shape
```

<img width="260" height="49" alt="image" src="https://github.com/user-attachments/assets/fbe895c9-066a-4558-b2e1-4c106995b147" />

```
# Load the Sunglass image with Alpha channel
glassPNG = cv2.imread(r"C:\Users\admin\Downloads\sun.png", -1)
plt.imshow(glassPNG[:,:,::-1]); plt.title("glassPNG")
```

<img width="730" height="429" alt="image" src="https://github.com/user-attachments/assets/8ff6c9d1-1ed5-4ce3-a484-4ed4758b38b0" />


```
glassPNG.shape
```

<img width="193" height="46" alt="image" src="https://github.com/user-attachments/assets/54fd50f8-1521-40b2-b2ac-d65ffa933fe4" />

```
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG, (190, 50))
print("image Dimension = {}".format(glassPNG.shape))
```

<img width="392" height="62" alt="image" src="https://github.com/user-attachments/assets/60d68c2a-fba2-4154-9f8a-7fd554b41b3f" />

```
# Separate the Color channels
glassBGR = glassPNG[:,:,0:3]

# Create a STRICTER mask (anything lighter than 200 becomes black)
gray_glass = cv2.cvtColor(glassPNG, cv2.COLOR_BGR2GRAY)
_, glassMask1 = cv2.threshold(gray_glass, 200, 255, cv2.THRESH_BINARY_INV)

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121); plt.imshow(glassBGR[:,:,::-1]); plt.title('Sunglass Color channels')
plt.subplot(122); plt.imshow(glassMask1, cmap='gray'); plt.title('Sunglass Alpha channel (Created)')
```

<img width="1245" height="272" alt="image" src="https://github.com/user-attachments/assets/a2fab261-c0c4-4ef7-b71d-83e7fb1ba084" />

```
# Make a copy
faceWithGlassesNaive = faceImage.copy()

# 1. Make them slightly WIDER (Width: 220, Height: 55)
glassPNG_resized = cv2.resize(glassPNG, (220, 55))
glassBGR_resized = glassPNG_resized[:,:,0:3]

# 2. Move UP slightly and center (Y: 275 to 330, X: 315 to 535)
faceWithGlassesNaive[275:330, 315:535] = glassBGR_resized

plt.imshow(faceWithGlassesNaive[:,:,::-1])
```

<img width="454" height="578" alt="image" src="https://github.com/user-attachments/assets/f4eacbe9-285c-4eff-9248-74ec2ecd5a63" />

```
# 1. Resize the glasses to the new perfect size
glassPNG_resized = cv2.resize(glassPNG, (220, 55))
glassBGR_resized = glassPNG_resized[:,:,0:3]

# 2. Re-create the mask to match the new size
gray_glass = cv2.cvtColor(glassPNG_resized, cv2.COLOR_BGR2GRAY)
_, glassMask1_resized = cv2.threshold(gray_glass, 200, 255, cv2.THRESH_BINARY_INV)

# 3. Create the 3-channel mask
glassMask = cv2.merge((glassMask1_resized, glassMask1_resized, glassMask1_resized))
glassMask = np.uint8(glassMask/255)

# 4. Make a copy of the face and get the NEW Eye ROI
faceWithGlassesArithmetic = faceImage.copy()
# We use the new coordinates: Y: 275 to 330, X: 315 to 535
eyeROI = faceWithGlassesArithmetic[275:330, 315:535]

# 5. Blend them
maskedEye = cv2.multiply(eyeROI, (1 - glassMask))
maskedGlass = cv2.multiply(glassBGR_resized, glassMask)
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# 6. Display the results
plt.figure(figsize=[20,20])
plt.subplot(131); plt.imshow(maskedEye[:,:,::-1]); plt.title("Masked Eye Region")
plt.subplot(132); plt.imshow(maskedGlass[:,:,::-1]); plt.title("Masked Sunglass Region")
plt.subplot(133); plt.imshow(eyeRoiFinal[:,:,::-1]); plt.title("Augmented Eye and Sunglass")
```

<img width="1252" height="192" alt="image" src="https://github.com/user-attachments/assets/2c4fa1b7-a69e-4c19-be02-5ae5afd3f795" />

```
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[275:330, 315:535] = eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20])
plt.subplot(121); plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image")
plt.subplot(122); plt.imshow(faceWithGlassesArithmetic[:,:,::-1]); plt.title("With Sunglasses")
```

<img width="940" height="794" alt="image" src="https://github.com/user-attachments/assets/42d98d44-b68b-4a56-bcff-0a77818d8ec9" />

## RESULT

Therefore, The Sunglass experiment is implemented successfully.
