import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

a = cv2.VideoCapture(0)

interpreter = tf.lite.Interpreter(model_path="cat_dog_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_labels = ['cats', 'dogs']

def prepare_image(image):
    test_image = cv2.resize(image, (224, 224))
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    
    return test_image

# Perform inference
def predict_image(interpreter, input_details, output_details, image):
    # Set the tensor to the input image
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

# Main loop
while True:
    ret, frame = a.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Prepare the image
    processed_image = prepare_image(frame)
    
    # Get prediction
    predictions = predict_image(interpreter, input_details, output_details, processed_image)
    
    # Get predicted class
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index] * 100
    if confidence > 98.0:
        label = f"{class_labels[predicted_class_index]}: {confidence:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    else:
        label = "none"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    
    # Show the frame
    cv2.imshow("Prediction", frame)
    
    # Break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
a.release()
cv2.destroyAllWindows()