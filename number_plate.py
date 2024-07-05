import cv2
import os
import easyocr
import pandas as pd
from datetime import datetime

# Path to Haar cascade for number plate detection
haarcascade = "D:/Python/Python Projects/Number Plate Detection System/haarcascade_russian_plate_number.xml"

# Create the directory to save the images if it doesn't exist
save_dir = "plates"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Minimum area for detected plates to consider
min_area = 500  # Adjust as per your requirement
count = 0

# Load the Haar cascade
plate_cascade = cv2.CascadeClassifier(haarcascade)

# Initialize OCR reader
reader = easyocr.Reader(['en'])

img_roi = None  # Initialize img_roi to None
ready_to_save = False  # Flag to indicate when an image is ready to be saved

# Path to the output Excel file
output_excel_path = "D:/Python/Python Projects/Number Plate Detection System/Detected_Number_Plates.xlsx"

# Check if the Excel file already exists
if os.path.exists(output_excel_path):
    df = pd.read_excel(output_excel_path)
else:
    # Initialize DataFrame for storing number plates, dates, and times
    df = pd.DataFrame(columns=["Date", "Time", "Number Plate"])

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Preprocess the image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_gray = cv2.equalizeHist(img_gray)  # Equalize the histogram

    # Detect plates
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in plates:
        area = w * h
        aspect_ratio = w / float(h)

        # Filter detections based on area and aspect ratio
        if area > min_area and 1.5 < aspect_ratio < 3.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x: x + w]  # Crop to the exact part
            
            ready_to_save = True  # Set the flag to indicate an image is ready to be saved

    cv2.imshow("Result", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s") and ready_to_save:
        # Save the cropped image
        img_path = os.path.join(save_dir, f"scanned_img_{count}.jpg")
        cv2.imwrite(img_path, img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)  # Wait for a short period to show the "Plate Saved" message

        # Perform OCR on the saved image
        img_saved = cv2.imread(img_path)
        if img_saved is not None:
            img_gray_saved = cv2.cvtColor(img_saved, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast to improve text extraction
            img_gray_saved = cv2.equalizeHist(img_gray_saved)
            
            # Apply thresholding to focus on darker regions
            _, img_thresh = cv2.threshold(img_gray_saved, 150, 255, cv2.THRESH_BINARY_INV)
            
            result = reader.readtext(img_thresh)

            # Extract the detected plate number if available
            if result:
                plate_number = result[0][1]  # Extract the text detected by OCR
                # Filter out non-alphanumeric characters
                plate_number = ''.join(c for c in plate_number if c.isalnum())
                print(f"Detected Plate Number: {plate_number}")
                # Get the current date and time
                current_datetime = datetime.now()
                current_date = current_datetime.strftime("%Y-%m-%d")
                current_time = current_datetime.strftime("%H:%M:%S")
                # Create a new row as a DataFrame
                new_row = pd.DataFrame({"Date": [current_date], "Time": [current_time], "Number Plate": [plate_number]})
                # Concatenate the new row to the existing DataFrame
                df = pd.concat([df, new_row], ignore_index=True)
                # Save the DataFrame to the Excel file
                df.to_excel(output_excel_path, index=False)
            else:
                print("No plate number detected")
        else:
            print(f"Failed to read saved image from {img_path}")

        count += 1  # Increment the count after successful OCR
        ready_to_save = False  # Reset the flag after saving the image

cap.release()
cv2.destroyAllWindows()
