Creating a complete Python program for a "smart-invoice-validator" involves several steps and integration of different libraries for OCR, data validation, and machine learning. Below is a simplified version of such a project. This example uses Tesseract for OCR, Pandas for data management, and scikit-learn for training a simple validation model. Comprehensive error handling and comments are included to ensure the program’s robustness and readability.

To implement this program, you need to have Python and the necessary libraries installed. You can install the additional libraries via pip:

```bash
pip install pytesseract opencv-python pandas scikit-learn
```

Also, make sure to have [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system.

Here's the basic structure of the implementation:

```python
import pytesseract
import cv2
import pandas as pd
from pytesseract import Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust the path as per your installation

def extract_text_from_image(image_path):
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unable to open.")

        # Convert image to gray scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to extract text
        text_data = pytesseract.image_to_data(gray_img, output_type=Output.DICT)
        
        logging.info("Text extraction successful.")
        return text_data
    
    except Exception as e:
        logging.error("Error during text extraction: %s", e)
        return None

def validate_invoice_data(text_data):
    if text_data is None:
        return None
    
    # Dummy implementation of validation logic
    # Example: Check if typical invoice fields are present
    required_fields = ["Invoice", "Date", "Total", "Due"]
    detected_fields = {word.lower() for word in text_data['text'] if word.strip() != ""}
    
    validation_results = {field: (field.lower() in detected_fields) for field in required_fields}
    
    logging.info("Validation results: %s", validation_results)
    return validation_results

def analyze_data(validation_results):
    if validation_results is None:
        return None
    
    # Dummy implementation using a simple rule-based decision
    # In a full implementation, train a machine learning model
    all_valid = all(validation_results.values())
    
    if all_valid:
        analysis = {"validity": "Valid", "confidence": 100}
    else:
        analysis = {"validity": "Invalid", "confidence": 50}
    
    logging.info("Analysis: %s", analysis)
    return analysis

def main():
    # Example image file path
    invoice_image = 'path/to/invoice.jpg'  # Replace with actual path

    # Extract text from the image
    text_data = extract_text_from_image(invoice_image)

    # Validate extracted data
    validation_results = validate_invoice_data(text_data)

    # Analyze the validation results
    analysis = analyze_data(validation_results)

    # Output results
    if analysis:
        logging.info("Invoice validation completed: %s", analysis)
    else:
        logging.error("Invoice validation failed.")

# Entry point for the script
if __name__ == '__main__':
    main()
```

### Key Points:

1. **OCR with Tesseract**: This program uses PyTesseract to extract text from images. Make sure you have Tesseract OCR installed on your machine. You might need to adjust the executable path based on your installation.

2. **Basic Validation**: A simple method checks for typical fields in invoices. This is a placeholder for more complex validation logic, potentially involving a machine learning model trained on invoice data.

3. **Error Handling**: The program includes error handling using try-except blocks to catch and log exceptions during image processing and data validation.

4. **Logging**: It uses Python’s logging module to report the journey through data extraction, validation, and analysis, making it easier to troubleshoot.

5. **Machine Learning (Optional Placeholder)**: For real scenarios, you would replace dummy validation logic with trained models from historical invoice data for reliable predictions.

Remember, this is a foundational script and more sophisticated features (like using specialized OCR models for field extraction, integration with databases for storage, or comprehensive ML models for validation) would require an expanded scope and additional resources.