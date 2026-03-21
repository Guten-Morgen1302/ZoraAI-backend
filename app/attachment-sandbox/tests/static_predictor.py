import os
import sys

# Add the attachment-sandbox root to sys.path so we can import 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.static_analysis.classifier import predict

# Replace with the path to your PDF
file_path = "C:\\D\\SL\\Hacks\\ZoraAI\\ZoraAI-backend\\app\\attachment-sandbox\\tests\\God's plan bytecamp.pdf"

# Run the prediction
probability, features = predict(file_path)

print(f"Risk Score: {probability:.2%}")
print("\nExtracted Features:")
for key, value in features.items():
    print(f"  {key}: {value}")
