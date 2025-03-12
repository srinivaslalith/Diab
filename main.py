import os

model_path = 'best_model_fold_1.keras'  # Update path if needed
if os.path.exists(model_path):
    print("Model file exists.")
else:
    print("Model file NOT found. Check the path.")

