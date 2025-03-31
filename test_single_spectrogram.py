import os
import torch
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Class names for reference
class_names = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']

# Class descriptions and medical implications
class_descriptions = {
    'Seizure': {
        'description': 'Abnormal, excessive electrical activity in the brain causing convulsions or sensory disturbances.',
        'implications': 'Indicates epileptic activity requiring immediate medical attention.',
        'characteristics': 'Sudden surge of electrical activity with characteristic frequency patterns.'
    },
    'GPD': {  # Generalized Periodic Discharges
        'description': 'Periodic discharges occurring in a generalized distribution across both hemispheres.',
        'implications': 'Often associated with encephalopathy, anoxic brain injury, or status epilepticus.',
        'characteristics': 'Periodic sharp waves or complexes occurring synchronously across the brain.'
    },
    'LRDA': {  # Lateralized Rhythmic Delta Activity
        'description': 'Rhythmic delta activity that is lateralized to one hemisphere.',
        'implications': 'May precede seizures or indicate focal cerebral dysfunction.',
        'characteristics': 'Rhythmic patterns of delta waves (1-3 Hz) predominant in one hemisphere.'
    },
    'Other': {
        'description': 'Patterns that don\'t fit into the other specific categories.',
        'implications': 'Further analysis may be needed to classify the specific type of activity.',
        'characteristics': 'Various patterns not matching standard categorizations.'
    },
    'GRDA': {  # Generalized Rhythmic Delta Activity
        'description': 'Rhythmic delta activity occurring synchronously across both hemispheres.',
        'implications': 'Often seen in metabolic encephalopathy, medication effects, or after seizures.',
        'characteristics': 'Continuous rhythmic delta waves (1-3 Hz) generalized across the brain.'
    },
    'LPD': {  # Lateralized Periodic Discharges
        'description': 'Periodic discharges occurring in a lateralized distribution, affecting one hemisphere.',
        'implications': 'Associated with acute focal brain injury, stroke, or focal status epilepticus.',
        'characteristics': 'Periodic sharp waves or complexes predominant in one hemisphere.'
    }
}

def load_model():
    """Load the trained EfficientNetB3 model"""
    try:
        # Import needed only for model definition
        from torchvision import models
        import torch.nn as nn
        
        # Create the model architecture (same as in training)
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=len(class_names))
        )
        
        # Load trained weights from final_model.pth
        model.load_state_dict(torch.load('final_model.pth', map_location=device))
        model = model.to(device)
        model.eval()
        
        print("Final model loaded successfully!")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_spectrogram(file_path):
    """Process a parquet spectrogram file into a tensor suitable for the model"""
    try:
        # Read parquet file
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Convert to numpy array
        spectrogram_array = df.values
        
        # Create grayscale image from spectrogram array
        # Normalize to 0-255 range
        img_min = spectrogram_array.min()
        img_max = spectrogram_array.max()
        if img_max > img_min:  # Avoid division by zero
            spectrogram_array = ((spectrogram_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            spectrogram_array = np.zeros_like(spectrogram_array, dtype=np.uint8)
        
        # Convert to PIL Image (grayscale)
        img = Image.fromarray(spectrogram_array)
        
        # Resize to 300x300
        img = img.resize((300, 300), Image.LANCZOS)
        
        # Convert grayscale to RGB (3 channels) by duplicating the single channel
        img_rgb = Image.new("RGB", img.size)
        img_rgb.paste(img)
        
        # Save the processed image for reference
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        img_rgb.save(os.path.join(output_dir, 'processed_spectrogram.png'))
        
        # Apply transformations (same as validation set)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img_rgb)
        return img_tensor, img_rgb
    
    except Exception as e:
        print(f"Error processing spectrogram: {e}")
        return None, None

def predict(model, img_tensor):
    """Make a prediction using the trained model"""
    try:
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted_class = torch.max(outputs, 1)
            
        return predicted_class.item(), probabilities.cpu().numpy()
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def visualize_results(img, predicted_class, probabilities, filename):
    """Visualize the spectrogram and the model's prediction with a detailed written report"""
    class_name = class_names[predicted_class]
    description = class_descriptions[class_name]
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension for output naming
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Create a larger figure with 3 areas: image, probabilities, and text report
    fig = plt.figure(figsize=(20, 10))
    
    # Define grid for the layout
    gs = plt.GridSpec(2, 3, figure=fig, height_ratios=[3, 1])
    
    # Plot the spectrogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Spectrogram Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Plot the prediction probabilities
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(range(len(class_names)), probabilities)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.set_ylabel('Probability')
    ax2.set_title('Class Prediction Probabilities', fontsize=14, fontweight='bold')
    
    # Highlight the predicted class
    bars[predicted_class].set_color('red')
    
    # Add a text box with the detailed report
    ax3 = fig.add_subplot(gs[0:, 2])
    ax3.axis('off')
    
    # Create the detailed text report
    report_text = f"FILE: {filename}\n\n"
    report_text += f"PREDICTION: {class_name}\n"
    report_text += f"Confidence: {probabilities[predicted_class]*100:.2f}%\n\n"
    report_text += f"DESCRIPTION:\n{description['description']}\n\n"
    report_text += f"MEDICAL IMPLICATIONS:\n{description['implications']}\n\n"
    report_text += f"CHARACTERISTICS:\n{description['characteristics']}\n\n"
    report_text += "PROBABILITIES:\n"
    for i, cls in enumerate(class_names):
        report_text += f"{cls}: {probabilities[i]*100:.2f}%\n"
    
    # Add the text box with report
    text_box = ax3.text(0, 0.95, report_text, fontsize=12, 
                      verticalalignment='top', 
                      bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8))
    
    # Add the main title
    fig.suptitle(f"EEG Spectrogram Analysis: {class_name}", fontsize=16, fontweight='bold', y=0.98)
    
    # Add a timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.02, 0.02, f"Generated: {timestamp}", fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for the title
    
    # Save results with unique filenames based on input file
    results_png = os.path.join(output_dir, f"{base_filename}_results.png")
    plt.savefig(results_png, dpi=300)
    
    # Print the detailed results to console as well
    print("\n" + "="*50)
    print(f"ANALYSIS FOR: {filename}")
    print(f"PREDICTION: {class_name}")
    print("="*50)
    print(f"Confidence: {probabilities[predicted_class]*100:.2f}%")
    print("\nDESCRIPTION:")
    print(description['description'])
    print("\nMEDICAL IMPLICATIONS:")
    print(description['implications'])
    print("\nCHARACTERISTICS:")
    print(description['characteristics'])
    print("\nPROBABILITIES:")
    for i, cls in enumerate(class_names):
        print(f"{cls}: {probabilities[i]*100:.2f}%")
    print("="*50)
    
    # Also save the results to a file
    report_txt = os.path.join(output_dir, f"{base_filename}_report.txt")
    with open(report_txt, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"ANALYSIS FOR: {filename}\n")
        f.write(f"PREDICTION: {class_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Confidence: {probabilities[predicted_class]*100:.2f}%\n\n")
        f.write("DESCRIPTION:\n")
        f.write(description['description'] + "\n\n")
        f.write("MEDICAL IMPLICATIONS:\n")
        f.write(description['implications'] + "\n\n")
        f.write("CHARACTERISTICS:\n")
        f.write(description['characteristics'] + "\n\n")
        f.write("PROBABILITIES:\n")
        for i, cls in enumerate(class_names):
            f.write(f"{cls}: {probabilities[i]*100:.2f}%\n")
        f.write("="*50)
    
    # Save a higher quality version for medical documentation
    pdf_file = os.path.join(output_dir, f"{base_filename}_report.pdf")
    plt.savefig(pdf_file)
    
    print(f"\nResults saved to:")
    print(f"- {results_png}")
    print(f"- {report_txt}")
    print(f"- {pdf_file}")
    
    plt.show()

def get_all_spectrograms():
    """Get all available spectrograms from both train and test directories"""
    spectrogram_dirs = ['train_spectrograms', 'test_spectrograms']
    all_files = {}
    
    for dir_name in spectrogram_dirs:
        if os.path.exists(dir_name):
            files = [f for f in os.listdir(dir_name) if f.endswith('.parquet')]
            if files:
                all_files[dir_name] = files
    
    return all_files

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Test EfficientNet model on a single spectrogram file.')
    parser.add_argument('--file', type=str, help='Filename of the spectrogram (e.g., 999896023.parquet)')
    parser.add_argument('--dir', type=str, choices=['train', 'test'], help='Directory containing the spectrogram (train or test)')
    args = parser.parse_args()
    
    # Get all available spectrograms
    all_spectrograms = get_all_spectrograms()
    
    if not all_spectrograms:
        print("No spectrogram directories found. Please make sure 'train_spectrograms' or 'test_spectrograms' exists.")
        return
    
    # If directory is specified via command line
    if args.dir:
        dir_name = f"{args.dir}_spectrograms"
        if dir_name not in all_spectrograms:
            print(f"Error: Directory '{dir_name}' not found or contains no parquet files.")
            return
        spectrogram_dir = dir_name
    else:
        # If no directory specified, ask user to select one
        available_dirs = list(all_spectrograms.keys())
        if len(available_dirs) == 1:
            spectrogram_dir = available_dirs[0]
            print(f"Using the only available directory: {spectrogram_dir}")
        else:
            print("Available spectrogram directories:")
            for i, dir_name in enumerate(available_dirs):
                print(f"  {i+1}. {dir_name} ({len(all_spectrograms[dir_name])} files)")
            choice = input("\nSelect directory (enter number): ")
            try:
                dir_index = int(choice) - 1
                if 0 <= dir_index < len(available_dirs):
                    spectrogram_dir = available_dirs[dir_index]
                else:
                    print("Invalid selection. Using 'train_spectrograms'.")
                    spectrogram_dir = 'train_spectrograms'
            except ValueError:
                print("Invalid input. Using 'train_spectrograms'.")
                spectrogram_dir = 'train_spectrograms'
    
    # Get the filename either from command line or user input
    if args.file:
        filename = args.file
    else:
        # Print first 5 files as examples
        print(f"\nFound {len(all_spectrograms[spectrogram_dir])} parquet files in {spectrogram_dir}. Examples:")
        for i, file in enumerate(all_spectrograms[spectrogram_dir][:5]):
            print(f"  {i+1}. {file}")
        
        # Ask user for input
        filename = input("\nEnter the filename (e.g., 999896023.parquet): ")
    
    # Ensure the filename has .parquet extension
    if not filename.endswith('.parquet'):
        filename += '.parquet'
    
    # Check if the file exists
    file_path = os.path.join(spectrogram_dir, filename)
    if not os.path.exists(file_path):
        print(f"Error: File '{filename}' not found in the {spectrogram_dir} directory!")
        return
    
    # Load the model
    model = load_model()
    if model is None:
        return
    
    # Process the spectrogram file
    img_tensor, img = process_spectrogram(file_path)
    if img_tensor is None:
        return
    
    # Make a prediction
    predicted_class, probabilities = predict(model, img_tensor)
    if predicted_class is None:
        return
    
    # Visualize and explain the results
    visualize_results(img, predicted_class, probabilities, file_path)

if __name__ == "__main__":
    main() 