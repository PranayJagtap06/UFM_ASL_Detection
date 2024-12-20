import matplotlib.pyplot as plt
import streamlit as st
import torchvision
import builtins
import logging
import mlflow
import base64
import random
import torch
import os
from PIL import Image
from torchvision import transforms
from typing import List, Optional, Tuple


torch.serialization.add_safe_globals([torchvision.transforms._presets.ImageClassification, torchvision.transforms.functional.InterpolationMode, builtins.set])


# Setup logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Setup page.
about = """This is a basic Image Classification model used to detect American Sign Language (ASL). The model leverages *EfficientNet_B0* pytorch model trained on 87,000 ASL images. This app is build in association with *Unified Mentor* for machine learning project submition. I'm thankful to *Unified Mentor* to provide this platform."""

st.set_page_config(page_title="American Sign Language Detector",
                   page_icon="👌👍🤚", menu_items={"About": f"{about}"})
st.title(body="Hard Understanding What Your Deaf Friend Is Saying? Here! Use this ASL Detector Tool👇")
st.markdown("*Assumes your friend uses American Sign Language for communication.*")


# Initialize session state
if 'effnet_transform' not in st.session_state:
    st.session_state.effnet_transform = None
if 'pt_model' not in st.session_state:
    st.session_state.pt_model = None
if 'test_images' not in st.session_state:
    st.session_state.test_images = None
if 'test_img' not in st.session_state:
    st.session_state.test_img = None
if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None
if 'img' not in st.session_state:
    st.session_state.img = None


# Setting mlflow tracking uri.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))


# Model loading function.
@st.cache_resource
def load_model(model_name: str) -> Optional[torch.nn.Module]:
    try:
        model_uri = f"models:/{model_name}@champion"
        pt_model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=torch.device('cpu'))
        return pt_model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f":red[Failed to load *{model_name}* model. Please try again later.]", icon="🚨")
        return None


# EfficientNet transform loading function.
@st.cache_data
def load_effnet_transform(artifact_path: str) -> Optional[torch.Tensor]:
    try:
        effnet_transform_path = mlflow.artifacts.download_artifacts(artifact_path=artifact_path.split('/')[-1], run_id=artifact_path.split('/')[-3], tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
        effnet_transform = torch.load(effnet_transform_path, weights_only=True)
        return effnet_transform
    except Exception as e:
        logger.error(f"Error loading effnet_transform: {str(e)}")
        st.error(f":red[Failed to load *EfficientNet_B0* transform. Please try again later.]", icon="🚨")
        return None


# Test images loading function.
@st.cache_data
def gather_test_imgs(root_directory: str) -> List[Image.Image]:
    test_images = []
    try:
        for root, dirs, files in os.walk(root_directory):
            for file in files:
                full_path = os.path.join(root, file)
                logger.info(f"Found file: {full_path}")
                test_images.append(full_path)
        return test_images
    except Exception as e:
        logger.error(f"Error gathering test images: {str(e)}")
        st.error(f":red[Failed to gather test images. Please try again later.]", icon="🚨")
        return None


# Profile image loader.
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Load model.
if 'pt_model' not in st.session_state or st.session_state.pt_model is None:
    with st.spinner(f":green[This may take a while... Loading *PyTorch ASL Detector* model... ]"):
        st.session_state.pt_model = load_model(model_name="asl_pytorch")


# Load efficientnet transform
if 'effnet_transform' not in st.session_state or st.session_state.effnet_transform is None:
    with st.spinner(f":green[This may take a while... Loading *EfficientNet_B0 Transform*...]"):
        st.session_state.effnet_transform = load_effnet_transform("mlflow-artifacts:/8f10cd84b0b84b94867b0022d2af76dc/a70ba2892500440c858569f11fec1a9b/artifacts/effnetb0_transform.pt")


# Load test images
if 'test_images' not in st.session_state or st.session_state.test_images is None:
    with st.spinner(f":green[This may take a while... Loading test images...]"):
        st.session_state.test_images = gather_test_imgs(root_directory="./assets/asl_alphabet_test")


## Main Function ##
# class_names
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I",	"J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: transforms = None) -> None:

    # 2. Open image
    img = Image.open(image_)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ###

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image)

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    if isinstance(image_, str):
        true_class = image_.split('/')[-1][:-9]
    else:
        true_class = image_.name.split("/")[-1][:-9]
        st.write(image_.name)
    pred_class = class_names[target_image_pred_label]
    plt.figure()
    st.markdown("""---""")
    st.header("Classification Result ✨✨✨")
    st.markdown(f"*Your Friend is saying the letter **{pred_class}***.")
    plt.imshow(img)
    if true_class == pred_class:
        plt.title(f"True: {true_class} | Pred: {pred_class} | Prob: {target_image_pred_probs.max():.3f}", c='g')
    else:
        plt.title(f"True: {true_class} | Pred: {pred_class} | Prob: {target_image_pred_probs.max():.3f}", c='r')
    plt.axis(False);

    # Use Streamlit to display the plot
    st.pyplot(plt)


def display_test_images_grid(test_images_list: List, test_img: str):
    # Create a session state variable to control expander
    if 'expand_test_images' not in st.session_state or st.session_state.expand_test_images:
        st.session_state.expand_test_images = False

    # Create an expander for test images
    with st.expander("OR Select from Test Images", expanded=st.session_state.expand_test_images):
        st.markdown("**Hint:** Remove uploaded image, if any, before selecting from test images.")

        # Calculate number of columns (adjust as needed)
        num_columns = 4
        
        # Create grid of columns
        grid_columns = st.columns(num_columns)
        
        # Track current column index
        col_index = 0
        
        # Iterate through test images
        for img_path in test_images_list:
            # Open the image
            try:
                img = Image.open(img_path)
                
                # Use the current column in the grid
                with grid_columns[col_index]:
                    # Create a button with the image
                    if st.button(label=os.path.basename(img_path), 
                                 key=f"test_img_{img_path}"):
                        # Return the selected image path when button is clicked
                        test_img = img_path
                        st.session_state.expand_test_images = False
                    
                    # Display the image
                    st.image(img, use_column_width=True, output_format='PNG')
                
                # Move to next column, wrap around if needed
                col_index = (col_index + 1) % num_columns
            except Exception as e:
                st.error(f"Error loading image {img_path}: {e}")
                return None

        if not test_img:
            return random.sample(test_images_list, k=1)[0]
    
    return test_img


# Image Selection Column
st.markdown("""---""")
st.header("Upload an Image 📷")
upload_cols = st.columns([2, 1])
with upload_cols[0]:
    # Option 1: File uploader
    uploaded_img = st.file_uploader("Choose a png or jpg file", type=["png", "jpg"])

    # Option 2: Test image selection
    test_img = display_test_images_grid(st.session_state.test_images, st.session_state.test_img) 

st.session_state.test_img = uploaded_img if uploaded_img else test_img

try:
    img = Image.open(st.session_state.test_img)
    upload_cols[1].image(img)
except Exception as e:
    logger.error(f"Error processing uploaded image: {str(e)}")
    st.error("Failed to process the uploaded image. Please try again with a different image.")

classify_btn = upload_cols[0].button(":red[Detect Sign]")

# classify_btn click event
if classify_btn:
    with st.spinner(":blue[Classifying...]"):
        pred_and_plot_image(st.session_state.pt_model, st.session_state.test_img, class_names, (244, 244), st.session_state.effnet_transform)


# Disclamer
st.write("\n"*3)
st.markdown("""----""")
st.write("""*Disclamer: Predictions made by the models may be inaccurate due to the nature of the models. This is a simple demonstration of how machine learning can be used to make predictions. For more accurate predictions, consider using more complex models and larger datasets.*""")
    
st.markdown("""---""")
st.markdown("Created by [Pranay Jagtap](https://pranayjagtap.netlify.app)")

# Get the base64 string of the image
img_base64 = get_image_base64("assets/pranay_sq.jpg")

# Create the HTML for the circular image
html_code = f"""
<style>
    .circular-image {{
        width: 125px;
        height: 125px;
        border-radius: 55%;
        overflow: hidden;
        display: inline-block;
    }}
    .circular-image img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
</style>
<div class="circular-image">
    <img src="data:image/jpeg;base64,{img_base64}" alt="Pranay Jagtap">
</div>
"""

# Display the circular image
st.markdown(html_code, unsafe_allow_html=True)
# st.image("assets/pranay_sq.jpg", width=125)
st.markdown("Electrical Engineer | Machine Learning Enthusiast"\
            "<br>📍 Nagpur, Maharashtra, India", unsafe_allow_html=True)
