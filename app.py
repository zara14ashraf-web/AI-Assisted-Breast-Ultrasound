import os

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from model import EfficientNetB3Classifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-Assisted Breast Ultrasound Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .footer {
        text-align: center;
        color: #8a989e;
        font-size: 0.75rem;
        line-height: 1.6;
        padding-top: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e5eaed;
    }

    [data-testid="stSidebar"] {
        background-color: #f7fafb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SAMPLES_DIR = os.path.join(
    BASE_DIR,
    "samples",
)

# Today's single EfficientNet-B3 checkpoint
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "Model_Best.pth",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "EfficientNet-B3"

IMAGE_SIZE = 300

# Decision threshold
THRESHOLD = 0.52

MEAN = [
    0.485,
    0.456,
    0.406,
]

STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# VERIFIED SAMPLE CASES
# ============================================================

SAMPLES = {

    "Sample 01 — Benign": {
        "id": "bus_0835-s",
        "image": "bus_0835-s.png",
        "mask": "bus_0835-s_MASK.png",
        "label": "Benign",
    },

    "Sample 02 — Benign": {
        "id": "bus_0090-l",
        "image": "bus_0090-l.png",
        "mask": "bus_0090-l_MASK.png",
        "label": "Benign",
    },

    "Sample 03 — Malignant": {
        "id": "bus_0650-r",
        "image": "bus_0650-r.png",
        "mask": "bus_0650-r_MASK.png",
        "label": "Malignant",
    },

    "Sample 04 — Malignant": {
        "id": "bus_0245-r",
        "image": "bus_0245-r.png",
        "mask": "bus_0245-r_MASK.png",
        "label": "Malignant",
    },
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ]
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_trained_model():

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"Model checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    model = EfficientNetB3Classifier()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        raise ValueError(
            "Unexpected checkpoint format."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    # Strict loading ensures that the deployed
    # architecture is exactly the trained architecture.
    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# IMAGE → TENSOR
# ============================================================

def image_to_tensor(image):

    return (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )


# ============================================================
# MASK → BBOX
# ============================================================

def bbox_from_mask(mask):

    mask_array = np.array(
        mask.convert("L")
    )

    mask_binary = (
        mask_array > 0
    )

    if not mask_binary.any():
        return None

    ys, xs = np.where(
        mask_binary
    )

    x1 = int(xs.min())
    x2 = int(xs.max())

    y1 = int(ys.min())
    y2 = int(ys.max())

    return [
        x1,
        y1,
        x2 - x1 + 1,
        y2 - y1 + 1,
    ]


# ============================================================
# MASK OVERLAY
# ============================================================

def create_mask_overlay(
    image,
    mask,
):

    image = image.convert("RGB")
    mask = mask.convert("L")

    if mask.size != image.size:

        mask = mask.resize(
            image.size
        )

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    mask_array = np.array(
        mask
    )

    mask_binary = (
        mask_array > 0
    )

    if not mask_binary.any():
        return image

    overlay = image_array.copy()

    overlay[
        mask_binary,
        0
    ] = 255

    overlay[
        mask_binary,
        1
    ] *= 0.35

    overlay[
        mask_binary,
        2
    ] *= 0.35

    result = (
        0.65 * image_array
        + 0.35 * overlay
    )

    result = np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        result
    )


# ============================================================
# GRAD-CAM++
# ============================================================

class GradCAMPlusPlus:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor,
        target_class,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            image_tensor
        )

        score = logits[
            0,
            target_class,
        ]

        score.backward()

        activations = self.activations
        gradients = self.gradients

        if (
            activations is None
            or gradients is None
        ):

            raise RuntimeError(
                "Grad-CAM++ activations or gradients "
                "were not captured."
            )

        gradients_2 = gradients.pow(2)

        gradients_3 = gradients.pow(3)

        sum_activations_gradients_3 = (
            gradients_3
            * activations
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        denominator = (
            2.0 * gradients_2
            + sum_activations_gradients_3
        )

        denominator = torch.where(
            denominator != 0.0,
            denominator,
            torch.ones_like(
                denominator
            ),
        )

        alpha = (
            gradients_2
            / denominator
        )

        positive_gradients = F.relu(
            gradients
        )

        weights = (
            alpha
            * positive_gradients
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0,
        ]

        cam = (
            cam.detach()
            .cpu()
            .numpy()
        )

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:
            cam /= maximum

        return cam

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

def get_gradcam_target_layer(
    model,
):

    # Final convolutional feature block
    return model.features[8]


# ============================================================
# GENERATE GRAD-CAM++
# ============================================================

def generate_gradcampp(
    model,
    image_tensor,
    predicted_class,
):

    target_layer = (
        get_gradcam_target_layer(
            model
        )
    )

    gradcam = GradCAMPlusPlus(
        model,
        target_layer,
    )

    try:

        cam = gradcam.generate(
            image_tensor,
            predicted_class,
        )

    finally:

        gradcam.remove_hooks()

    return cam


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    cam,
):

    image_resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    )

    image_array = np.array(
        image_resized
    ).astype(
        np.float32
    ) / 255.0

    cam_uint8 = (
        cam * 255
    ).astype(
        np.uint8
    )

    heatmap = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        dtype=np.float32,
    )

    heatmap[:, :, 0] = (
        cam_uint8
    )

    heatmap[:, :, 1] = (
        255 - cam_uint8
    ) * 0.5

    heatmap[:, :, 2] = (
        255 - cam_uint8
    )

    heatmap /= 255.0

    overlay = (
        0.55 * image_array
        + 0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1,
    )

    return Image.fromarray(
        (
            overlay * 255
        ).astype(
            np.uint8
        )
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(
    model,
    image,
):

    image_tensor = image_to_tensor(
        image
    )

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    benign_probability = (
        probabilities[0].item()
    )

    malignant_probability = (
        probabilities[1].item()
    )

    predicted_class = (
        1
        if malignant_probability >= THRESHOLD
        else 0
    )

    prediction = (
        "Malignant"
        if predicted_class == 1
        else "Benign"
    )

    return {
        "prediction": prediction,
        "predicted_class": predicted_class,
        "benign_probability": benign_probability,
        "malignant_probability": malignant_probability,
        "image_tensor": image_tensor,
    }


# ============================================================
# PROBABILITY DISPLAY
# ============================================================

def show_probability_distribution(
    benign_probability,
    malignant_probability,
):

    prob1, prob2 = st.columns(
        2,
        gap="large",
    )

    with prob1:

        st.metric(
            "Benign",
            f"{benign_probability * 100:.1f}%",
        )

        st.progress(
            float(benign_probability)
        )

    with prob2:

        st.metric(
            "Malignant",
            f"{malignant_probability * 100:.1f}%",
        )

        st.progress(
            float(malignant_probability)
        )

    st.caption(
        "These percentages represent the model's output "
        "distribution. They do not represent diagnostic "
        "accuracy, disease probability, or clinical certainty."
    )


# ============================================================
# MODEL SEPARATION
# ============================================================

def show_model_separation(
    benign_probability,
    malignant_probability,
):

    separation = abs(
        benign_probability
        - malignant_probability
    )

    if separation < 0.10:

        st.warning(
            "**Low model separation**"
        )

        st.caption(
            "The model assigns relatively similar output "
            "values to both classes."
        )

    elif separation < 0.20:

        st.warning(
            "**Moderate model separation**"
        )

        st.caption(
            "The model shows a preference for one class, "
            "but the alternative remains relatively close."
        )

    else:

        st.success(
            "**Clearer model separation**"
        )

        st.caption(
            "The model shows a more distinct difference "
            "between the two class outputs."
        )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_trained_model()

except Exception as error:

    st.error(
        "Unable to load the trained AI model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRANDING
    # --------------------------------------------------------

    st.markdown("## 🩺 Breast Ultrasound AI")

    st.caption(
        "AI-Assisted Imaging Analysis"
    )

    st.divider()

    # --------------------------------------------------------
    # QUICK OVERVIEW
    # --------------------------------------------------------

    st.markdown("### 🔍 System Overview")

    st.write(
        "This research prototype uses deep learning to "
        "classify breast ultrasound images as benign or "
        "malignant and provides a visual explanation of "
        "the model's prediction."
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "🧠 AI Model",
        expanded=True,
    ):

        st.markdown(
            "**EfficientNet-B3**"
        )

        st.caption(
            "A convolutional neural network used for "
            "binary breast lesion classification."
        )

        st.write(
            "**Classes:** Benign • Malignant"
        )

        st.write(
            "**Input:** Complete ultrasound image"
        )

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    with st.expander(
        "🔬 Explainable AI",
        expanded=False,
    ):

        st.markdown(
            "**Grad-CAM++**"
        )

        st.write(
            "The model's prediction is accompanied by a "
            "visual attention map highlighting image regions "
            "that contributed to the selected prediction."
        )

        st.caption(
            "Attention maps indicate model focus and should "
            "not be interpreted as definitive lesion segmentation."
        )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    with st.expander(
        "⚙️ How It Works",
        expanded=False,
    ):

        st.markdown(
            """
            **01 — Upload**

            Provide a breast ultrasound image.

            **02 — Analyze**

            The EfficientNet-B3 model processes the image.

            **03 — Prediction**

            The system provides a benign or malignant output
            with the model's probability distribution.

            **04 — Explanation**

            Grad-CAM++ visualizes regions associated with
            the selected prediction.
            """
        )

    # --------------------------------------------------------
    # INTERPRETING RESULTS
    # --------------------------------------------------------

    with st.expander(
        "📊 Understanding the Output",
        expanded=False,
    ):

        st.write(
            "**Prediction**"
        )

        st.caption(
            "The class selected by the model using the "
            "configured decision threshold."
        )

        st.write(
            "**Probability Distribution**"
        )

        st.caption(
            "Shows the model's relative output for the "
            "Benign and Malignant classes."
        )

        st.write(
            "**Model Separation**"
        )

        st.caption(
            "Indicates how distinctly the model's outputs "
            "favor one class over the other."
        )

    # --------------------------------------------------------
    # IMPORTANT NOTICE
    # --------------------------------------------------------

    st.divider()

    st.warning(
        "**Research & Educational Use Only**\n\n"
        "This AI system is a research prototype. Its "
        "predictions may be incorrect and should not be "
        "used as a standalone basis for clinical diagnosis "
        "or treatment decisions."
    )

    # --------------------------------------------------------
    # DEVELOPER
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 👩‍💻 Meet the Developer"
    )

    st.markdown(
        "**Zara Ashraf**"
    )

    st.caption(
        "Medical Imaging Technologist"
    )

    st.write(
        "Combining medical imaging knowledge with "
        "artificial intelligence to explore meaningful "
        "applications of AI in healthcare."
    )

    st.caption(
        "Designed & developed as an independent "
        "medical imaging AI project."
    )
# ============================================================
# HERO
# ============================================================

st.write("")

# Small identity label
st.caption(
    "AI-ASSISTED MEDICAL IMAGING"
)

# Main title
st.markdown(
    "# 🩺 Breast Ultrasound Analysis"
)

# Subtitle
st.markdown(
    "### Deep Learning • Classification • Explainable AI"
)

st.write("")

st.markdown(
    """
    Explore how artificial intelligence can assist in the
    analysis of breast ultrasound images through deep learning
    classification and visual model explanation.
    """
)

st.write("")

# ============================================================
# HERO HIGHLIGHTS
# ============================================================

hero1, hero2, hero3 = st.columns(
    3,
    gap="medium",
)

with hero1:

    with st.container(border=True):

        st.markdown(
            "#### 🧠 AI Model"
        )

        st.markdown(
            "**EfficientNet-B3**"
        )

        st.caption(
            "Deep learning architecture"
        )


with hero2:

    with st.container(border=True):

        st.markdown(
            "#### 🎯 Classification"
        )

        st.markdown(
            "**Benign • Malignant**"
        )

        st.caption(
            "Binary lesion classification"
        )


with hero3:

    with st.container(border=True):

        st.markdown(
            "#### 🔬 Explainability"
        )

        st.markdown(
            "**Grad-CAM++**"
        )

        st.caption(
            "Visual model attention"
        )

st.write("")

st.divider()
# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
    **Research & Educational Prototype**

    This application is designed for research and educational
    purposes only. AI predictions may be incorrect and should
    not be used as a standalone basis for diagnosis or treatment.
    Always rely on assessment by a qualified healthcare professional.
    """
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

with st.expander(
    "🧠 Model Overview",
    expanded=False,
):

    st.markdown(
        """
        This system uses a **single EfficientNet-B3 model** to
        analyze complete breast ultrasound images and classify
        them into two categories: **Benign** or **Malignant**.
        """
    )

    st.write("")

    overview1, overview2, overview3 = st.columns(
        3,
        gap="medium",
    )

    with overview1:

        with st.container(border=True):

            st.markdown(
                "#### 🧠 Architecture"
            )

            st.markdown(
                "**EfficientNet-B3**"
            )

            st.caption(
                "Deep convolutional neural network"
            )

    with overview2:

        with st.container(border=True):

            st.markdown(
                "#### 🎯 Classification"
            )

            st.markdown(
                "**Benign / Malignant**"
            )

            st.caption(
                "Binary lesion classification"
            )

    with overview3:

        with st.container(border=True):

            st.markdown(
                "#### 🔬 Explainability"
            )

            st.markdown(
                "**Grad-CAM++**"
            )

            st.caption(
                "Visualizes model attention"
            )

    st.write("")

    st.caption(
        "The model analyzes the complete ultrasound image rather "
        "than relying on a manually selected lesion crop."
    )
# ============================================================
# WHY WAS THIS DEVELOPED?
# ============================================================

with st.expander(
    "🔬 Why was this developed?",
    expanded=False,
):

    st.write(
        "As a Medical Imaging Technologist, I developed this "
        "project to explore how artificial intelligence can "
        "complement medical imaging workflows and support "
        "research in breast ultrasound analysis."
    )

    st.write(
        "Breast ultrasound lesions can demonstrate considerable "
        "variation in their visual appearance. This project "
        "investigates whether deep learning can learn useful "
        "visual patterns for differentiating benign and "
        "malignant breast lesions."
    )

    st.write(
        "The purpose is not to replace radiologists or medical "
        "imaging professionals. Instead, the system explores "
        "AI as an assistive research tool and provides a visual "
        "representation of model attention through Grad-CAM++."
    )


# ============================================================
# ANALYSIS AREA
# ============================================================

st.divider()

st.markdown(
    "### Analyze Breast Ultrasound"
)

st.caption(
    "Upload an ultrasound image or explore the representative "
    "BUS-BRA sample cases."
)

mode = st.radio(
    "Choose an analysis method",
    [
        "Upload Ultrasound",
        "Explore Sample Cases",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# UPLOAD MODE
# ============================================================

if mode == "Upload Ultrasound":

    st.write("")

    st.markdown(
        "#### Upload an ultrasound image"
    )

    st.caption(
        "Choose a breast ultrasound image for "
        "AI-assisted research analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload ultrasound",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        label_visibility="collapsed",
        help=(
            "Supported formats: PNG, JPG and JPEG. "
            "Remove patient-identifying information "
            "before uploading."
        ),
    )

    if uploaded_file is None:

        st.info(
            "PNG, JPG and JPEG images are supported. "
            "Please remove patient-identifying information "
            "before uploading."
        )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        width, height = image.size

        preview_col, info_col = st.columns(
            [2.3, 1],
            gap="large",
        )

        with preview_col:

            st.image(
                image,
                caption="Uploaded Ultrasound",
                use_container_width=True,
            )

        with info_col:

            st.markdown(
                "#### Image Information"
            )

            st.markdown(
                f"""
                <div style="
                    font-size: 1.35rem;
                    font-weight: 700;
                    color: #173f4d;
                    margin: 0.25rem 0 0.15rem 0;
                ">
                    {width} × {height} px
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "Image dimensions"
            )

            st.write("")

            st.markdown(
                "**Format**"
            )

            st.caption(
                "Ultrasound image"
            )

            st.write("")

            st.caption(
                "Ensure patient-identifying information "
                "has been removed before analysis."
            )

        st.write("")

        analyze = st.button(
            "🔍 Analyze Ultrasound",
            type="primary",
            use_container_width=True,
        )

        if analyze:

            try:

                with st.spinner(
                    "Analyzing ultrasound..."
                ):

                    result = predict(
                        model,
                        image,
                    )

                    prediction = result[
                        "prediction"
                    ]

                    predicted_class = result[
                        "predicted_class"
                    ]

                    benign_probability = result[
                        "benign_probability"
                    ]

                    malignant_probability = result[
                        "malignant_probability"
                    ]

                    image_tensor = result[
                        "image_tensor"
                    ]

                    cam = generate_gradcampp(
                        model,
                        image_tensor,
                        predicted_class,
                    )

                    gradcam_image = (
                        create_gradcam_overlay(
                            image,
                            cam,
                        )
                    )

                # ====================================================
                # AI ASSESSMENT
                # ====================================================

                st.divider()

                st.subheader(
                    "AI Assessment"
                )

                if prediction == "Benign":

                    st.success(
                        "### Model Prediction: BENIGN"
                    )

                else:

                    st.error(
                        "### Model Prediction: MALIGNANT"
                    )

                show_model_separation(
                    benign_probability,
                    malignant_probability,
                )

                with st.expander(
                    "View model probability distribution",
                    expanded=False,
                ):

                    show_probability_distribution(
                        benign_probability,
                        malignant_probability,
                    )

                # ====================================================
                # VISUAL EXPLANATION
                # ====================================================

                st.write("")

                st.subheader(
                    "Visual Explanation"
                )

                st.caption(
                    "Grad-CAM++ highlights image regions associated "
                    "with the model's selected prediction."
                )

                visual1, visual2 = st.columns(
                    2,
                    gap="medium",
                )

                with visual1:

                    st.image(
                        image,
                        caption="Original Ultrasound",
                        use_container_width=True,
                    )

                with visual2:

                    st.image(
                        gradcam_image,
                        caption="Grad-CAM++ Attention",
                        use_container_width=True,
                    )

                st.info(
                    "The highlighted Grad-CAM++ regions represent "
                    "model attention and are not a definitive lesion "
                    "segmentation."
                )

                # ====================================================
                # TECHNICAL NOTE
                # ====================================================

                with st.expander(
                    "View analysis details",
                    expanded=False,
                ):

                    st.write(
                        "**Input strategy:** "
                        "Complete ultrasound image"
                    )

                    st.write(
                        f"**Input resolution:** "
                        f"{IMAGE_SIZE} × {IMAGE_SIZE}"
                    )

                    st.write(
                        f"**Decision threshold:** "
                        f"{THRESHOLD}"
                    )

                    st.write(
                        "**Architecture:** "
                        "EfficientNet-B3"
                    )

                    st.write(
                        "**Grad-CAM++ target:** "
                        "features[8]"
                    )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)


# ============================================================
# SAMPLE CASE MODE
# ============================================================

else:

    st.write("")

    st.caption(
        "Explore the four representative BUS-BRA sample cases."
    )

    sample_names = list(
        SAMPLES.keys()
    )

    if "selected_sample" not in st.session_state:

        st.session_state.selected_sample = (
            sample_names[0]
        )

    st.markdown(
        "#### Representative Sample Cases"
    )

    st.caption(
        "Reference lesion masks are shown for visualization "
        "and dataset comparison only."
    )

    sample_cols = st.columns(
        4,
        gap="medium",
    )

    for i, sample_name in enumerate(
        sample_names
    ):

        with sample_cols[i]:

            if st.button(
                f"Sample {i + 1}",
                key=f"sample_button_{i}",
                use_container_width=True,
            ):

                st.session_state.selected_sample = (
                    sample_name
                )

    selected_sample_name = (
        st.session_state.selected_sample
    )

    sample = SAMPLES[
        selected_sample_name
    ]

    image_path = os.path.join(
        SAMPLES_DIR,
        sample["image"],
    )

    mask_path = os.path.join(
        SAMPLES_DIR,
        sample["mask"],
    )

    if not os.path.exists(
        image_path
    ):

        st.error(
            f"Sample image not found: {sample['image']}"
        )

        st.stop()

    if not os.path.exists(
        mask_path
    ):

        st.error(
            f"Sample mask not found: {sample['mask']}"
        )

        st.stop()

    image = Image.open(
        image_path
    ).convert("RGB")

    mask = Image.open(
        mask_path
    ).convert("L")

    # ------------------------------------------------------------
    # MASK VISUALIZATION ONLY
    # ------------------------------------------------------------

    mask_overlay = create_mask_overlay(
        image,
        mask,
    )

    # ------------------------------------------------------------
    # MODEL ANALYSIS
    # ------------------------------------------------------------

    try:

        result = predict(
            model,
            image,
        )

        prediction = result[
            "prediction"
        ]

        predicted_class = result[
            "predicted_class"
        ]

        benign_probability = result[
            "benign_probability"
        ]

        malignant_probability = result[
            "malignant_probability"
        ]

        image_tensor = result[
            "image_tensor"
        ]

        cam = generate_gradcampp(
            model,
            image_tensor,
            predicted_class,
        )

        gradcam_image = create_gradcam_overlay(
            image,
            cam,
        )

    except Exception as error:

        st.error(
            "Sample analysis failed."
        )

        st.exception(error)

        st.stop()

    # ============================================================
    # CASE HEADER
    # ============================================================

    st.write("")

    st.markdown(
        f"### {selected_sample_name}"
    )

    st.caption(
        f"Case ID: `{sample['id']}`"
    )

    # ============================================================
    # VISUAL EXPLANATION
    # ============================================================

    st.write("")

    st.subheader(
        "Visual Explanation"
    )

    st.caption(
        "Compare the original ultrasound, reference lesion mask, "
        "and Grad-CAM++ attention."
    )

    visual1, visual2, visual3 = st.columns(
        3,
        gap="medium",
    )

    with visual1:

        st.image(
            image,
            caption="Original Ultrasound",
            use_container_width=True,
        )

    with visual2:

        st.image(
            mask_overlay,
            caption="Reference Lesion Mask",
            use_container_width=True,
        )

    with visual3:

        st.image(
            gradcam_image,
            caption="Grad-CAM++ Attention",
            use_container_width=True,
        )

    st.caption(
        "The reference mask is provided for dataset visualization. "
        "Grad-CAM++ represents model attention and should not be "
        "interpreted as definitive lesion segmentation."
    )

    # ============================================================
    # AI ASSESSMENT
    # ============================================================

    st.write("")

    assessment_col, reference_col = st.columns(
        [1.35, 1],
        gap="large",
    )

    with assessment_col:

        st.subheader(
            "AI Assessment"
        )

        if prediction == "Benign":

            st.success(
                "Model prediction: **BENIGN**"
            )

        else:

            st.error(
                "Model prediction: **MALIGNANT**"
            )

        show_model_separation(
            benign_probability,
            malignant_probability,
        )

        with st.expander(
            "View model probability distribution",
            expanded=False,
        ):

            show_probability_distribution(
                benign_probability,
                malignant_probability,
            )

    with reference_col:

        st.subheader(
            "Reference Information"
        )

        st.caption(
            "DATASET REFERENCE"
        )

        st.markdown(
            f"**{sample['label']}**"
        )

        st.caption(
            "CASE ID"
        )

        st.markdown(
            f"**{sample['id']}**"
        )

        if prediction == sample["label"]:

            st.success(
                "Model prediction matches the dataset reference."
            )

        else:

            st.warning(
                "Model prediction differs from the dataset reference."
            )

        st.caption(
            "Reference information is displayed only "
            "for dataset comparison."
        )


# ============================================================
# TECHNICAL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "Technical Model Information"
):

    technical1, technical2 = st.columns(
        2
    )

    with technical1:

        st.write(
            "**Architecture:** "
            "EfficientNet-B3"
        )

        st.write(
            "**Task:** "
            "Binary breast lesion classification"
        )

        st.write(
            "**Classes:** "
            "Benign and Malignant"
        )

        st.write(
            f"**Input resolution:** "
            f"{IMAGE_SIZE} × {IMAGE_SIZE}"
        )

    with technical2:

        st.write(
            f"**Decision threshold:** "
            f"{THRESHOLD}"
        )

        st.write(
            f"**Inference device:** "
            f"{device}"
        )

        st.write(
            "**Explainability:** "
            "Grad-CAM++"
        )

        st.write(
            "**Grad-CAM++ target layer:** "
            "features[8]"
        )

        st.write(
            "**Model input:** "
            "Complete ultrasound image"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>AI-Assisted Breast Ultrasound Analysis</b><br>
        Breast Ultrasound Research Prototype<br><br>
        Developed by <b>Zara Ashraf</b><br>
        BS Medical Imaging Technology
    </div>
    """,
    unsafe_allow_html=True,
)
