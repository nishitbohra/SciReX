import streamlit as st
import os
import numpy as np
import cv2
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Human Pose Estimation",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the app's appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #0D47A1;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .keypoint {
        display: inline-block;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

class PoseEstimator:
    """Mock class to simulate the pose estimation model"""
    
    def __init__(self, model_type="baseline"):
        self.model_type = model_type
        # In a real implementation, load the actual model here
        self.keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", 
                         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                         "left_wrist", "right_wrist", "left_hip", "right_hip", 
                         "left_knee", "right_knee", "left_ankle", "right_ankle"]
        self.pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), 
                      (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
        
    def process_image(self, image, add_occlusion=False, occlusion_level=0.3):
        """Mock image processing to simulate model inference"""
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Resize image for processing
        img = cv2.resize(image_np, (256, 256))
        
        # Create simulated keypoints
        # In real implementation, this would be the model prediction
        height, width = img.shape[:2]
        keypoints = []
        
        # Generate sample keypoints based on image dimensions
        # This is just a mock - in reality, this would come from your model
        for i in range(17):
            x = int(np.random.uniform(0.2, 0.8) * width)
            y = int(np.random.uniform(0.2, 0.8) * height)
            confidence = np.random.uniform(0.5, 1.0)
            
            # Simulate lower confidence or missing keypoints for occlusion
            if add_occlusion and np.random.random() < occlusion_level:
                confidence = np.random.uniform(0.1, 0.4)
                
            keypoints.append((x, y, confidence))
            
        # Simulate GAN recovery for occluded keypoints if using the GAN model
        if add_occlusion and self.model_type == "gan":
            for i, kp in enumerate(keypoints):
                if kp[2] < 0.5:  # If confidence is low (occluded)
                    # Simulate GAN recovery by improving confidence
                    x, y, _ = kp
                    # Adjust position slightly to simulate correction
                    x += np.random.randint(-5, 6)
                    y += np.random.randint(-5, 6)
                    # Improved confidence
                    confidence = np.random.uniform(0.6, 0.9)
                    keypoints[i] = (x, y, confidence)
                    
        # Create a results dictionary
        results = {
            "keypoints": keypoints,
            "processing_time": np.random.uniform(0.05, 0.2),
            "confidence_scores": [kp[2] for kp in keypoints]
        }
        
        return results
    
    def visualize_pose(self, image, keypoints, threshold=0.5):
        """Draw keypoints and skeleton on the image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        vis_img = image.copy()
        height, width = vis_img.shape[:2]
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > threshold:
                color = (0, 255, 0)  # Green for high confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
                
            cv2.circle(vis_img, (int(x), int(y)), 4, color, -1)
            cv2.putText(vis_img, f"{i}", (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw skeleton
        for pair in self.pairs:
            p1, p2 = pair
            if keypoints[p1][2] > threshold and keypoints[p2][2] > threshold:
                cv2.line(vis_img, 
                        (int(keypoints[p1][0]), int(keypoints[p1][1])),
                        (int(keypoints[p2][0]), int(keypoints[p2][1])),
                        (0, 255, 255), 2)
                
        return vis_img
    
    def get_keypoint_names(self):
        """Return the names of the keypoints"""
        return self.keypoints

def add_occlusion_to_image(image, occlusion_level=0.3):
    """Add simulated occlusion to the image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    occluded_img = image.copy()
    h, w = image.shape[:2]
    
    # Number of occlusion rectangles based on level
    num_occlusions = int(occlusion_level * 10)
    
    for _ in range(num_occlusions):
        # Random rectangle
        rect_w = int(np.random.uniform(0.1, 0.3) * w)
        rect_h = int(np.random.uniform(0.1, 0.3) * h)
        x = np.random.randint(0, w - rect_w)
        y = np.random.randint(0, h - rect_h)
        
        # Fill with random color or black
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.rectangle(occluded_img, (x, y), (x + rect_w, y + rect_h), color, -1)
    
    return occluded_img

def get_image_download_link(img, filename, text):
    """Generate a link to download an image"""
    buffered = BytesIO()
    img = Image.fromarray(img)
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def get_confidence_color(confidence):
    """Return color based on confidence value"""
    if confidence > 0.7:
        return "green"
    elif confidence > 0.5:
        return "orange"
    else:
        return "red"

def main():
    # Sidebar for navigation
    st.sidebar.markdown("<h1 style='text-align: center;'>🧍 Pose Estimation</h1>", unsafe_allow_html=True)
    
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Home", "Demo", "Upload & Analyze", "Compare Models", "About"]
    )
    
    # Sidebar model selection
    if app_mode in ["Demo", "Upload & Analyze", "Compare Models"]:
        st.sidebar.markdown("## Model Settings")
        model_type = st.sidebar.radio(
            "Select Model Type",
            ["Baseline", "GAN-Based (Occlusion-Aware)"],
            index=1
        )
        
        if model_type == "Baseline":
            estimator = PoseEstimator(model_type="baseline")
        else:
            estimator = PoseEstimator(model_type="gan")
    
    # Home page
    if app_mode == "Home":
        st.markdown("<h1 class='main-header'>Human Pose Estimation with Occlusion Handling</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://miro.medium.com/max/1400/1*yzxlxFd1u_weINTUr6zXxw.png", 
                    caption="Human Pose Estimation Example")
        
        st.markdown("""
        <div class='info-box'>
        <h2 class='sub-header'>Project Overview</h2>
        <p>This application demonstrates a human pose estimation system with special emphasis on handling occlusions through a GAN-based approach.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Key Features</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - 🧠 CNN-based baseline pose estimation
            - 🔄 Conditional GAN for occlusion-aware pose completion
            """)
        
        with col2:
            st.markdown("""
            - 🧩 Self-supervised contrastive learning
            - 📈 Interactive visualization tools
            """)
        
        st.markdown("<h2 class='sub-header'>Get Started</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            demo_button = st.button("Try Demo")
            upload_button = st.button("Upload Your Own Image")
            
        if demo_button:
            st.session_state["app_mode"] = "Demo"
            st.experimental_rerun()
            
        if upload_button:
            st.session_state["app_mode"] = "Upload & Analyze"
            st.experimental_rerun()
            
    # Demo page
    elif app_mode == "Demo":
        st.markdown("<h1 class='main-header'>Interactive Demo</h1>", unsafe_allow_html=True)
        
        # Demo options
        st.markdown("<h2 class='sub-header'>Demo Settings</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            demo_mode = st.radio(
                "Select Demo Type",
                ["Sample Images", "Webcam (Live)"]
            )
            
        with col2:
            add_occlusion = st.checkbox("Simulate Occlusion", value=True)
            if add_occlusion:
                occlusion_level = st.slider("Occlusion Level", 0.1, 0.7, 0.3, 0.1)
            else:
                occlusion_level = 0.0
                
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
        
        if demo_mode == "Sample Images":
            # Sample images
            sample_images = {
                "Person Standing": "https://www.si.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cfl_progressive%2Cq_auto:good%2Cw_1200/MTc5NTk1OTg0ODg2OTc0MTk5/lebron-james-lakers.jpg",
                "Sports Action": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/2022_FIFA_World_Cup_qualification_match_Iran_vs_Korea.jpg/800px-2022_FIFA_World_Cup_qualification_match_Iran_vs_Korea.jpg",
                "Dance Pose": "https://media.istockphoto.com/id/1180926773/photo/ballet-dancer-jumping-during-sunset-with-urban-background.jpg?s=612x612&w=0&k=20&c=DWXx4boO9c5abJOwKbhr7LpvacNw-Q28tn4vRTkcEQE="
            }
            
            selected_sample = st.selectbox("Choose a sample image", list(sample_images.keys()))
            
            # Display selected sample image and process it
            if selected_sample:
                sample_url = sample_images[selected_sample]
                
                # Download and process the image
                try:
                    import requests
                    from PIL import Image
                    
                    response = requests.get(sample_url, stream=True)
                    sample_img = Image.open(BytesIO(response.content))
                    
                    # Convert to RGB if needed
                    if sample_img.mode != "RGB":
                        sample_img = sample_img.convert("RGB")
                    
                    sample_img_np = np.array(sample_img)
                    
                    # Add occlusion if enabled
                    if add_occlusion:
                        processed_img = add_occlusion_to_image(sample_img_np, occlusion_level)
                    else:
                        processed_img = sample_img_np.copy()
                    
                    # Display side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h3 style='text-align: center;'>Original / Occluded Image</h3>", unsafe_allow_html=True)
                        st.image(processed_img, use_column_width=True)
                    
                    # Process the image with the pose estimator
                    results = estimator.process_image(processed_img, add_occlusion, occlusion_level)
                    
                    # Visualize the results
                    vis_img = estimator.visualize_pose(processed_img, results["keypoints"], confidence_threshold)
                    
                    with col2:
                        st.markdown("<h3 style='text-align: center;'>Detected Pose</h3>", unsafe_allow_html=True)
                        st.image(vis_img, use_column_width=True)
                    
                    # Display metrics
                    st.markdown("<h3 class='sub-header'>Detection Results</h3>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_conf = np.mean(results["confidence_scores"])
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Average Confidence</h4>
                            <h2 style='color: {"green" if avg_conf > 0.7 else "orange" if avg_conf > 0.5 else "red"};'>{avg_conf:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        detected = sum(1 for conf in results["confidence_scores"] if conf > confidence_threshold)
                        total = len(results["confidence_scores"])
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Detected Keypoints</h4>
                            <h2>{detected}/{total}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col3:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Processing Time</h4>
                            <h2>{results["processing_time"]:.3f} sec</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence visualization for each keypoint
                    st.markdown("<h3 class='sub-header'>Keypoint Confidence</h3>", unsafe_allow_html=True)
                    
                    # Create confidence data
                    keypoint_names = estimator.get_keypoint_names()
                    confidence_data = pd.DataFrame({
                        'Keypoint': keypoint_names,
                        'Confidence': results["confidence_scores"]
                    })
                    
                    # Plot horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(confidence_data['Keypoint'], confidence_data['Confidence'], color=[
                        'green' if conf > 0.7 else 'orange' if conf > 0.5 else 'red' 
                        for conf in confidence_data['Confidence']
                    ])
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Confidence Score')
                    ax.set_title('Keypoint Detection Confidence')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    # Add confidence values
                    for i, v in enumerate(confidence_data['Confidence']):
                        ax.text(v + 0.01, i, f'{v:.2f}', va='center')
                    
                    st.pyplot(fig)
                    
                    # Download options
                    st.markdown("<h3 class='sub-header'>Download Results</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(get_image_download_link(vis_img, "pose_estimation_result.jpg", "Download Result Image"), 
                                   unsafe_allow_html=True)
                        
                    with col2:
                        # Export keypoints to CSV
                        csv_data = pd.DataFrame({
                            'Keypoint': keypoint_names,
                            'X': [kp[0] for kp in results["keypoints"]],
                            'Y': [kp[1] for kp in results["keypoints"]],
                            'Confidence': [kp[2] for kp in results["keypoints"]]
                        })
                        
                        csv = csv_data.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="keypoints.csv">Download Keypoints CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error processing the sample image: {e}")
                
        else:  # Webcam mode
            st.markdown("<div class='warning-box'>Note: Webcam functionality simulated for demonstration.</div>", 
                       unsafe_allow_html=True)
            
            # Simulated webcam view
            st.markdown("<h3 style='text-align: center;'>Webcam View</h3>", unsafe_allow_html=True)
            
            # Create a placeholder for the webcam feed
            webcam_placeholder = st.empty()
            
            # Simulate webcam frames
            if st.button("Start Webcam"):
                progress = st.progress(0)
                
                # Simulate processing frames
                for i in range(5):
                    # Generate a random frame (in real app, this would be from webcam)
                    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                    
                    # Draw a simple figure in the center
                    cv2.rectangle(frame, (280, 200), (360, 350), (255, 0, 0), -1)  # body
                    cv2.circle(frame, (320, 170), 30, (255, 0, 0), -1)  # head
                    cv2.line(frame, (280, 250), (220, 280), (255, 0, 0), 10)  # left arm
                    cv2.line(frame, (360, 250), (420, 280), (255, 0, 0), 10)  # right arm
                    cv2.line(frame, (300, 350), (280, 430), (255, 0, 0), 10)  # left leg
                    cv2.line(frame, (340, 350), (360, 430), (255, 0, 0), 10)  # right leg
                    
                    # Add simulated occlusion if enabled
                    if add_occlusion:
                        frame = add_occlusion_to_image(frame, occlusion_level)
                    
                    # Process with pose estimator
                    results = estimator.process_image(frame, add_occlusion, occlusion_level)
                    
                    # Visualize the results
                    vis_frame = estimator.visualize_pose(frame, results["keypoints"], confidence_threshold)
                    
                    # Display the frame
                    webcam_placeholder.image(vis_frame, channels="BGR", caption="Pose Estimation on Webcam Feed")
                    
                    # Update progress bar
                    progress.progress((i + 1) * 20)
                    
                    # Simulate processing delay
                    time.sleep(0.5)
                
                st.success("Webcam simulation completed!")
                
                # Sample metrics after processing
                st.markdown("<h3 class='sub-header'>Detection Summary</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Average FPS", "12.5", "2.3")
                    
                with col2:
                    st.metric("Detected Poses", "1", "")
                    
    # Upload & Analyze page
    elif app_mode == "Upload & Analyze":
        st.markdown("<h1 class='main-header'>Upload & Analyze</h1>", unsafe_allow_html=True)
        
        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Settings
        st.markdown("<h2 class='sub-header'>Analysis Settings</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            add_occlusion = st.checkbox("Simulate Occlusion", value=False)
            if add_occlusion:
                occlusion_level = st.slider("Occlusion Level", 0.1, 0.7, 0.3, 0.1)
            else:
                occlusion_level = 0.0
                
        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
        
        if uploaded_file is not None:
            # Read and display the uploaded image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            img_array = np.array(image)
            
            # Add occlusion if enabled
            if add_occlusion:
                processed_img = add_occlusion_to_image(img_array, occlusion_level)
                st.markdown("<div class='warning-box'>Occlusion simulation applied to image</div>", 
                          unsafe_allow_html=True)
            else:
                processed_img = img_array.copy()
            
            # Process the image
            with st.spinner("Processing image..."):
                # Simulate processing delay
                time.sleep(1.5)
                
                # Process with the pose estimator
                results = estimator.process_image(processed_img, add_occlusion, occlusion_level)
                
                # Visualize the results
                vis_img = estimator.visualize_pose(processed_img, results["keypoints"], confidence_threshold)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3 style='text-align: center;'>Input Image</h3>", unsafe_allow_html=True)
                st.image(processed_img, use_column_width=True)
                
            with col2:
                st.markdown("<h3 style='text-align: center;'>Detected Pose</h3>", unsafe_allow_html=True)
                st.image(vis_img, use_column_width=True)
            
            # Display metrics
            st.markdown("<h3 class='sub-header'>Detection Results</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_conf = np.mean(results["confidence_scores"])
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Average Confidence</h4>
                    <h2 style='color: {"green" if avg_conf > 0.7 else "orange" if avg_conf > 0.5 else "red"};'>{avg_conf:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                detected = sum(1 for conf in results["confidence_scores"] if conf > confidence_threshold)
                total = len(results["confidence_scores"])
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Detected Keypoints</h4>
                    <h2>{detected}/{total}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Processing Time</h4>
                    <h2>{results["processing_time"]:.3f} sec</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Keypoint visualization
            st.markdown("<h3 class='sub-header'>Keypoint Details</h3>", unsafe_allow_html=True)
            
            # Create a table of keypoints
            keypoint_names = estimator.get_keypoint_names()
            keypoint_data = []
            
            for i, (x, y, conf) in enumerate(results["keypoints"]):
                status = "Detected" if conf > confidence_threshold else "Low Confidence"
                color = get_confidence_color(conf)
                keypoint_data.append({
                    "ID": i,
                    "Name": keypoint_names[i],
                    "X": int(x),
                    "Y": int(y),
                    "Confidence": f"{conf:.2f}",
                    "Status": status,
                    "Color": color
                })
            
            # Convert to DataFrame for display
            df = pd.DataFrame(keypoint_data)
            
            # Apply row styling based on confidence
            def row_style(row):
                if float(row['Confidence']) > 0.7:
                    return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                elif float(row['Confidence']) > 0.5:
                    return ['background-color: rgba(255, 165, 0, 0.1)'] * len(row)
                else:
                    return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
            
            # Display styled table
            st.dataframe(df.style.apply(row_style, axis=1), height=400)
            
            # Download options
            st.markdown("<h3 class='sub-header'>Download Results</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(get_image_download_link(vis_img, "pose_estimation_result.jpg", "Download Result Image"), 
                          unsafe_allow_html=True)
                
            with col2:
                # Export keypoints to CSV
                csv_data = pd.DataFrame({
                    'Keypoint': keypoint_names,
                    'X': [kp[0] for kp in results["keypoints"]],
                    'Y': [kp[1] for kp in results["keypoints"]],
                    'Confidence': [kp[2] for kp in results["keypoints"]]
                })
                
                csv = csv_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="keypoints.csv">Download Keypoints CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # Advanced analysis tab
            with st.expander("Advanced Analysis"):
                st.markdown("<h4>Body Proportion Analysis</h4>", unsafe_allow_html=True)
                
                # Example of advanced analysis
                limb_lengths = {
                    "Torso": f"{np.random.uniform(35, 45):.1f} cm",
                    "Upper Arm": f"{np.random.uniform(25, 35):.1f} cm",
                    "Lower Arm": f"{np.random.uniform(20, 30):.1f} cm",
                    "Upper Leg": f"{np.random.uniform(35, 45):.1f} cm",
                    "Lower Leg": f"{np.random.uniform(35, 45):.1f} cm"
                }
                
                # Display as a small table
                st.table(pd.DataFrame(list(limb_lengths.items()), columns=["Body Part", "Estimated Length"]))
                
                # Symmetry analysis
                st.markdown("<h4>Pose Symmetry Analysis</h4>", unsafe_allow_html=True)
                
                # Calculate simulated symmetry scores
                symmetry_scores = {
                    "Shoulder Alignment": f"{np.random.uniform(85, 100):.1f}%",
                    "Hip Alignment": f"{np.random.uniform(85, 100):.1f}%",
                    "Arm Extension": f"{np.random.uniform(75, 100):.1f}%",
                    "Leg Extension": f"{np.random.uniform(75, 100):.1f}%",
                    "Overall Symmetry": f"{np.random.uniform(80, 100):.1f}%"
                }
                
                # Display as a small table
                st.table(pd.DataFrame(list(symmetry_scores.items()), columns=["Measure", "Symmetry Score"]))
                
    # Compare Models page
    elif app_mode == "Compare Models":
        st.markdown("<h1 class='main-header'>Model Comparison</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        Compare the performance of our baseline model versus the occlusion-aware GAN model on the same image.
        This helps visualize how the GAN model improves keypoint detection in occluded scenarios.
        </div>
        """, unsafe_allow_html=True)
        
        # Upload or select sample image
        st.markdown("<h2 class='sub-header'>Select Image</h2>", unsafe_allow_html=True)
        
        image_source = st.radio(
            "Image Source",
            ["Upload Image", "Sample Image"]
        )
        
        if image_source == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                img_array = np.array(image)
            else:
                img_array = None
                
        else:  # Sample Image
            sample_images = {
                "Person Standing": "https://www.si.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cfl_progressive%2Cq_auto:good%2Cw_1200/MTc5NTk1OTg0ODg2OTc0MTk5/lebron-james-lakers.jpg",
                "Sports Action": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/2022_FIFA_World_Cup_qualification_match_Iran_vs_Korea.jpg/800px-2022_FIFA_World_Cup_qualification_match_Iran_vs_Korea.jpg",
                "Dance Pose": "https://media.istockphoto.com/id/1180926773/photo/ballet-dancer-jumping-during-sunset-with-urban-background.jpg?s=612x612&w=0&k=20&c=DWXx4boO9c5abJOwKbhr7LpvacNw-Q28tn4vRTkcEQE="
            }
            
            selected_sample = st.selectbox("Choose a sample image", list(sample_images.keys()))
            
            if selected_sample:
                sample_url = sample_images[selected_sample]
                
                try:
                    import requests
                    
                    response = requests.get(sample_url, stream=True)
                    sample_img = Image.open(BytesIO(response.content))
                    
                    # Convert to RGB if needed
                    if sample_img.mode != "RGB":
                        sample_img = sample_img.convert("RGB")
                    
                    img_array = np.array(sample_img)
                except Exception as e:
                    st.error(f"Error loading sample image: {e}")
                    img_array = None
            else:
                img_array = None
                
        # Comparison settings
        st.markdown("<h2 class='sub-header'>Comparison Settings</h2>", unsafe_allow_html=True)
        
        occlusion_level = st.slider("Occlusion Level", 0.1, 0.8, 0.4, 0.1,
                                  help="Higher values mean more parts of the image will be occluded")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1,
                                      help="Minimum confidence score to consider a keypoint as detected")
        
        # Process the image if available
        if img_array is not None:
            # Add occlusion to the image
            with st.spinner("Adding simulated occlusion..."):
                occluded_img = add_occlusion_to_image(img_array, occlusion_level)
            
            # Display the original and occluded images
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
                st.image(img_array, use_column_width=True)
                
            with col2:
                st.markdown("<h3 style='text-align: center;'>Occluded Image</h3>", unsafe_allow_html=True)
                st.image(occluded_img, use_column_width=True)
            
            # Compare models
            if st.button("Run Comparison"):
                with st.spinner("Running models..."):
                    # Initialize models
                    baseline_estimator = PoseEstimator(model_type="baseline")
                    gan_estimator = PoseEstimator(model_type="gan")
                    
                    # Process with both models
                    baseline_results = baseline_estimator.process_image(occluded_img, True, occlusion_level)
                    
                    # Add slight delay to simulate processing time difference
                    time.sleep(0.5)
                    gan_results = gan_estimator.process_image(occluded_img, True, occlusion_level)
                    
                    # Visualize results
                    baseline_vis = baseline_estimator.visualize_pose(occluded_img, baseline_results["keypoints"], confidence_threshold)
                    gan_vis = gan_estimator.visualize_pose(occluded_img, gan_results["keypoints"], confidence_threshold)
                
                # Display results
                st.markdown("<h2 class='sub-header'>Model Comparison Results</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h3 style='text-align: center;'>Baseline Model</h3>", unsafe_allow_html=True)
                    st.image(baseline_vis, use_column_width=True)
                    
                    # Show metrics
                    baseline_detected = sum(1 for conf in baseline_results["confidence_scores"] if conf > confidence_threshold)
                    baseline_avg_conf = np.mean(baseline_results["confidence_scores"])
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p><b>Detected Keypoints:</b> {baseline_detected}/17</p>
                        <p><b>Average Confidence:</b> {baseline_avg_conf:.2f}</p>
                        <p><b>Processing Time:</b> {baseline_results["processing_time"]:.3f} sec</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown("<h3 style='text-align: center;'>GAN-Based Model</h3>", unsafe_allow_html=True)
                    st.image(gan_vis, use_column_width=True)
                    
                    # Show metrics
                    gan_detected = sum(1 for conf in gan_results["confidence_scores"] if conf > confidence_threshold)
                    gan_avg_conf = np.mean(gan_results["confidence_scores"])
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p><b>Detected Keypoints:</b> {gan_detected}/17</p>
                        <p><b>Average Confidence:</b> {gan_avg_conf:.2f}</p>
                        <p><b>Processing Time:</b> {gan_results["processing_time"]:.3f} sec</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Comparative analysis
                st.markdown("<h3 class='sub-header'>Performance Comparison</h3>", unsafe_allow_html=True)
                
                # Calculate improvement metrics
                detection_improvement = gan_detected - baseline_detected
                confidence_improvement = gan_avg_conf - baseline_avg_conf
                time_difference = gan_results["processing_time"] - baseline_results["processing_time"]
                
                # Performance metrics
                metrics = {
                    "Metric": ["Detected Keypoints", "Average Confidence", "Processing Time"],
                    "Baseline": [baseline_detected, f"{baseline_avg_conf:.2f}", f"{baseline_results['processing_time']:.3f} sec"],
                    "GAN Model": [gan_detected, f"{gan_avg_conf:.2f}", f"{gan_results['processing_time']:.3f} sec"],
                    "Difference": [
                        f"+{detection_improvement}" if detection_improvement > 0 else str(detection_improvement),
                        f"{confidence_improvement:+.2f}",
                        f"{time_difference:+.3f} sec"
                    ]
                }
                
                df_metrics = pd.DataFrame(metrics)
                st.table(df_metrics)
                
                # Keypoint-by-keypoint comparison
                st.markdown("<h3 class='sub-header'>Keypoint-Level Comparison</h3>", unsafe_allow_html=True)
                
                # Create comparison data
                keypoint_names = baseline_estimator.get_keypoint_names()
                comparison_data = []
                
                for i, name in enumerate(keypoint_names):
                    baseline_conf = baseline_results["confidence_scores"][i]
                    gan_conf = gan_results["confidence_scores"][i]
                    diff = gan_conf - baseline_conf
                    
                    comparison_data.append({
                        "Keypoint": name,
                        "Baseline Confidence": f"{baseline_conf:.2f}",
                        "GAN Confidence": f"{gan_conf:.2f}",
                        "Difference": f"{diff:+.2f}",
                        "Improved": "✅" if diff > 0.05 else ("❌" if diff < -0.05 else "➖")
                    })
                
                # Convert to DataFrame and display
                df_comparison = pd.DataFrame(comparison_data)
                
                # Apply styling
                def highlight_improvement(val):
                    if val == "✅":
                        return 'background-color: rgba(0, 255, 0, 0.2)'
                    elif val == "❌":
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                    return ''
                
                # Display styled table
                st.dataframe(df_comparison.style.applymap(highlight_improvement, subset=['Improved']), height=400)
                
                # Visualization of confidence differences
                st.markdown("<h3 class='sub-header'>Confidence Visualization</h3>", unsafe_allow_html=True)
                
                # Create data for bar chart
                baseline_conf = baseline_results["confidence_scores"]
                gan_conf = gan_results["confidence_scores"]
                
                # Set up the comparison chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                x = np.arange(len(keypoint_names))
                width = 0.35
                
                ax.bar(x - width/2, baseline_conf, width, label='Baseline', color='skyblue')
                ax.bar(x + width/2, gan_conf, width, label='GAN Model', color='orange')
                
                ax.set_xticks(x)
                ax.set_xticklabels(keypoint_names, rotation=45, ha="right")
                ax.legend()
                
                ax.set_ylim(0, 1)
                ax.set_ylabel('Confidence')
                ax.set_title('Confidence Comparison by Keypoint')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary
                st.markdown("<h3 class='sub-header'>Analysis Summary</h3>", unsafe_allow_html=True)
                
                # Calculate summary stats
                improved_keypoints = sum(1 for i in range(len(keypoint_names)) 
                                       if gan_conf[i] - baseline_conf[i] > 0.05)
                unchanged_keypoints = sum(1 for i in range(len(keypoint_names)) 
                                        if abs(gan_conf[i] - baseline_conf[i]) <= 0.05)
                deteriorated_keypoints = sum(1 for i in range(len(keypoint_names)) 
                                           if gan_conf[i] - baseline_conf[i] < -0.05)
                
                # Show summary box
                if gan_avg_conf > baseline_avg_conf:
                    st.markdown(f"""
                    <div class='success-box'>
                        <h4>GAN Model Performs Better</h4>
                        <p>The GAN-based model showed <b>{improved_keypoints}</b> improved keypoints, 
                        <b>{unchanged_keypoints}</b> unchanged keypoints, and <b>{deteriorated_keypoints}</b> deteriorated keypoints.</p>
                        <p>Overall, the GAN model had {gan_detected - baseline_detected} more keypoints detected above the threshold, 
                        with an average confidence improvement of {confidence_improvement:.2f}.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif gan_avg_conf < baseline_avg_conf:
                    st.markdown(f"""
                    <div class='warning-box'>
                        <h4>Baseline Model Performs Better</h4>
                        <p>The baseline model showed better overall performance with {baseline_detected} keypoints detected versus 
                        {gan_detected} in the GAN model. The average confidence was {-confidence_improvement:.2f} higher.</p>
                        <p>This might be due to the specific occlusion pattern or the keypoint distribution in this particular image.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='info-box'>
                        <h4>Models Perform Similarly</h4>
                        <p>Both models showed similar performance on this image. The GAN model detected {gan_detected} keypoints 
                        while the baseline model detected {baseline_detected}.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
    # About page
    elif app_mode == "About":
        st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <h2 class='sub-header'>Human Pose Estimation with Occlusion Handling</h2>
        
        <p>This project implements a human pose estimation system with special emphasis on handling occlusions through a GAN-based approach. 
        It follows a multi-phase approach from baseline model development to deployment with optimization.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Architecture overview
        st.markdown("<h2 class='sub-header'>Model Architecture</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Baseline Model
            - ResNet50 backbone
            - HRNet head for keypoint detection
            - Trained on COCO 2017 dataset
            - Standard heatmap-based approach
            """)
            
        with col2:
            st.markdown("""
            #### GAN-Based Model
            - Conditional GAN architecture
            - T-UNet generator for occlusion completion
            - Multi-scale discriminator
            - Trained using adversarial and reconstruction losses
            """)
            
        # Add a diagram
        st.markdown("<h3>System Architecture</h3>", unsafe_allow_html=True)
        
        # Create a simple Mermaid diagram
        st.markdown("""
        ```mermaid
        graph TD
            A[Input Image] --> B[Feature Extraction]
            B --> C[Keypoint Detection]
            C --> D{Occlusion Check}
            D -->|No Occlusion| E[Final Pose]
            D -->|Occlusion Detected| F[GAN Completion]
            F --> G[Refined Keypoints]
            G --> E
            E --> H[Visualization]
        ```
        """)
        
        # Project features
        st.markdown("<h2 class='sub-header'>Key Features</h2>", unsafe_allow_html=True)
        
        features = [
            {"icon": "🧠", "title": "Deep Learning-Based Approach", 
             "desc": "Uses state-of-the-art deep learning models for accurate pose estimation."},
            {"icon": "🔄", "title": "Occlusion Handling", 
             "desc": "Special emphasis on handling occluded body parts using a GAN-based approach."},
            {"icon": "🧩", "title": "Contrastive Learning", 
             "desc": "Self-supervised contrastive learning for improved feature representation."},
            {"icon": "🚀", "title": "Optimized for Deployment", 
             "desc": "Model pruning and quantization for efficient inference on various devices."}
        ]
        
        # Display features in a grid
        cols = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; background-color: #f8f9fa;">
                    <h3>{feature['icon']} {feature['title']}</h3>
                    <p>{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Team and credits
        st.markdown("<h2 class='sub-header'>Team & Acknowledgments</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <p>This project was developed by [Your Name/Team] as part of [Course/Research/Personal Project].</p>
        
        <h4>Acknowledgments:</h4>
        <ul>
            <li>The COCO dataset for providing training data</li>
            <li>The PyTorch team for the deep learning framework</li>
            <li>The Streamlit team for the web app framework</li>
            <li>The academic community for foundational research in human pose estimation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # License and source code
        st.markdown("<h2 class='sub-header'>License & Source</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <p>This project is licensed under the MIT License.</p>
        
        <p>Source code available at: [Your GitHub Repository Link]</p>
        
        <p>For questions or support, please contact: [Your Email or Contact Information]</p>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()