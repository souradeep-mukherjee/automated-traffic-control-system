import time
import streamlit as st
from ultralytics import YOLO
import cv2
import glob
import os

# Load the YOLOv8 model (use a pre-trained model such as 'yolov8n', 'yolov8s', etc.)
model = YOLO('yolov8n.pt')  # Replace with a fine-tuned model if available for specific traffic datasets

# Define the object classes to count
target_classes = {"car": 2, "bus": 5}  # Class IDs based on the YOLOv8 COCO dataset


# Function to simulate traffic lights
def simulate_traffic_light(lane_name, is_green):
    if is_green:
        return f"🟢 Green Light: Lane '{lane_name}'"
    else:
        return f"🔴 Red Light: Lane '{lane_name}'"


# Streamlit app UI
st.set_page_config(layout="wide")  # Use the entire width of the screen
st.title("Automated Traffic Light Simulation")

# Sidebar for directory input
st.sidebar.title("Image Upload Configuration")
directory_path = st.sidebar.text_input("Enter the path to the images directory:")
output_directory = st.sidebar.text_input("Enter the path to save annotated images:")

# Create a button to load images from the specified directory
if st.sidebar.button("Load Images from Directory"):
    if directory_path and output_directory:
        # Get all image files from the specified directory
        image_extensions = ('*.jpg', '*.png', '*.jpeg')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))

        if not image_files:
            st.sidebar.error("No images found in the specified directory!")
        else:
            st.sidebar.success(f"Loaded {len(image_files)} images from '{directory_path}'")
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
    else:
        st.sidebar.error("Please enter valid directory paths for input and output.")

# If no images are loaded, exit
if 'image_files' not in locals() or not image_files:
    st.write("Please load images using the sidebar.")
    st.stop()

# Process images to annotate and count vehicles
lane_counts = {}
annotated_images = []

for idx, image_path in enumerate(image_files, start=1):
    # Read the image from the path
    image = cv2.imread(image_path)
    if image is None:
        st.write(f"Could not load image: {image_path}")
        continue

    # Run YOLOv8 inference
    results = model.predict(source=image, save=False, conf=0.3)

    # Initialize counts for cars and buses
    car_count = 0
    bus_count = 0

    # Parse detections
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id == target_classes["car"]:
                car_count += 1
            elif class_id == target_classes["bus"]:
                bus_count += 1

    # Annotate the image
    annotated_image = results[0].plot()  # YOLOv8's built-in function to draw annotations
    annotated_image_path = os.path.join(output_directory, f"ann_image_{idx}.jpg")
    cv2.imwrite(annotated_image_path, annotated_image)
    annotated_images.append(annotated_image_path)

    # Store lane counts
    lane_name = os.path.basename(image_path).split('.')[0]  # Use file name as lane identifier
    lane_counts[lane_name] = {"cars": car_count, "buses": bus_count, "total": car_count + bus_count}

# Priority Scheduling
# Sort lanes by total vehicle count in descending order (highest priority first)
sorted_lanes = sorted(lane_counts.items(), key=lambda x: x[1]['total'], reverse=True)

# Batch traffic light results
batch_size = 4
traffic_light_results = []
for i in range(0, len(sorted_lanes), batch_size):
    batch = sorted_lanes[i:i + batch_size]
    batch_results = []
    for priority, (lane, counts) in enumerate(batch, start=1):
        is_green = priority == 1  # Highest priority lane gets the green light
        light_status = simulate_traffic_light(lane, is_green)
        batch_results.append(
            f"<p class='stText'>{light_status} | Lane '{lane}': Cars = {counts['cars']}, Buses = {counts['buses']}, Total = {counts['total']}</p>"
        )
    traffic_light_results.append(batch_results)

# Layout: Three columns
col1, col2, col3 = st.columns([2, 2, 3])

# Left column: Display original images in 2x2 grid format
with col1:
    st.markdown('<p class="stHeader">Original Traffic Images</p>', unsafe_allow_html=True)
    for i in range(0, len(image_files), 4):
        batch = image_files[i:i + 4]
        cols = st.columns(4)
        for idx, image_path in enumerate(batch):
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cols[idx].image(image, use_column_width=True, caption=os.path.basename(image_path))

# Middle column: Display annotated images in 2x2 grid format
with col2:
    st.markdown('<p class="stHeader">Annotated Traffic Images</p>', unsafe_allow_html=True)
    for i in range(0, len(annotated_images), 4):
        batch = annotated_images[i:i + 4]
        cols = st.columns(4)
        for idx, annotated_image_path in enumerate(batch):
            image = cv2.imread(annotated_image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cols[idx].image(image, use_column_width=True, caption=os.path.basename(annotated_image_path))

# Right column: Display traffic light results with 5-second interval between batches
with col3:
    st.markdown('<p class="stHeader">Traffic Light Results</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="stText">Traffic light simulation results (Green/Red lights) based on priority scheduling:</p>',
        unsafe_allow_html=True)

    if traffic_light_results:
        for batch_index, batch_results in enumerate(traffic_light_results):
            st.markdown(f'<p class="stHeader">Batch {batch_index + 1}</p>', unsafe_allow_html=True)
            for result in batch_results:
                st.markdown(result, unsafe_allow_html=True)

            if batch_index < len(traffic_light_results) - 1:  # Avoid sleeping after the last batch
                st.markdown('<p class="stText">Processing next batch in 5 seconds...</p>', unsafe_allow_html=True)
                time.sleep(5)  # Wait 5 seconds before showing the next batch

        # Cleanup: Delete annotated images
        for annotated_image_path in annotated_images:
            try:
                os.remove(annotated_image_path)
            except OSError as e:
                st.write(f"Error deleting file {annotated_image_path}: {e}")

        st.markdown('<p class="stText">All annotated images have been deleted from the directory.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="stText">No results to display. Please process images and run the simulation.</p>', unsafe_allow_html=True)
#KEEP THE DOCUMENT AT 67% FOR MAX VISIBILITY