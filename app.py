import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

st.set_page_config(
    page_title=("Image Processing Toolbox"),
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🧰 Image Processing Toolbox")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🖼️ Image Info",
        "🎨 Filters & Effects",
        "📊 Analysis Dashboard",
        "🙂 Face Detection",
        "🌈 Color Palette",
        "⬇️ Download / Export"
    ]
)

if page == "🏠 Home":
    st.title("🖼️ Image Processing Toolbox")
    st.subheader("A simple, powerful image analysis web app")
    st.markdown(
        """
         ### 🚀 What this app can do:
        - Upload and preview images
        - Apply image filters & effects
        - Visualize image data with dashboards
        - Detect faces using OpenCV
        - Extract dominant color palettes
        - Download processed results

        👈 Use the **sidebar** to navigate through features.
        """
    )
    st.info("Start by uploading an image from the sidebar pages.")
    
    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)",type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file
        st.success("Image uploaded successfully!")

    
    if st.session_state.uploaded_image is not None:
        image = Image.open(st.session_state.uploaded_image)
        st.image(image,caption="Uploaded image",use_column_width=True)
    
if page == "🖼️ Image Info":
        st.title("🖼️ Image Information")
        
        if st.session_state.uploaded_image is None:
            st.warning("Please upload an image from the Home page first.")
        else:
            image = Image.open(st.session_state.uploaded_image)

            width , height = image.size
            image_format = image.format
            file_size_kb = st.session_state.uploaded_image.size/1024
            aspect_ratio = round(width/height,2)
            if abs(aspect_ratio - 1) < 0.01:
                shape = "Square"
            elif aspect_ratio> 1:
                shape ="Landscape (Wide)"
            else:
                shape = "Portrait (Tall)"

            col1 , col2 = st.columns([2,1])

            with col1:
                st.image(image,caption="Uploaded Image",use_column_width=True)
            
            with col2:
                st.markdown("### 📋 Image Details")
                st.metric("Width (px)",width)
                st.metric("Height (px)",height)
                st.metric("Format",image_format)
                st.metric("File Size (kb)", f"{file_size_kb:.2f}")
                st.metric("Aspect Ratio",aspect_ratio)
                st.metric("Shape ", shape)
if page == "🎨 Filters & Effects":
    st.title("🎨 Filters & Effects")

    if st.session_state.uploaded_image is None:
        st.warning("Please upload an image from the Home page first.")
    else:
        st.session_state.uploaded_image.seek(0)
        file_bytes = np.asarray(bytearray(st.session_state.uploaded_image.read()),dtype= np.uint8)
        cv_image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)

        st.sidebar.subheader("Filter Options")
        apply_grayscale = st.sidebar.checkbox("Grayscale")
        blur_strength = st.sidebar.slider("Gaussian Blur",0,20,0)
        apply_canny = st.sidebar.checkbox("Edge Detection")

        processed_img = cv_image.copy()

        #grayscale 
        if apply_grayscale:
            processed_img = cv2.cvtColor(processed_img,cv2.COLOR_RGB2GRAY)
            processed_img = cv2.cvtColor(processed_img,cv2.COLOR_GRAY2RGB)

        #gaussian blur 
        if blur_strength > 0:
            processed_img = cv2.GaussianBlur(processed_img,(blur_strength*2+1,blur_strength*2+1),0)
        
        #edgedetection
        if apply_canny:
            gray = cv2.cvtColor(processed_img,cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray,100,200)
            processed_img = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)

        col1 , col2 = st.columns(2)
        with col1:
            st.image(cv_image,caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_img,caption="Processed Image", use_column_width=True)
        st.session_state.filtered_image = processed_img
if page == "📊 Analysis Dashboard":
    st.title("📊 Image Analysis Dashboard")
    if st.session_state.uploaded_image is None:
        st.warning("Please upload an image from the Home page first.")
    else:
        st.session_state.uploaded_image.seek(0)
        file_bytes= np.asarray(bytearray(st.session_state.uploaded_image.read()),dtype=np.uint8)
        cv_image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

        rgb_img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY) 

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎨 RGB Histogram")
            fig,ax = plt.subplots()

            colors = ("r","g","b")
            for i,col in enumerate(colors):
                hist = cv2.calcHist([rgb_img],[i],None,[256],[0,255])
                ax.plot(hist,color= col)

            ax.set_title("RGB Color Distribution")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

        with col2:
            st.subheader("🔅 Brightness Histogram")
            fig2,ax2 = plt.subplots()

            hist_gray = cv2.calcHist([gray_img],[0],None,[256],[0,255])
            ax2.plot(hist_gray)

            ax2.set_title("Brightness Distribution")
            ax2.set_xlabel("Pixel Intensity")
            ax2.set_ylabel("Frequency")

            st.pyplot(fig2)
        
        st.markdown("### 📋 Image Statistics")

        mean_val = np.mean(gray_img)
        min_val = np.min(gray_img)
        max_val = np.max(gray_img)

        c1,c2,c3 = st.columns(3)

        c1.metric("Average Brightness",f"{mean_val:.2f}")
        c2.metric("Min Intensity", int(min_val))
        c3.metric("Max Intensity", int(max_val))

        st.markdown("---")
        st.markdown("### 🧠 Image Summary")
         
        if mean_val < 80:
            summary_text = "The image is overall TOO DARK"
            st.info("🔴 " + summary_text)
        elif mean_val < 160:
            summary_text = "The image has MODERATE brightness"
            st.info("🟡 " + summary_text)
        else:
            summary_text = "The image is BRIGHT and well-lit"
            st.info("🟢 " + summary_text)
        
        if min_val < 30:
            dark_text = "Very dark regions are present in the image"
            st.info("⚫ " + dark_text)
        else:
            dark_text = "No strong dark regions detected"
            st.info("⚫ " + dark_text)
        if max_val > 220:
            bright_text = "Very bright highlights are present in the image"
            st.info("⚪ " + bright_text)
        else:
            bright_text = "No extreme bright regions detected"
            st.info("⚪ " + bright_text)
        st.session_state.mean_val = mean_val
        st.session_state.min_val = min_val
        st.session_state.max_val = max_val

        st.session_state.summary_text = summary_text
        st.session_state.dark_text = dark_text
        st.session_state.bright_text = bright_text

if page == "🙂 Face Detection":
    st.title("🙂 Face Detection")
    if st.session_state.uploaded_image is None:
        st.warning("Please upload an image from the Home page first.")
    else:
        st.session_state.uploaded_image.seek(0)
        file_bytes = np.asarray(bytearray(st.session_state.uploaded_image.read()),dtype=np.uint8)
        img = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=6,minSize=(30,30))

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        
        rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        st.image(rgb_img,caption = "Face Detection Result",use_column_width=True)

        st.metric("Faces Detected", len(faces))
        st.session_state.face_image = rgb_img
        st.session_state.face_count = len(faces)

if page == "🌈 Color Palette":
    st.title("🌈 Color Palette Generator")
    if st.session_state.uploaded_image is None:
        st.warning("Please upload an image from the Home page first.")
    else:
        st.session_state.uploaded_image.seek(0)
        file_bytes = np.asarray(bytearray(st.session_state.uploaded_image.read()),dtype=np.uint8)
        img = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        st.image(img,caption="Original Image",use_column_width=True)

        small_img = cv2.resize(img,(100,100))
        pixels = small_img.reshape(-1,3)

        k = st.slider("Number of Colors",2,8,5)

        with st.spinner("🎨 Generating color palette... please wait"): 
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)

            st.markdown("### 🎨 Dominant Colors")

            for color in colors:
                r,g,b = color
                brightness = (r + g + b) / 3

                text_color = "white" if brightness < 128 else "black"

                st.markdown(
                            f"""
                            <div style=" 
                            background-color: rgb({r},{g},{b});
                            padding: 20px;
                            margin: 10px 0;
                            border-radius: 10px;
                            color:{text_color} ;
                            font-weight: bold;">
                            RGB: ({r}, {g}, {b})
                            </div>
                            """,
                            unsafe_allow_html=True
                            )
if page == "⬇️ Download / Export":
    st.title("⬇️ Download / Export")
     
    if "filtered_image" not in st.session_state and "face_image" not in st.session_state:
        st.warning("No processed image available. Apply a filter or detection first.")
    else:
        option = st.selectbox("Choose Image", ["Filtered", "Face Detection"])

        if option == "Filtered":
            img = st.session_state.get("filtered_image", None)
        else:
            img = st.session_state.get("face_image", None)

        if img is None:
            st.warning("No image available. Please apply filter or face detection first.")
        else:
            st.image(img, caption="Processed Image", use_column_width=True)
            
            import io 
            from PIL import Image

            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf,format="PNG")
            byte_Im = buf.getvalue()

            st.download_button(label="⬇️ Download Processed Image",data=byte_Im,file_name="processed_image.png",mime="image/png")
        face_count = st.session_state.get("face_count", 0)
        mean_val = st.session_state.get("mean_val", 0)
        min_val = st.session_state.get("min_val", 0)
        max_val = st.session_state.get("max_val", 0)

        summary_text = st.session_state.get("summary_text", "No summary available")
        dark_text = st.session_state.get("dark_text", "")
        bright_text = st.session_state.get("bright_text", "")
        report = f"""
        ==============================
            IMAGE ANALYSIS REPORT
        ==============================

        📊 Image Statistics
        ------------------------------
        Average Brightness : {mean_val:.2f}
        Min Intensity      : {min_val}
        Max Intensity      : {max_val}

        🧠 Summary
        ------------------------------
        Overall : {summary_text}
        Dark Regions : {dark_text}
        Bright Regions : {bright_text}

        🙂 Face Detection
        ------------------------------
        Faces Detected : {face_count}

        ==============================
        """

        st.download_button(
            label="⬇️ Download Full Report",
            data=report,
            file_name="image_analysis_report.txt",
            mime="text/plain"
        )