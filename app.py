import cv2 as cv
import numpy as np
import gradio as gr

# Filtre fonksiyonları
def apply_gaussian_blur(frame):
    return cv.GaussianBlur(frame, (15, 15), 0)

def apply_sharpening_filter(frame, strength=1):
    base_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel = base_kernel + (strength - 1) * np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return cv.filter2D(frame, -1, kernel)

def apply_edge_detection(frame):
    return cv.Canny(frame, 100, 200)

def apply_invert_filter(frame):
    return cv.bitwise_not(frame)

def adjust_brightness_contrast(frame, alpha=1.0, beta=0):
    return cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_grayscale_filter(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def apply_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    return cv.transform(frame, kernel)

def apply_posterize(frame, levels=4):
    frame = frame.astype(np.float32)
    posterized = np.floor(frame / (256 / levels)) * (256 / levels)
    return posterized.astype(np.uint8)

def apply_emboss(frame):
    kernel = np.array([[ -2, -1,  0], [-1,  1,  1], [ 0,  1,  2]])
    return cv.filter2D(frame, -1, kernel)

def apply_dilate(frame):
    kernel = np.ones((5,5), np.uint8)
    return cv.dilate(frame, kernel, iterations=1)

def apply_erode(frame):
    kernel = np.ones((5,5), np.uint8)
    return cv.erode(frame, kernel, iterations=1)

def apply_film_grain(frame, grain_intensity, grain_size):
    h, w, c = frame.shape
    noise = np.random.uniform(0, 1, (h // grain_size, w // grain_size, c)).astype(np.float32)
    noise = cv.resize(noise, (w, h), interpolation=cv.INTER_LINEAR)
    noise = (noise * (grain_intensity * 255)).astype(np.uint8)
    noisy_image = cv.add(frame, noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_fisheye(frame):
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([0.3, -0.3, 0, 0], dtype=np.float32) 
    return cv.fisheye.undistortImage(frame, K, D)

def apply_pixelate(frame, pixel_size=10):
    h, w = frame.shape[:2]
    frame = cv.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv.INTER_LINEAR)
    return cv.resize(frame, (w, h), interpolation=cv.INTER_NEAREST)

def apply_heatmap(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.applyColorMap(gray, cv.COLORMAP_JET)

def apply_mosaic(frame, block_size=10):
    h, w = frame.shape[:2]
    frame = cv.resize(frame, (w // block_size, h // block_size), interpolation=cv.INTER_LINEAR)
    return cv.resize(frame, (w, h), interpolation=cv.INTER_NEAREST)

def apply_cartoon(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 7)
    edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 10)
    color = cv.bilateralFilter(frame, 9, 250, 250)
    return cv.bitwise_and(color, color, mask=edges)

def apply_chalk_drawing(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    inv_gray = cv.bitwise_not(gray)
    blurred = cv.GaussianBlur(inv_gray, (21, 21), 0)
    return cv.divide(gray, 255 - blurred, scale=256)

def apply_sketch(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    inv_gray = cv.bitwise_not(gray)
    blurred = cv.GaussianBlur(inv_gray, (21, 21), 0)
    return cv.divide(gray, blurred, scale=256)

def apply_soft_focus(frame):
    blurred = cv.GaussianBlur(frame, (15, 15), 0)
    return cv.addWeighted(frame, 0.7, blurred, 0.3, 0)

def apply_solarize(frame, threshold=128):
    solarized = np.where(frame < threshold, frame, 255 - frame)
    return solarized.astype(np.uint8)

def apply_filter(filter_type, input_image=None, sharpness_strength=1, brightness_strength=50, grain_intensity=0.2, grain_size=4):
    if input_image is not None:
        frame = input_image
    else:
        cap = cv.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "No footage from the webcam."

    if filter_type == "Gaussian Blur":
        return apply_gaussian_blur(frame)
    elif filter_type == "Sharpen":
        return apply_sharpening_filter(frame, strength=sharpness_strength)
    elif filter_type == "Edge Detection":
        return apply_edge_detection(frame)
    elif filter_type == "Invert":
        return apply_invert_filter(frame)
    elif filter_type == "Brightness":
        return adjust_brightness_contrast(frame, alpha=1.0, beta=brightness_strength)
    elif filter_type == "Grayscale":
        return apply_grayscale_filter(frame)
    elif filter_type == "Sepia":
        return apply_sepia(frame)
    elif filter_type == "Posterize":
        return apply_posterize(frame, levels=4)    
    elif filter_type == "Emboss":
        return apply_emboss(frame)
    elif filter_type == "Dilate":
        return apply_dilate(frame)
    elif filter_type == "Erode":
        return apply_erode(frame)         
    elif filter_type == "Film Grain":
        return apply_film_grain(frame, grain_intensity, grain_size)
    elif filter_type == "Fisheye":
        return apply_fisheye(frame)
    elif filter_type == "Pixelate":
        return apply_pixelate(frame, pixel_size=10)
    elif filter_type == "Heatmap":
        return apply_heatmap(frame)
    elif filter_type == "Mosaic":
        return apply_mosaic(frame, block_size=10)
    elif filter_type == "Cartoon Effect":
        return apply_cartoon(frame)
    elif filter_type == "Chalk Drawing":
        return apply_chalk_drawing(frame)
    elif filter_type == "Sketch":
        return apply_sketch(frame)
    elif filter_type == "Soft Focus":
        return apply_soft_focus(frame)
    elif filter_type == "Solarize":
        return apply_solarize(frame)

with gr.Blocks() as demo:
    gr.Markdown("📸 Görsel ya da Webcaminden Canlı Filtre Uygula")

    # Ana Düzen
    with gr.Row():
        with gr.Column(scale=1):
            # Filtre Seçim Listesi
            filter_type = gr.Dropdown(
                label="Filtreni Seç",
                choices=[
                    "Gaussian Blur",
                    "Sharpen",
                    "Edge Detection",
                    "Invert",
                    "Brightness",
                    "Grayscale",
                    "Sepia",
                    "Posterize",
                    "Emboss",
                    "Dilate",
                    "Erode",
                    "Film Grain",
                    "Fisheye",
                    "Pixelate",
                    "Heatmap",
                    "Mosaic",
                    "Cartoon Effect",
                    "Chalk Drawing",
                    "Sketch",
                    "Soft Focus",
                    "Solarize"
                ],
                value="Gaussian Blur"
            )

            # Kaydırma Çubukları
            sharpness_strength = gr.Slider(label="Sharpness Strength", minimum=1, maximum=5, value=1, step=1, visible=False)
            brightness_strength = gr.Slider(label="Brightness Strength", minimum=-100, maximum=100, value=50, step=1, visible=False)
            grain_intensity = gr.Slider(label="Grain Intensity", minimum=0.0, maximum=1.0, value=0.2, step=0.01, interactive=True, visible=False)
            grain_size = gr.Slider(label="Grain Size", minimum=1, maximum=20, value=4, step=1, interactive=True, visible=False)

            # Görsel Yükleme Alanı
            input_image = gr.Image(label="Fotoğrafını Yükle ya da Canlı Olarak Webcamden Filtre Uygula", type="numpy", interactive=True)

        with gr.Column(scale=2):
            # Çıktı Alanı
            gr.Markdown("Yeni Görseliniz")
            output_image = gr.Image(label="Filtre Uygulandı")

    # Kaydırma Çubuklarının Görünürlüğünü Güncelleyen Fonksiyon
    def update_visibility(filter_type):
        sharpness_visible = (filter_type == "Sharpen")
        brightness_visible = (filter_type == "Brightness")
        grain_visible = (filter_type == "Film Grain")

        return (
            gr.update(visible=sharpness_visible, interactive=sharpness_visible),
            gr.update(visible=brightness_visible, interactive=brightness_visible),
            gr.update(visible=grain_visible, interactive=grain_visible),
            gr.update(visible=grain_visible, interactive=grain_visible)
        )

    # Filtre türü değiştiğinde kaydırma çubuklarının görünürlüğünü güncelle ve otomatik filtre uygula
    filter_type.change(fn=lambda filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size: apply_filter(filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size),
                       inputs=[filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size],
                       outputs=output_image)

    # Filtre türü değiştiğinde kaydırma çubuklarının görünürlüğünü güncelle
    filter_type.change(update_visibility, inputs=filter_type, outputs=[sharpness_strength, brightness_strength, grain_intensity, grain_size])

    # Kaydırma çubukları değiştiğinde otomatik filtre uygula
    sharpness_strength.change(lambda filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size: apply_filter(filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size),
                               inputs=[filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size],
                               outputs=output_image)

    brightness_strength.change(lambda filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size: apply_filter(filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size),
                               inputs=[filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size],
                               outputs=output_image)

    grain_intensity.change(lambda filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size: apply_filter(filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size),
                           inputs=[filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size],
                           outputs=output_image)

    grain_size.change(lambda filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size: apply_filter(filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size),
                      inputs=[filter_type, input_image, sharpness_strength, brightness_strength, grain_intensity, grain_size],
                      outputs=output_image)

# Gradio Arayüzü Başlat
demo.launch()
