import gradio as gr
import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
from scripts.brats_inference import inference
from utils.utils import load_config, create_tensor_from_volume
config = load_config("bsm_brats_config.yaml")
prediction = None
def load_modalities(file_obj):
    try:
        # Nếu file_obj có thuộc tính 'seek', đọc trực tiếp
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
            data = file_obj.read()
        else:
            # Nếu không, coi file_obj là đường dẫn và mở file
            with open(file_obj, "rb") as f:
                data = f.read()
        print(f"Độ dài dữ liệu nhận được: {len(data)} bytes")
        if len(data) == 0:
            print("❌ Dữ liệu file rỗng!")
            return None
        buffer = io.BytesIO(data)
        modalities = torch.load(buffer, map_location="cpu", weights_only=False)
        return modalities
    except Exception as e:
        print(f"Error loading file: {e}")
        return None



def plot_slice(modalities_tensor, channel, depth_idx):
    global prediction
    if channel is None or depth_idx is None or modalities_tensor is None:
        prediction = None
        return None
    modality_slice = modalities_tensor[channel, depth_idx, :, :]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(modality_slice, cmap='gray')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img

def plot_prediction(modalities_tensor, depth_idx):
    global prediction
    if depth_idx is None or modalities_tensor is None:
        prediction = None
        return None
    modality_slice = modalities_tensor[:, depth_idx, :, :]
    modality_slice = modality_slice.transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(modality_slice, cmap='gray')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img

# Khi upload file, tự động hiển thị ảnh của kênh Flair (mặc định)
def handle_file_upload(modalities_file):
    global prediction
    print("handle_file_upload triggered. File:", modalities_file)
    prediction = None  # Reset prediction khi file mới hoặc bị xóa
    if modalities_file is None:
        return None, gr.update(maximum=100, value=0), None, None, None  # Thêm dòng này

    modalities_tensor = load_modalities(modalities_file)
    if modalities_tensor is not None:
        depth = modalities_tensor.shape[1]
        channel = 0  # Mặc định là "Flair"
        depth_idx = 0
        initial_image = plot_slice(modalities_tensor, channel, depth_idx)
        return modalities_tensor, gr.update(maximum=depth - 1, value=0), initial_image, None, None  # Reset luôn output_image
    return None, gr.update(maximum=100, value=0), None, None, None


# Auto update ảnh khi thay đổi channel/slice
def auto_update_image(input_modalities_tensor, output_modalities_state, channel_name, slice_index):
    channel_map = {"Flair": 0, "T1ce": 1, "T2": 2}
    channel = channel_map.get(channel_name, 0)  # Mặc định là 0 nếu không tìm thấy
    if prediction is not None:
        output_modalities_state = create_tensor_from_volume(input_modalities_tensor, prediction, modal=channel)
        output_modalities_state = output_modalities_state.permute(3, 0, 1, 2).cpu().numpy()
    if input_modalities_tensor is None or channel is None or slice_index is None or output_modalities_state is None:
        return None, None, None
    return plot_slice(input_modalities_tensor, channel, slice_index), plot_prediction(output_modalities_state, slice_index), output_modalities_state

# Khi bấm predict: hiện slice hiện tại ở khung output
def predict_and_show_current_slice(input_modalities_tensor, channel_name, slice_index):
    global prediction
    channel_map = {"Flair": 0, "T1ce": 1, "T2": 2}
    channel = channel_map.get(channel_name, 0)  # Mặc định là 0 nếu không tìm thấy
    prediction =  inference(config, input_modalities_tensor, cpkt="checkpoint/segformer3d_bsm/model.safetensors")[0]
    output_modalities_tensor = create_tensor_from_volume(input_modalities_tensor, prediction, modal=channel)
    output_modalities_tensor = output_modalities_tensor.permute(3, 0, 1, 2).cpu().numpy()
    return output_modalities_tensor, plot_prediction(output_modalities_tensor, slice_index)

with gr.Blocks(css=""" 
body { background-color: #e8f5fc; }
h1 { color: #006494; display: flex; align-items: center; gap: 10px; }
h1::before { content: ""; display: inline-block; width: 40px; height: 40px; background-image: url('https://cdn-icons-png.flaticon.com/512/3774/3774299.png'); background-size: cover; }
.gr-button { background-color: #0288d1 !important; color: white !important; }
.gr-button:hover { background-color: #0277bd !important; }
.gr-slider-label { color: #006494 !important; }
""") as demo:
    
    gr.Markdown("<h1>Medical Image Segmentation</h1>")
    gr.Markdown("<p style='color: #045c7d; font-weight: bold;'>Upload your modality file (.pt), select channel and slice depth to auto preview. Click Predict to show current slice in another frame.</p>")

    input_modalities_state = gr.State(None)  # Lưu tensor sau khi load
    output_modalities_state = gr.State(None)  # Lưu tensor sau khi load

    with gr.Row():
        modalities_input = gr.File(label="Upload .pt File", file_types=[".pt"])
    
    channel_selector = gr.Dropdown(
        choices=["Flair", "T1ce", "T2"], 
        value="Flair", 
        label="Choose channel"
    )
    
    depth_slider = gr.Slider(0, 100, step=1, value=0, label="Choose slice depth")

    predict_btn = gr.Button("Predict")

    with gr.Row():
        input_image = gr.Image(label="Current Slice by Channel (Auto Update)")
        output_image = gr.Image(label="Prediction Result (On Predict)")

    # Khi upload file: load modalities và hiển thị ảnh mặc định
    modalities_input.change(
        fn=handle_file_upload,
        inputs=[modalities_input],
        outputs=[input_modalities_state, depth_slider, input_image, output_image, output_modalities_state]  
    )

    # Auto update Current Slice khi channel hoặc depth thay đổi
    channel_selector.change(
        fn=auto_update_image, 
        inputs=[input_modalities_state, output_modalities_state, channel_selector, depth_slider],
        outputs=[input_image, output_image, output_modalities_state]
    )

    depth_slider.change(
        fn=auto_update_image, 
        inputs=[input_modalities_state, output_modalities_state, channel_selector, depth_slider],
        outputs=[input_image, output_image, output_modalities_state]
    )

    # Bấm Predict sẽ hiển thị ở khung Prediction Result
    predict_btn.click(
        fn=predict_and_show_current_slice, 
        inputs=[input_modalities_state, channel_selector, depth_slider],
        outputs=[output_modalities_state, output_image]
    )

demo.launch()