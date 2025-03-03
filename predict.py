import os
import shutil
import tarfile
import zipfile
import mimetypes
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from weights_downloader import WeightsDownloader
from cog_model_helpers import optimise_images
from config import config
import requests
  

os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

mimetypes.add_type("image/webp", ".webp")
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
LORA_DIR = "ComfyUI/models/loras"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]
VIDEO_TYPES = [".mp4", ".mov", ".avi", ".mkv"]

with open("example_workflow.json", "r") as file:
    EXAMPLE_WORKFLOW_JSON = file.read()


class Predictor(BasePredictor):
    def setup(self, weights: str):
        if bool(weights):
            self.handle_user_weights(weights)

        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def handle_user_weights(self, weights: str):
        print(f"Downloading user weights from: {weights}")
        WeightsDownloader.download("weights.tar", weights, config["USER_WEIGHTS_PATH"])
        for item in os.listdir(config["USER_WEIGHTS_PATH"]):
            source = os.path.join(config["USER_WEIGHTS_PATH"], item)
            destination = os.path.join(config["MODELS_PATH"], item)
            if os.path.isdir(source):
                if not os.path.exists(destination):
                    print(f"Moving {source} to {destination}")
                    shutil.move(source, destination)
                else:
                    for root, _, files in os.walk(source):
                        for file in files:
                            if not os.path.exists(os.path.join(destination, file)):
                                print(
                                    f"Moving {os.path.join(root, file)} to {destination}"
                                )
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(
                                    f"Skipping {file} because it already exists in {destination}"
                                )

    def handle_input_file(self, input_file: Path):
        file_extension = self.get_file_extension(input_file)

        if file_extension == ".tar":
            with tarfile.open(input_file, "r") as tar:
                tar.extractall(INPUT_DIR)
        elif file_extension == ".zip":
            with zipfile.ZipFile(input_file, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)
        elif file_extension in IMAGE_TYPES + VIDEO_TYPES:
            shutil.copy(input_file, os.path.join(INPUT_DIR, f"input{file_extension}"))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print("====================================")
        print(f"Inputs uploaded to {INPUT_DIR}:")
        self.comfyUI.get_files(INPUT_DIR)
        print("====================================")

    def get_file_extension(self, input_file: Path) -> str:
        file_extension = os.path.splitext(input_file)[1].lower()
        if not file_extension:
            with open(input_file, "rb") as f:
                file_signature = f.read(4)
            if file_signature.startswith(b"\x1f\x8b"):  # gzip signature
                file_extension = ".tar"
            elif file_signature.startswith(b"PK"):  # zip signature
                file_extension = ".zip"
            else:
                try:
                    with Image.open(input_file) as img:
                        file_extension = f".{img.format.lower()}"
                        print(f"Determined file type: {file_extension}")
                except Exception as e:
                    raise ValueError(
                        f"Unable to determine file type for: {input_file}, {e}"
                    )
        return file_extension

    def download_file(self, url: str, dest_path: str):
        try:
            # Ensure the directory exists before attempting to create the file
            directory = os.path.dirname(dest_path)
            if directory:  # Only try to create if there's a directory component
                os.makedirs(directory, exist_ok=True)
                
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded: {dest_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return False

    def handle_replicate_loras(self, urls_str: str, lora_names_str: str = ""):
        if not urls_str.strip():
            return
                
        urls = [url.strip() for url in urls_str.split('\n') if url.strip()]
        if not urls:
            return
                
        # Process custom lora names if provided
        custom_names = []
        if lora_names_str.strip():
            custom_names = [name.strip() for name in lora_names_str.split(',')]
        
        # Ensure LORA_DIR exists
        os.makedirs(LORA_DIR, exist_ok=True)
        
        # Track used names to avoid internal conflicts
        used_names = set()
        
        for i, url in enumerate(urls):
            # Determine the lora name to use
            if i < len(custom_names) and custom_names[i].strip():
                lora_name = custom_names[i]
            else:
                lora_name = "lora" if i == 0 else f"lora_{i+1}"
            
            # Add a suffix if this name was already used in this batch
            original_name = lora_name
            counter = 1
            while lora_name in used_names:
                lora_name = f"{original_name}_{counter}"
                counter += 1
            
            used_names.add(lora_name)
                    
            # Create a unique tar path for each download
            tar_path = os.path.join(INPUT_DIR, f"replicate_lora_{i}.tar")
            
            # Ensure INPUT_DIR exists
            os.makedirs(INPUT_DIR, exist_ok=True)
            
            if self.download_file(url, tar_path):
                try:
                    # Create temporary extraction directory
                    temp_dir = os.path.join(INPUT_DIR, f"temp_extract_{i}")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Extract the tar file to the temp directory
                    with tarfile.open(tar_path, "r") as tar:
                        tar.extractall(temp_dir)
                    
                    # Find the first safetensors file recursively
                    safetensor_file = None
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith(".safetensors"):
                                safetensor_file = os.path.join(root, file)
                                break
                        if safetensor_file:
                            break
                    
                    if safetensor_file:
                        # Destination path with custom name
                        dest_path = os.path.join(LORA_DIR, f"{lora_name}.safetensors")
                        
                        # Check if file already exists in destination
                        if os.path.exists(dest_path):
                            print(f"Warning: File {dest_path} already exists. Creating a backup.")
                            backup_path = os.path.join(LORA_DIR, f"{lora_name}_backup.safetensors")
                            counter = 1
                            while os.path.exists(backup_path):
                                backup_path = os.path.join(LORA_DIR, f"{lora_name}_backup_{counter}.safetensors")
                                counter += 1
                            shutil.copy(dest_path, backup_path)
                        
                        # Move and rename the file
                        shutil.copy(safetensor_file, dest_path)
                        print(f"Successfully copied {safetensor_file} to {dest_path}")
                    else:
                        print(f"No .safetensors file found in the archive from {url}")
                    
                    # Clean up
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                except Exception as e:
                    print(f"Error extracting LoRA from tar {i}: {e}")
                    import traceback
                    traceback.print_exc()

    def handle_external_lora_links(self, urls_str: str, lora_names_str: str = ""):
        if not urls_str.strip():
            return
            
        # Split URLs into a list and strip any extra whitespace
        urls = [url.strip() for url in urls_str.split('\n') if url.strip()]
        if not urls:
            return

        # Ensure the LORA_DIR exists
        os.makedirs(LORA_DIR, exist_ok=True)

        for i, url in enumerate(urls):
            # Extract the original filename from the URL (if available)
            original_filename = url.split('/')[-1]
            
            # Construct the destination path directly in LORA_DIR
            dest_path = os.path.join(LORA_DIR, original_filename)
            
            # Use wget to download the file directly to LORA_DIR
            wget_command = f"wget --content-disposition '{url}' -P {LORA_DIR}"
            print(f"Running wget command: {wget_command}")
            
            # Execute the wget command
            os.system(wget_command)
            

 
    def predict(
        self,
        workflow_json: str = Input(
            description="Your ComfyUI workflow as JSON string or URL. You must use the API version of your workflow. Get it from ComfyUI using 'Save (API format)'. Instructions here: https://github.com/replicate/cog-comfyui",
            default="",
        ),
        input_file: Path = Input(
            description="Input image, video, tar or zip file. Read guidance on workflows and input files here: https://github.com/replicate/cog-comfyui. Alternatively, you can replace inputs with URLs in your JSON workflow and the model will download them.",
            default=None,
        ),
        return_temp_files: bool = Input(
            description="Return any temporary files, such as preprocessed controlnet images. Useful for debugging.",
            default=False,
        ),
        replicate_lora: str = Input(
            description="URLs to tar files containing replicate LoRAs. Enter one URL per line for multiple LoRAs.",
            default="",
        ),
        lora_name: str = Input(
            description="Custom names for LoRA files (comma-separated list). If fewer names than links are provided, remaining LoRAs will use default naming (lora, lora_2, etc.).",
            default="lora",
        ),
        external_lora_link: str = Input(
            description="Direct URLs to LoRA files. Enter one URL per line for multiple LoRAs.",
            default="",
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        randomise_seeds: bool = Input(
            description="Automatically randomise seeds (seed, noise_seed, rand_seed)",
            default=True,
        ),
        force_reset_cache: bool = Input(
            description="Force reset the ComfyUI cache before running the workflow. Useful for debugging.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
    
        if input_file:
            self.handle_input_file(input_file)
        if replicate_lora:
            self.handle_replicate_loras(replicate_lora, lora_name)
        if external_lora_link:
            self.handle_external_lora_links(external_lora_link, lora_name)
    
        # Check for GPU availability
        gpu_available = self.check_gpu_availability()
        
        if not gpu_available:
            print("No GPU available. Cannot run ComfyUI workflow.")
            return [Path(os.path.join(OUTPUT_DIR, "no_gpu.txt"))]
        
        # Only proceed with ComfyUI workflow if GPU is available
        workflow_json_content = workflow_json
        if workflow_json.startswith(("http://", "https://")):
            try:
                response = requests.get(workflow_json)
                response.raise_for_status()
                workflow_json_content = response.text
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to download workflow JSON from URL: {e}")
    
        wf = self.comfyUI.load_workflow(workflow_json_content or EXAMPLE_WORKFLOW_JSON)
    
        self.comfyUI.connect()
    
        if force_reset_cache or not randomise_seeds:
            self.comfyUI.reset_execution_cache()
    
        if randomise_seeds:
            self.comfyUI.randomise_seeds(wf)
    
        self.comfyUI.run_workflow(wf)
    
        output_directories = [OUTPUT_DIR]
        if return_temp_files:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)
    
        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(output_directories)
        )

    def check_gpu_availability(self):
        """Check if a GPU is available for use."""
        try:
            # Try importing torch first to check GPU availability
            import torch
            if torch.cuda.is_available():
                print(f"GPU available: {torch.cuda.get_device_name(0)}")
                return True
            
            # Alternative check using nvidia-smi
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                print("GPU available via nvidia-smi")
                return True
        except (ImportError, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Create a no_gpu.txt file to indicate no GPU was available
        with open(os.path.join(OUTPUT_DIR, "no_gpu.txt"), "w") as f:
            f.write("No GPU available for running ComfyUI workflow.")
        
        return False
