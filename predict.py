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

with open("default_workflow.json", "r") as file:
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
        
        for i, url in enumerate(urls):
            # Determine the lora name to use
            if i < len(custom_names) and custom_names[i].strip():
                lora_name = custom_names[i]
            else:
                lora_name = "lora" if i == 0 else f"lora_{i+1}"
                
            # Create a unique tar path for each download
            tar_path = os.path.join(INPUT_DIR, f"replicate_lora_{i}.tar")
            if self.download_file(url, tar_path):
                try:
                    with tarfile.open(tar_path, "r") as tar:
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith(".safetensors"):
                                extracted_path = os.path.join(INPUT_DIR, member.name)
                                tar.extract(member, INPUT_DIR)
                                
                                # Use custom name with safetensors extension
                                custom_lora_filename = f"{lora_name}.safetensors"
                                
                                dest_path = os.path.join(LORA_DIR, custom_lora_filename)
                                shutil.move(extracted_path, dest_path)
                                print(f"Moved and renamed to {custom_lora_filename} in {LORA_DIR}")
                                # Process only the first safetensors file in each tar
                                break
                except Exception as e:
                    print(f"Error extracting LoRA from tar {i}: {e}")

    def handle_external_lora_links(self, urls_str: str, lora_names_str: str = ""):
        if not urls_str.strip():
            return
            
        urls = [url.strip() for url in urls_str.split('\n') if url.strip()]
        if not urls:
            return
        
        # Process custom lora names if provided
        custom_names = []
        if lora_names_str.strip():
            custom_names = [name.strip() for name in lora_names_str.split(',')]
        
        for i, url in enumerate(urls):
            # Determine the lora name to use
            if i < len(custom_names) and custom_names[i].strip():
                lora_name = custom_names[i]
            else:
                lora_name = "lora" if i == 0 else f"lora_{i+1}"
            
            custom_lora_filename = f"{lora_name}.safetensors"
            lora_path = os.path.join(LORA_DIR, custom_lora_filename)
            
            if self.download_file(url, lora_path):
                print(f"Downloaded external LoRA to {lora_path}")


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
