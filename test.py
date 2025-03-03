import os
import shutil

# Define LORA_DIR where you want to download the files
LORA_DIR = "/teamspace/studios/this_studio/test"  # Target directory for LoRA models

def handle_external_lora_links(urls_str: str, lora_names_str: str = ""):
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
        
        # Check if the file has been successfully downloaded
        if os.path.exists(dest_path):
            print(f"Successfully downloaded and saved the LoRA to {dest_path}")
        else:
            print(f"Failed to download or locate the file: {dest_path}")

# Example usage
handle_external_lora_links(urls_str="https://civitai.com/api/download/models/1471549?type=Model&format=SafeTensor&token=47cb4c3be00f51afefef207254221b1c")
