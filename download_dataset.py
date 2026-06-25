import urllib.request
import re
import os

file_id = "1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
base_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_dir, "dataset.csv")

def get_confirm_token(response):
    for key, value in response.info().items():
        if key.lower() == 'set-cookie':
            match = re.search(r'download_warning_([^=]+)=([^;]+)', value)
            if match:
                return match.group(1), match.group(2)
    return None, None

def download_file():
    print("Initiating download from Google Drive...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            confirm_key, confirm_val = get_confirm_token(response)
            
            if confirm_key:
                confirm_url = f"{url}&confirm={confirm_val}"
                print(f"Confirmation required, downloading via confirm URL...")
                req = urllib.request.Request(confirm_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as confirm_response:
                    content = confirm_response.read()
            else:
                content = response.read()
            
            # Save the file.
            # Let's inspect content header or extension to name it properly.
            # We'll save it as dataset.csv or check if it's zip.
            # Let's write to raw first.
            with open(output_path, "wb") as f:
                f.write(content)
            print(f"Downloaded successfully. Size: {len(content)} bytes")
            
            # Let's inspect the first 200 bytes to see what format it is.
            with open(output_path, "rb") as f:
                header = f.read(200)
                print("Header preview:", header[:100])
                if b"PK\x03\x04" in header:
                    print("This is a ZIP file.")
                elif b"InvoiceNo" in header or b"StockCode" in header or b"," in header:
                    print("This is likely a CSV file.")
    except Exception as e:
        print(f"Error downloading file: {e}")

if __name__ == "__main__":
    download_file()
