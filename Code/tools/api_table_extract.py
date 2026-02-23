from pathlib import Path
import requests
import json
import time
 
api_key = ""  # ex: up_xxxYYYzzzAAAbbbCCC

in_path = Path("/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/GUS/data/raw pics")
out_path = in_path.parent / "extracted"

# filenames = [f"pop_age_{year}_f.png" for year in [1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994]]
# filenames += [f"pop_age_men_{year}_f.png" for year in [1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994]]
filenames = [f"pop__tot_age_men_educ_{year} [meta].png" for year in [1986, 1987, 1988, 1991, 1992, 1993, 1994]]

url = "https://api.upstage.ai/v1/document-digitization"
headers = {"Authorization": f"Bearer {api_key}"}

for filename in filenames:
    files = {"document": open(str(in_path / filename), "rb")}
    data = {"ocr": "auto", "base64_encoding": "['table']", "model": "document-parse", "mode": "enhanced", "output_formats": "['html', 'text', 'markdown']"}
    
    print(f"Processing {filename}...")
    
    response = requests.post(url, headers=headers, files=files, data=data)
    
    # Save json response to file
    with open(str(out_path / filename.replace(".png",".json")), "w") as f:
        f.write(response.text)
    
    time.sleep(1)  # Sleep for 1 second to avoid hitting rate limits
    # break