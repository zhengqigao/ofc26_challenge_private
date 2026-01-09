import os 

submission_dir = "./submission"

# Iterate through all files in the submission folder
for file in os.listdir(submission_dir):
    if file.endswith(".csv"):
        file_path = os.path.abspath(os.path.join(submission_dir, file))
        command = f"kaggle competitions submit -c ofc-2026-ml-challenge -f {file_path} -m message"
        os.system(command)