import os 
import time
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove_if_submit_success", action="store_true", default=False)
    args = parser.parse_args()
    submission_dir = "./submission"

    count = 0
    # Iterate through all files in the submission folder
    submit_fail_list = []
    for file in os.listdir(submission_dir):
        if file.endswith(".csv"):
            file_path = os.path.abspath(os.path.join(submission_dir, file))
            command = f"kaggle competitions submit -c ofc-2026-ml-challenge -f {file_path} -m message"
            try:
                return_flag =os.system(command)
            except Exception as e:
                print(f"Error submitting {file}: {e}")
                submit_fail_list.append(file)
            if return_flag == 0:
                print(f"Submit success: {file}")
                if args.remove_if_submit_success:
                    os.remove(file_path)
                    print(f"Remove success: {file}")
                count += 1
        if count % 10 == 0:
            time.sleep(20)
    print(f"Submit fail list: {submit_fail_list}")