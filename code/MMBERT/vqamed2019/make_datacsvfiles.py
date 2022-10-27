# from utils import make_df
from utils import load_data


# python3 train.py --run_name  "bench_mark1" --mixed_precision --batch_size 16 --num_vis 64 --epochs 10

# df = make_df("/Users/mohammed/Workspace/csci499-project/dataset/testing/VQAMed2019_Test_Questions_w_Ref_Answers.txt")
# df.to_csv("/Users/mohammed/Workspace/csci499-project/dataset/testdf.csv",index=False)

def check_val_images():
    train, val, test = load_data("../../../dataset/", remove = None)
    for i in train["img_id"].values:
        print(i)
        f = open(i)
        f.close()

check_val_images()