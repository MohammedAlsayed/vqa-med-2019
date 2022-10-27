from utils import make_df


# python3 train.py --run_name  "bench_mark1" --mixed_precision --batch_size 16 --num_vis 64 --epochs 10

df = make_df("/Users/mohammed/Workspace/csci499-project/dataset/testing/VQAMed2019_Test_Questions_w_Ref_Answers.txt")
df.to_csv("/Users/mohammed/Workspace/csci499-project/dataset/testdf.csv",index=False)