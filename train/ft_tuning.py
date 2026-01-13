import subprocess
import shutil
from  loguru import logger
import os


gpus = ["0", "1"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

def load_json(data_path):
    with open(data_path, "r") as f:
        data = f.read()
    return data

import json

def save_jsonl(data, data_path):
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
            
    print(f"Save to {data_path}")

def load_jsonl(data_path):
    with open(data_path, "r") as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]
    return data 

def covert_instruction_input_output_to_instruction_response(data_path):
    
    data = load_jsonl(data_path)  
    new_data = []
    for line in data:
        new_data.append({"instruction": line.get("instruction", "") + "\n" + line.get("input", "") if line.get("input")!= "" else line.get("instruction"),
                         "response": line.get("output")})
        
    return new_data 

def save_jsonl(data, data_path):
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
            
    print(f"Save to {data_path}")
    

def run_bash_script(model_name_or_path, data_path, output_dir,template="alpaca"):
    logger.info(f"Finetuning model {model_name_or_path} with template {template}")
    data = covert_instruction_input_output_to_instruction_response(data_path)
    # for fanno
    import random
    random.shuffle(data)
    data = data[:1000]
    
    save_jsonl(data, "data/selected_data.jsonl")
    # shutil.copy(data_path, "data/selected_data.jsonl")
    bash_command = [
        "bash", "ft.sh",
        "--model_name_or_path", model_name_or_path,
        "--dataset", "SELECTED_DATA",
        "--output_dir", output_dir,
        "--template", template,
        "--device", ",".join(gpus)
    ]
    subprocess.run(bash_command)


def sft(data_paths, base_model, template="alpaca"):
    assert base_model in ["llama2-7b", "llama3-8b", "llama3.1-8b", "mistralv3-7b", "llama3.2-3b"]
    model_base_path = {
        "llama2-7b": "/share/home/u24147/.cache/modelscope/hub/models/modelscope/Llama-2-7b-ms",
        "llama3-8b": "/share/home/u24147/data/tiny_model/Meta-Llama-3-8B", 
        "llama3.1-8b": "/share/home/u24147/data/tiny_model/Meta-Llama-3.1-8B",
        "llama3.2-3b": "/share/home/u24147/data/tiny_model/llama3.2-3b"
        # "llama3.1-8b": "/share/home/tj24147/data/huggingface_model/LLaMA/Meta-Llama-3.1-8B",
        # "mistralv3-7b": "/share/home/tj24147/data/huggingface_model/Mistral/mistral-7b-v0.3"
        # "mistralv3-8b": "/share/home/tj24147/data/huggingface_model/Mistral/mistral-7b-v0.3"
    }
    ft_model_path = model_base_path[base_model]
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
        ft_output_dir = f"/share/home/u24147/data/sft/{base_model}-{name}"
        logger.info(f"Finetuning model {base_model} with data {path}")
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=ft_output_dir,
            template=template
        )
    



# conda activate flash_attn
# /share/home/tj24147/MetaMath/ft_tuning.py
# data_path = ["/share/home/tj24147/arixv_data/fanno_rebuttal_naacl/ablation_11_25_change1_20000_converted.jsonl"]
# data_path = ["/share/home/tj24147/arixv_data/tag-instruct-related-data/main/magpie_longest_10k.jsonl"]
# data_path = ["/share/home/tj24147/arixv_data/fanno_rebuttal_naacl/fanno_final_data.jsonl"]
# data_path = ["/share/home/tj24147/MetaMath/tmp_trainer/alpaca-longest-instruction-10k.jsonl"] # "/share/home/tj24147/MetaMath/tmp_trainer/WizardLM-longest-10k.jsonl"
    

# def load_jsonl(data_path):
#     with open(data_path, "r") as f:
#         data = f.readlines()
#     return data



# /share/home/u24147/gsm8k.jsonl

# dataset/WizardLM_evol_instruct_V2_196k_processed.jsonl

if __name__ == "__main__":

    # data_path = ["/home/zhe/arixv_data/tag-instruct-related-data/ablation/diversity_1w_data.jsonl"]
    # data_path = ["/home/zhe/data/alpaca_response_longest_5k.jsonl",
    #              "/home/zhe/data/alpaca_instruction_longest_5k.jsonl",
    #              "/home/zhe/data/alpaca_ours_5k.jsonl",
    #              "/home/zhe/data/alpaca_ours_direct_5k.jsonl"]
    
    
    # data_path = ["/home/zhe/arixv_data/tag-instruct-related-data/main/wizardlm/wizardlm_evol_instruct_10k_responses.jsonl",
    #              "/home/zhe/arixv_data/tag-instruct-related-data/main/alpaca/alpaca_data_10k_response.jsonl",
    # data_path = [ "/home/zhe/arixv_data/ablation_11_25_change3_20000_data.jsonl",
    #              "/home/zhe/arixv_data/fanno_rebuttal_naacl/fanno_final_data.jsonl",
    #              "/home/zhe/arixv_data/alpaca_data_cleaned_10k_response.jsonl"]

                #  "/home/zhe/arixv_data/tag-instruct-related-data/main/alpaca/alpaca_data_10k_ins.jsonl"]

                #  "/home/zhe/arixv_data/tag-instruct-related-data/main/diversity_1w_data-12-12.jsonl"]
                #  "/home/zhe/arixv_data/tag-instruct-related-data/main/magpie/magpie_10k_responses.jsonl"]
                
                
    # data_path = ["/home/zhe/arixv_data/alpaca_evol_instruct_70k.jsonl",
    # data_path = [  "/home/zhe/arixv_data/tag-instruct-related-data/main/alpaca/alpaca_data_cleaned.jsonl",
                #  "/home/zhe/arixv_data/tag-instruct-related-data/main/alpaca/top_10_percent_filtered_data.json"]
                
    # data_path = ["/home/zhe/arixv_data/tag-instruct-related-data/main/magpie/magpie_random_10k_mistralgen.jsonl"]
    # data_path = ["/home/zhe/arixv_data/fanno_response2.jsonl"]
    # /home/zhe/fanno_response.jsonl
    # data_path = ["/home/zhe/fanno_response.jsonl"]
    # data_path = ["/home/zhe/arixv_data/fanno_llama3gen_response2.jsonl"]
    data_path =[
        # "/home/zhe/arixv_data/fanno_response2_code_math.jsonl"
        # "/home/zhe/arixv_data/fanno-cm-50k.jsonl"
        
        # "/home/zhe/models/trained_model/llama3-8b-alpaca_cleaned_top_10k_ratio"
        # "/home/zhe/arixv_data/fanno_llama3gen_response2.jsonl"
        # "/home/zhe/arixv_data/tag-instruct-related-data/main/alpaca/alpaca_data_cleaned_ppl_max_10k.jsonl",
        # "/home/zhe/arixv_data/tag-instruct-related-data/main/alpaca/alpaca_data_cleaned_ppl_min_10k.jsonl"
        # "/home/zhe/arixv_data/fanno_response2_shortins_resrewrite.jsonl",
        # "/home/zhe/arixv_data/fanno_response2_shortins.jsonl"
        # "/home/zhe/arixv_data/tag-instruct-related-data/main/magpie/magpie_random_10k_llama3-8b_response.jsonl",
        # "/home/zhe/arixv_data/fanno_response2_ratio_llama3-8b_gen.jsonl",
        # "/home/zhe/arixv_data/fanno_response2_ratio_mistral_gen.jsonl",
        # "/home/zhe/arixv_data/tag-instruct-related-data/main/magpie/magpie_random_10k_llama3-8b_response.jsonl",
        # "/home/zhe/response_longest_5k2_converted.jsonl",
        # "/home/zhe/sft_data_filtered_by_avg_internal_score_5k2.json",
        # "/home/zhe/sft_data_filtered_by_avg_weighted_score_5k2.json"
        # converted
        # "/home/zhe/sft_data_filtered_by_avg_internal_score_5k2_converted.jsonl",
        # "/home/zhe/sft_data_filtered_by_avg_weighted_score_5k2_converted.jsonl"
        # "/home/zhe/dataset/WizardLM_evol_instruct_V2_196k_processed.jsonl",
        # "/home/zhe/dataset/alpaca_data_cleaned.jsonl",
        # "/home/zhe/dataset/magpie_pro_3.3_500k.jsonl"
        
        # new-experiment
        
        # "/home/zhe/tag-instruct-experiment/alpaca_5k.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/evol/evol_0.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/evol/evol_1.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/evol/evol_2.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/evol/evol_3.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/evol/evol_4.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tree-instruct/tree_instruct_0.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tree-instruct/tree_instruct_1.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tree-instruct/tree_instruct_2.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tree-instruct/tree_instruct_3.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tree-instruct/tree_instruct_4.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/autoevol/autoevol_0.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/autoevol/autoevol_1.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/autoevol/autoevol_2.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/autoevol/autoevol_3.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/autoevol/autoevol_4.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/codeclm/codeclm_iter_0.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/codeclm/codeclm_iter_1.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/codeclm/codeclm_iter_2.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/codeclm/codeclm_iter_3.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/codeclm/codeclm_iter_4.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct/tag_instruct_0.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct/tag_instruct_1.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct/tag_instruct_2.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct/tag_instruct_3.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct/tag_instruct_4.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-reward/tag_instruct_reward_0.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-reward/tag_instruct_reward_1.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-reward/tag_instruct_reward_2.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-reward/tag_instruct_reward_3.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-reward/tag_instruct_reward_4.jsonl",
        # "/home/zhe/tag-reward/dataset/sft/laip/response_longest_5k2_converted.jsonl",
        # "/home/zhe/tag-reward/dataset/sft/wizardlm/alpaca_evol_instruct_70k.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct/tag_instruct_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tree-instruct/tree_instruct_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/autoevol/autoevol_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/evol/evol_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/codeclm/codeclm_all.jsonl"
        # "/home/zhe/tag-reward/dataset/sft/wizardlm/WizardLM_evol_instruct_V2_196k_processed.jsonl"
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-experiment/0/tag_instruct_0_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-experiment/1/tag_instruct_1_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-experiment/2/tag_instruct_2_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-experiment/3/tag_instruct_3_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-experiment/4/tag_instruct_4_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-experiment/5/tag_instruct_5_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-experiment/6/tag_instruct_6_all.jsonl",
        # "/home/zhe/tag-instruct-experiment/outputs/top_results.jsonl",
        # "/home/zhe/tag-instruct-experiment/outputs/bottom_results.jsonl",
        # "/home/zhe/tag-instruct-experiment/outputs/random_results.jsonl"
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-qwen72b/tag_instruct_4_qwen.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/codeclm-qwen72b/codeclm_iter_4_qwen.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/autoevol-qwen72b/autoevol_4_qwen.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/evol-qwen72b/evol_4_qwen.jsonl"
        # "/home/zhe/tag-instruct-experiment/result/tree-instruct-qwen/tree_instruct_4_qwen.jsonl"
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-reward-magpie-qwen72b/tag_instruct_reward_4_magpie_qwen.jsonl",
        # "/home/zhe/tag-instruct-experiment/result/tag-instruct-magpie-qwen72b/tag_instruct_4_magpie_qwen.jsonl",
        # "/home/zhe/tag-instruct-experiment/magpie_5k.jsonl"
        # "/share/home/tj24147/sft_common_data/tree_instruct_4_magpie_qwen.jsonl",
        # "/share/home/tj24147/sft_common_data/evol_4_magpie_qwen.jsonl",
        # "/share/home/tj24147/sft_common_data/codeclm_iter_4_magpie_qwen.jsonl",
        # "/share/home/tj24147/sft_common_data/autoevol_4_magpie_qwen.jsonl"
        # "/share/home/tj24147/Fanno2/experiment/response-100k/ucb_aug_5_top10k_formatted.jsonl"
        # "/share/home/tj24147/Fanno2/experiment/ablation3/initial_seed_formated_a3.jsonl",
        # "/share/home/tj24147/Fanno2/experiment/ablation2/ucb_aug_5_formated_a2.jsonl",
        # "/share/home/tj24147/Fanno2/experiment/ablation1/ucb_aug_15_formated_a1.jsonl",
        # "/share/home/tj24147/Fanno2/experiment/response-100k/fanno_5k.jsonl",
        # "/share/home/tj24147/Fanno2/experiment/response-100k/fanno_20k.jsonl"
        # "/share/home/tj24147/tag-instruct-light/experiment/alpaca_reward_tag_instruct/tag_instruct_reward_4_alpaca_reward.jsonl"
        # "/share/home/u24147/PKU-UPGLM/urban_planning_model/data/plangpt_demo.json"
        # "/share/home/u24147/funny/DAFT/data/magpie_5k.jsonl"
        
        # "/share/home/u24147/gsm8k.jsonl"
        # "/share/home/u24147/tag-instruct-experiment/gsm8k_boxed_format.jsonl",
        # "/share/home/u24147/tag-instruct-experiment/result/math-evolution-qwen72b-7k/math_evol_2.jsonl",
        # "/share/home/u24147/tag-instruct-experiment/result/tag-instruct-math-qwen7b_7k/math_problem_generation_2.jsonl"
        # "/share/home/u24147/research/Fanno2/experiment/fanno_qwew25_72b_instruct/ucb_aug_3.jsonl"
        # "/share/home/u24147/research/Fanno2/experiment/fanno_qwen25_14b_instruct/ucb_aug_3.jsonl"
        "/share/home/u24147/research/Fanno2/experiment/fanno_qwen25_14b_instruct_math/math_aug_3.jsonl"
        
    ]
    
    for path in data_path:
        # 3b
        sft([path], "llama2-7b", template="llama2")
        # llama2-7b
        # sft([path], "llama3.2-3b")
        
        
        

        
    
        
        
        