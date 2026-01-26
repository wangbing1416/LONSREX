import os
import argparse
import requests
import math
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def call_model_sglang(query, url):
    data = {
        "text": query,
        "logprob_start_len": -1,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "top_logprobs_num": 1024,  # Fetch more candidates to ensure real/fake appear
        "sampling_params": {"max_new_tokens": 1, "temperature": 0}
    }
    responses = requests.post(url, json=data, timeout=1200)
    return responses.json()


def read_jsonl(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = []
        with open(path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f'processing {path.split("/")[-1]}'):
                data.append(json.loads(line))
    return data


def get_token_probabilities(query_before_token, url, tokens):
    """
    query_before_token: prompt before \\boxed{
    tokens: ["real", "fake"] or other candidates
    returns a dict of token -> probability
    """
    result = call_model_sglang(query_before_token, url)

    top_logprobs = result['meta_info']['output_top_logprobs'][0]
    probs = {}
    for token in tokens:
        found = False
        for candidate in top_logprobs:
            if candidate[-1].lower() == token.lower():  # index -1 -> token, index 1 -> token ids
                probs[token] = math.exp(candidate[0])  # convert logits to probabilities (index 0 is logits)
                found = True
                break
        if not found:
            probs[token] = 0.0  # treat as near-zero probability when missing from top-k
    return probs

def example_use():
    # Example usage
    url = "http://127.0.0.1:30011/generate"  # model endpoint

    # Original query: remove the final real/fake and append \boxed{ (ensure tokenization matches model)
    query_before_token = """To evaluate the veracity of this message, we need to consider several factors, including the source, the context, and the content of the statement.
    
    1. **Source Analysis:**
       - Dmitry Medvedev, a former President of Russia, is a credible source. However, as a current deputy head of Russia's Security Council, his statements should be scrutinized carefully, especially in the context of the ongoing Ukraine conflict.
    
    2. **Context Analysis:**
       - The message is presented as a statement from a lecture at the 'Knowledge First' event as part of the World Youth Festival in Sochi. This context is plausible, but it is important to verify the authenticity of the event and the lecture.
    
    3. **Content Analysis:**
       - The statement includes several key points:
         - Russia has no expansionist ambitions.
         - The Ukraine conflict is a response to provocations from the West.
         - Russia emphasizes the distinction between geographical and strategic borders.
         - Russia claims Ukraine as an "integral part of Russian strategic and historical borders."
         - Russia asserts its rights to shelf territories in the Arctic Ocean.
    
    4. **Verification of Claims:**
       - The claim that Russia has no expansionist ambitions is disputed by many international observers and the actions of Russia in Ukraine and other regions.
       - The assertion that the Ukraine conflict is a response to provocations from the West is contested by many countries and international organizations.
       - The emphasis on strategic borders is a common diplomatic concept but is often used to justify territorial claims.
       - The claim that Ukraine is an "integral part of Russian strategic and historical borders" is disputed by Ukraine and many other countries.
       - The assertion about Russia's rights to shelf territories in the Arctic Ocean is a legitimate claim in international law, but it is part of a broader geopolitical strategy.
    
    5. **Conclusion:**
       - While the statement is from a credible source, the content is highly contested and often disputed in the international community.
       - The message aligns with Russia's official stance but is not universally accepted.
    
    Given the above analysis, the message is more likely to be a real statement from a credible source, but the claims are disputed and not universally accepted.
    
    \\boxed{"""

    probs = get_token_probabilities(query_before_token, url, ["real", "fake"])
    print(probs)

    # Example output: {"real": 0.85, "fake": 0.15}


# ===== Process single sample and save =====
def process_and_save(sample, url, output_dir):
    query = sample.get('input', '')
    prompt_template = (
        "Given the following message, let's think step by step and predict its veracity. "
        "If it is more likely to be a real message, return \\boxed{{real}}; otherwise, return \\boxed{{fake}}: \n {problem}"
    )
    input_prompt = prompt_template.format(problem=query)

    # Find the field containing 'output'
    output_field = next((k for k in sample.keys() if 'output' in k.lower()), None)
    output_text = sample.get(output_field, '') if output_field else ''

    id_ = str(sample.get('id', 'unknown_id'))
    steps = output_text.split("\n\n")
    cancelled_steps = []

    # Step 1: remove each step to measure sensitivity
    for idx, step_to_remove in enumerate(steps):
        remaining_steps = steps[:idx] + steps[idx+1:]
        remaining_text = "\n\n".join(remaining_steps)

        # Construct the prompt
        if "\\boxed{" in remaining_text:
            prompt_before_boxed = input_prompt + remaining_text.split("\\boxed{")[0] + "\\boxed{"
        else:
            prompt_before_boxed = input_prompt + remaining_text + "Therefore, the message is \\boxed{"

        try:
            probabilities = get_token_probabilities(prompt_before_boxed, url, ["real", "fake"])
        except Exception as e:
            print(e)
            probabilities = {"real": None, "fake": None, "error": str(e)}

        cancelled_steps.append({
            "step_text": step_to_remove,
            "probabilities": probabilities
        })

    # Step 2: keep all steps in the prompt
    if "\\boxed{" in output_text:
        prompt_before_boxed = input_prompt + output_text.split("\\boxed{")[0] + "\\boxed{"
    else:
        prompt_before_boxed = input_prompt + output_text + "Therefore, the message is \\boxed{"

    try:
        probabilities = get_token_probabilities(prompt_before_boxed, url, ["real", "fake"])
    except Exception as e:
        print(e)
        probabilities = {"real": None, "fake": None, "error": str(e)}

    cancelled_steps.append({
        "step_text": "",
        "probabilities": probabilities
    })

    # Save results
    sample["steps_scores"] = cancelled_steps
    output_path = os.path.join(output_dir, f"{id_}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Wrote output to {}".format(output_path))

    return id_


# ===== Parallel processing entry point =====
def process_all_samples(input_list, url, output_dir, max_workers=32):
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_and_save, sample, url, output_dir) for sample in input_list]
        for future in tqdm(futures, total=len(futures)):
            _ = future.result()  # completed IDs can be collected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:30011/generate")
    parser.add_argument("--data_path", type=str, default="qwen25-15b-s1-32k.jsonl")
    parser.add_argument("--student", type=str, default="qwen25_15b")
    parser.add_argument("--number_process", type=int, default=128)
    args = parser.parse_args()

    args.output_path = args.data_path.replace(".jsonl", f"_{args.student}_scores")

    data = read_jsonl(args.data_path)

    process_all_samples(
        input_list=data,
        url=args.url,
        output_dir=args.output_path,
        max_workers=args.number_process
    )
