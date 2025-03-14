import os
import json
import asyncio
import logging
from datetime import datetime
from tqdm import tqdm
from lightrag_flamingo_demo import run_lightrag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"lightrag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration parameters
MODES = ["local", "global", "hybrid", "naive", "mix"]
TOP_KS = [10, 20, 60]
RESPONSE_TYPE = "Single Paragraph"
MARKDOWN_DIR = "./markdown_files"
OUTPUT_FILE = f"./evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

async def evaluate_lightrag():
    try:
        # Load questions
        with open("./Docs/questions_entity.json", "r") as f:
            questions_data = json.load(f)
        logger.info(f"Loaded {sum(len(q) for q in questions_data.values())} questions across {len(questions_data)} categories")
        
        # Load policies (ground truth)
        with open("./Docs/policies.json", "r") as f:
            policies = json.load(f)
        logger.info(f"Loaded {len(policies)} policies with ground truth data")
        
        results = []
        total_evaluations = 0
        completed_evaluations = 0
        
        # Calculate total number of evaluations
        for policy in policies:
            for question_type, questions in questions_data.items():
                for _ in questions:
                    for _ in MODES:
                        for _ in TOP_KS:
                            total_evaluations += 1
        
        logger.info(f"Starting evaluation with {total_evaluations} total combinations")
        
        for policy in tqdm(policies, desc="Processing policies"):
            policy_name = policy["name"]
            markdown_file = os.path.join(MARKDOWN_DIR, f"{policy_name}.md")
            
            # Skip if markdown file doesn't exist
            if not os.path.exists(markdown_file):
                logger.warning(f"Markdown file {markdown_file} not found. Skipping policy {policy_name}.")
                continue
            
            # Process each question type
            for question_type, questions in questions_data.items():
                ground_truth_key = question_type.replace("_questions", "")
                if ground_truth_key not in policy and ground_truth_key != "combined_field":
                    logger.warning(f"Ground truth key {ground_truth_key} not found for policy {policy_name}. Skipping.")
                    continue
                
                # Get ground truth
                if ground_truth_key == "combined_field":
                    ground_truth = policy  # All fields for combined questions
                else:
                    ground_truth = policy.get(ground_truth_key, "")
                
                # Process each question in this type
                for question_template in questions:
                    # Replace placeholders with actual values
                    question = question_template.replace("[INSURED]", policy["insured"])
                    question = question.replace("[POLICY_NAME]", policy["name"])
                    question = question.replace("[POLICY_TYPE]", policy["policy_type"])
                    
                    # Run with different modes and top_k values
                    for mode in MODES:
                        for top_k in TOP_KS:
                            try:
                                logger.debug(f"Running query on {policy_name} with mode={mode}, top_k={top_k}")
                                
                                # Run LightRAG
                                response = await run_lightrag(
                                    file_path=markdown_file,
                                    query=question,
                                    mode=mode,
                                    top_k=top_k,
                                    response_type=RESPONSE_TYPE
                                )
                                
                                # Store result
                                results.append({
                                    "policy_name": policy_name,
                                    "question_type": question_type,
                                    "question": question,
                                    "response": response,
                                    "ground_truth": ground_truth,
                                    "mode": mode,
                                    "top_k": top_k,
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                completed_evaluations += 1
                                
                                # Save intermediate results periodically
                                if len(results) % 10 == 0:
                                    with open(OUTPUT_FILE, "w") as f:
                                        json.dump(results, f, indent=2)
                                    logger.info(f"Saved intermediate results. Completed {completed_evaluations}/{total_evaluations} evaluations")
                                
                            except Exception as e:
                                logger.error(f"Error processing {policy_name}, {question_type}, {mode}, {top_k}: {e}", exc_info=True)
        
        # Save final results
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {OUTPUT_FILE}")
        logger.info(f"Completed {completed_evaluations}/{total_evaluations} evaluations")
        
    except Exception as e:
        logger.error(f"Fatal error in evaluation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(evaluate_lightrag())