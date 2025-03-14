import os
import json
import asyncio
import logging
from datetime import datetime
from tqdm import tqdm
from lightrag_flamingo import run_lightrag

dir_path = f"./qa_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"./{dir_path}/lightrag_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration parameters
MODES = ["local", "global", "hybrid", "naive", "mix"]
TOP_KS = [10, 20, 60]
RESPONSE_TYPE = "Single Paragraph"
MARKDOWN_DIR = "./markdown_files"

OUTPUT_FILE = f"./{dir_path}/evaluation_results.json"

if not os.path.exists(dir_path):
    os.mkdir(dir_path)

OUTPUT_FILE = f"./{dir_path}/qa_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

async def evaluate_qa_lightrag():
    try:
        # Load QA data
        with open("./Docs/questions_doc.json", "r") as f:
            qa_data = json.load(f)
        
        total_questions = sum(len(policy["qa_pairs"]) for policy in qa_data)
        logger.info(f"Loaded {total_questions} Q&A pairs across {len(qa_data)} policy documents")
        
        results = []
        total_evaluations = 0
        completed_evaluations = 0
        
        # Calculate total number of evaluations
        for policy in qa_data:
            for qa_pair in policy["qa_pairs"]:
                for mode in MODES:
                    for top_k in TOP_KS:
                        total_evaluations += 1
        
        logger.info(f"Starting evaluation with {total_evaluations} total combinations")
        
        for policy in tqdm(qa_data, desc="Processing policies"):
            policy_name = policy["policy_name"]
            entity = policy["entity"]
            markdown_file = os.path.join(MARKDOWN_DIR, f"{policy_name}.md")
            working_dir = f"./{dir_path}/{policy_name}"
            
            if not os.path.exists(working_dir):
                os.mkdir(working_dir)
            
            # Skip if markdown file doesn't exist
            if not os.path.exists(markdown_file):
                logger.warning(f"Markdown file {markdown_file} not found. Skipping policy {policy_name}.")
                continue
            
            # Process each QA pair
            for qa_pair in policy["qa_pairs"]:
                question = qa_pair["question"]
                ground_truth = qa_pair["answer"]
                
                # Run with different modes and top_k values
                for mode in MODES:
                    for top_k in TOP_KS:
                        try:
                            logger.debug(f"Running query on {policy_name} with mode={mode}, top_k={top_k}")
                            
                            # Run LightRAG
                            response, context = await run_lightrag(
                                working_dir=working_dir,
                                file_path=markdown_file,
                                query=question,
                                mode=mode,
                                top_k=top_k,
                                response_type=RESPONSE_TYPE
                            )
                            
                            # Store result
                            results.append({
                                "policy_name": policy_name,
                                "entity": entity,
                                "question": question,
                                "response": response,
                                "context": context,
                                "ground_truth": ground_truth,
                                "mode": mode,
                                "top_k": top_k,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            completed_evaluations += 1
                            
                            # Save intermediate results periodically
                            if completed_evaluations % 10 == 0:
                                with open(OUTPUT_FILE, "w") as f:
                                    json.dump(results, f, indent=2)
                                logger.info(f"Saved intermediate results. Completed {completed_evaluations}/{total_evaluations} evaluations")
                            
                        except Exception as e:
                            logger.error(f"Error processing {policy_name}, question '{question[:30]}...', mode={mode}, top_k={top_k}: {e}", exc_info=True)
        
        # Save final results
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {OUTPUT_FILE}")
        logger.info(f"Completed {completed_evaluations}/{total_evaluations} evaluations")
        
    except Exception as e:
        logger.error(f"Fatal error in evaluation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(evaluate_qa_lightrag())