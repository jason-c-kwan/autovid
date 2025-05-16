# autogen/conductor.py
import yaml
import json
import argparse
import os
from dotenv import load_dotenv
import logging
import sys # Added import

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
import core.wrappers

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

def main():

    # Set up argparse
    parser = argparse.ArgumentParser(description="Run the Autovid pipeline.")
    parser.add_argument("--agents-config", type=str, default="config/agents.yaml",
                        help="Path to the agents configuration file.")
    parser.add_argument("--pipeline-config", type=str, default="config/pipeline.yaml",
                        help="Path to the pipeline configuration file.")
    parser.add_argument("--data-dir", type=str, help="Override data directory.")
    parser.add_argument("--workspace-root", type=str, help="Override workspace root directory.")

    args = parser.parse_args()

    # Read agent settings
    try:
        with open(args.agents_config, 'r') as f:
            agent_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Agents config file not found: {args.agents_config}")
        sys.exit(1)
    except yaml.YAMLError:
        logging.error(f"Error parsing agents config file: {args.agents_config}")
        sys.exit(1)

    # Instantiate agents (assuming agent_cfg has appropriate structure)
    # This part might need adjustment based on the actual agents.yaml structure
    planner = AssistantAgent(name="planner", llm_config=agent_cfg.get("planner_llm_config"))
    runner  = UserProxyAgent(name="runner", human_input_mode="NEVER") # Assuming UserProxyAgent for runner

    # Read pipeline steps and globals
    try:
        with open(args.pipeline_config, 'r') as f:
            pipeline_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Pipeline config file not found: {args.pipeline_config}")
        sys.exit(1)
    except yaml.YAMLError:
        logging.error(f"Error parsing pipeline config file: {args.pipeline_config}")
        sys.exit(1)

    # Override globals with CLI arguments if provided
    data_dir = args.data_dir if args.data_dir else pipeline_cfg.get("globals", {}).get("data_dir")
    workspace_root = args.workspace_root if args.workspace_root else pipeline_cfg.get("globals", {}).get("workspace_root")
    steps = pipeline_cfg.get("steps", [])

    if not data_dir:
        logging.error("Data directory not specified. Use --data-dir or set in pipeline config.")
        sys.exit(1)
    if not workspace_root:
        logging.error("Workspace root not specified. Use --workspace-root or set in pipeline config.")
        sys.exit(1)

    # Execute pipeline steps
    for step in steps:
        step_id = step.get("id")
        logging.info(f"Executing step: {step_id}")

        try:
            if step_id == "check_datasets":
                # Call check_datasets and get the list of stems
                stems_result = runner.call(core.wrappers.check_datasets, data_dir)
                if isinstance(stems_result, list):
                    stems = stems_result
                    logging.info(f"Found {len(stems)} datasets: {stems}")
                else:
                    logging.warning(f"check_datasets did not return a list. Result: {stems_result}")
                    stems = [] # Ensure stems is a list to avoid errors in the next step

            elif step_id == "extract_transcript":
                if 'stems' in locals() and stems: # Ensure stems were obtained from check_datasets
                    for stem in stems:
                        pptx_path = os.path.join(data_dir, f"{stem}.pptx")
                        logging.info(f"Extracting transcript for {pptx_path}")
                        try:
                            # Call extract_transcript for each stem
                            transcript_result = runner.call(core.wrappers.extract_transcript, pptx_path, workspace_root)
                            logging.info(f"Transcript extraction for {stem} completed.")
                            # Process transcript_result if needed
                            # planner.receive({"step": step_id, "stem": stem, "result": transcript_result})
                        except RuntimeError as e:
                            logging.error(f"RuntimeError extracting transcript for {stem}: {e}", exc_info=True)
                        except Exception as e:
                            logging.error(f"An unexpected error occurred extracting transcript for {stem}: {e}", exc_info=True)
                elif 'stems' not in locals():
                     logging.warning("Skipping extract_transcript: check_datasets step did not run or failed.")
                else:
                     logging.info("No stems found to extract transcripts.")

            # Add other steps here as needed
            # elif step_id == "another_step":
            #     ...

            # Send result to planner if needed (adjust based on actual planner usage)
            # if 'stems' in locals():
            #     planner.receive({"step": step_id, "result": stems})
            # elif 'transcript_result' in locals():
            #     planner.receive({"step": step_id, "result": transcript_result})

        except Exception as e:
            logging.error(f"An error occurred during step '{step_id}': {e}", exc_info=True)

    logging.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
