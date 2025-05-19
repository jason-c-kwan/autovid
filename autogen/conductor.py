# autogen/conductor.py
import yaml
import json
import argparse
import os
from dotenv import load_dotenv
import logging
import sys # Added import
from semantic_kernel import Kernel
from semantic_kernel.memory.null_memory import NullMemory
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion as GoogleChatCompletion
from semantic_kernel.prompt_template.prompt_template_config import PromptExecutionSettings
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
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
    planner_cfg = agent_cfg["planner"]
    
    # Extract the nested model_client config
    mc_cfg = planner_cfg["model_client"]
    
    # 3a) Build the raw Google Gemini client from sk_client.init_args
    # Accessing nested init_args for both sk_client and its own init_args
    sk_client_cfg = mc_cfg["init_args"]["sk_client"]["init_args"]
    google_client = GoogleChatCompletion(
        gemini_model_id=sk_client_cfg["ai_model_id"],
        api_key=sk_client_cfg["api_key"]
    )
    
    # 3b) Create the Kernel (using NullMemory by default)
    #    If you had kernel.init_args to customize, you'd use them here
    kernel = Kernel(memory=NullMemory())
    
    # 3c) Build prompt settings from prompt_settings.init_args
    # Accessing nested init_args for prompt_settings
    ps_cfg = mc_cfg["init_args"]["prompt_settings"]["init_args"]
    prompt_settings = PromptExecutionSettings(
        temperature=ps_cfg.get("temperature", 1.0),
        # you can also pull max_tokens, top_p, etc. if configured
    )
    
    # 3d) Wrap in the SK adapter
    planner_model_client = SKChatCompletionAdapter(
        google_client,
        kernel=kernel,
        prompt_settings=prompt_settings
    )
    
    # 3e) Instantiate your agent
    planner = AssistantAgent(
        name="planner",
        model_client=planner_model_client
        # human_input_mode parameter removed as it's not supported by this version of AssistantAgent
    )

    runner  = UserProxyAgent(
        name="runner"
        # human_input_mode parameter removed as it's not supported by this version of UserProxyAgent
    ) # Assuming UserProxyAgent for runner

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
    data_dir = args.data_dir if args.data_dir else pipeline_cfg.get("data_dir")
    workspace_root = args.workspace_root if args.workspace_root else pipeline_cfg.get("workspace_root")
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
                # Define manifest path for check_datasets
                check_datasets_manifest_dir = os.path.join(workspace_root, "00_check_datasets")
                os.makedirs(check_datasets_manifest_dir, exist_ok=True)
                check_datasets_manifest_filename = "pairs_manifest.json"
                check_datasets_manifest_path = os.path.join(check_datasets_manifest_dir, check_datasets_manifest_filename)

                # Call check_datasets and get the list of stems
                json_string_result = core.wrappers.check_datasets(data_dir, check_datasets_manifest_path)
                stems = [] # Initialize stems as an empty list
                try:
                    result_data = json.loads(json_string_result)
                    if isinstance(result_data, dict) and "pairs" in result_data and isinstance(result_data["pairs"], list):
                        stems = [pair["stem"] for pair in result_data["pairs"] if "stem" in pair]
                        logging.info(f"Found {len(stems)} datasets: {stems}")
                        if not stems:
                            logging.warning("No stems extracted from check_datasets result.")
                    else:
                        logging.error(f"check_datasets result is not in the expected format (dict with a 'pairs' list). Result: {result_data}")
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON from check_datasets: {json_string_result}")

            elif step_id == "extract_transcript":
                if 'stems' in locals() and stems: # Ensure stems were obtained from check_datasets
                    all_transcripts = []
                    for stem in stems:
                        pptx_path = os.path.join(data_dir, f"{stem}.pptx")
                        logging.info(f"Extracting transcript for {pptx_path}")
                        
                        # Define manifest path for extract_transcript
                        transcript_manifest_dir = os.path.join(workspace_root, "01_transcripts")
                        os.makedirs(transcript_manifest_dir, exist_ok=True)
                        transcript_manifest_filename = f"{stem}_transcript_manifest.json"
                        transcript_manifest_path = os.path.join(transcript_manifest_dir, transcript_manifest_filename)
                        
                        try:
                            # Call extract_transcript for each stem
                            transcript_json_string = core.wrappers.extract_transcript(pptx_path, transcript_manifest_path)
                            logging.info(f"Transcript extraction for {stem} completed (manifest: {transcript_manifest_path}).")
                            # Optionally, parse transcript_json_string and log details or check status
                            try:
                                transcript_data = json.loads(transcript_json_string)
                                all_transcripts.append(transcript_data)
                                if transcript_data.get("status") == "success":
                                    logging.info(f"Transcript extraction for {stem} reported success.")
                                else:
                                    logging.warning(f"Transcript extraction for {stem} reported status: {transcript_data.get('status')}. Error: {transcript_data.get('error')}")
                            except json.JSONDecodeError:
                                logging.error(f"Failed to parse JSON from extract_transcript for {stem}: {transcript_json_string}")
                            # planner.receive({"step": step_id, "stem": stem, "result": transcript_data})
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
