# autogen/conductor.py
import yaml
import json
import argparse
import os
import tempfile # Added for tts_run step
from dotenv import load_dotenv
import logging
import sys # Added import
from pathlib import Path
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

    # Initialize pipeline variables to persist across steps
    stems = []
    all_transcripts = []
    all_tts_manifests = []
    all_rvc_manifests = []
    all_splice_manifests = []
    all_video_analyses = []
    
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
                if stems: # Ensure stems were obtained from check_datasets
                    for stem in stems:
                        pptx_path = os.path.join(data_dir, f"{stem}.pptx")
                        logging.info(f"Extracting transcript for {pptx_path}")
                        
                        # Define manifest path for extract_transcript
                        transcript_manifest_dir = os.path.join(workspace_root, "01_transcripts")
                        os.makedirs(transcript_manifest_dir, exist_ok=True)
                        transcript_manifest_filename = f"{stem}_transcript_manifest.json"
                        transcript_manifest_path = os.path.join(transcript_manifest_dir, transcript_manifest_filename)
                        
                        try:
                            # Get parameters for the step
                            step_params = step.get("parameters", {})
                            cue_from_yaml = step_params.get("cue") # Will be "[transition]" or None if not set

                            call_kwargs = {}
                            if cue_from_yaml is not None:
                                call_kwargs['cue_token'] = cue_from_yaml
                            # We could also handle step_id override from parameters if needed:
                            # call_kwargs['step_id'] = step_params.get("step_id", "extract_transcript") # Uses default from wrapper if not in YAML

                            # Call extract_transcript for each stem
                            transcript_json_string = core.wrappers.extract_transcript(
                                pptx_path,
                                transcript_manifest_path,
                                **call_kwargs # Pass cue_token if specified in YAML, otherwise wrapper default applies
                            )
                            logging.info(f"Transcript extraction for {stem} completed (manifest: {transcript_manifest_path}). Cue token used from YAML: {cue_from_yaml if cue_from_yaml is not None else 'wrapper default'}")
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

            elif step_id == "tts_run":
                step_params = step.get("parameters", {})
                engine = step_params.get("engine")

                if not engine:
                    logging.error(f"TTS engine not specified in parameters for step '{step_id}'. Skipping.")
                    continue

                if not all_transcripts:
                    logging.warning(f"Skipping TTS step '{step_id}': No transcripts available from previous steps.")
                    continue
                tts_manifest_output_dir = os.path.join(workspace_root, "02_tts_audio")
                os.makedirs(tts_manifest_output_dir, exist_ok=True)
                logging.info(f"TTS output directory: {tts_manifest_output_dir}")

                for idx, transcript_data in enumerate(all_transcripts):
                    # Try to get a stem or ID from transcript_data for a more descriptive filename
                    # Fallback to index if not available.
                    # Assuming transcript_data might have 'source_stem' or 'id' from extract_transcript manifest
                    source_identifier = transcript_data.get("source_stem") # Defaulting to check for source_stem
                    if not source_identifier: # Fallback if source_stem is not present
                         source_identifier = transcript_data.get("id", f"transcript_{idx}")


                    current_tts_manifest_filename = f"{source_identifier}_{engine}_{idx}_manifest.json"
                    current_tts_manifest_path = os.path.join(tts_manifest_output_dir, current_tts_manifest_filename)
                    
                    logging.info(f"Processing transcript {idx+1}/{len(all_transcripts)} for TTS using {engine} engine. Output manifest: {current_tts_manifest_path}")

                    try:
                        if engine == "piper":
                            # Piper TTS wrapper expects a file path to the transcript JSON
                            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json", encoding='utf-8') as tmp_transcript_file:
                                json.dump(transcript_data, tmp_transcript_file)
                                temp_transcript_file_path = tmp_transcript_file.name
                            
                            try:
                                # Get chunk_mode from step parameters, default to 'sentence'
                                chunk_mode = step_params.get("chunk_mode", "sentence")
                                
                                manifest_dict = core.wrappers.piper_tts(
                                    transcript_path=temp_transcript_file_path,
                                    output_path=current_tts_manifest_path,
                                    step_id=step_id, # Pass the main step_id for config lookup
                                    config_path=args.pipeline_config,
                                    chunk_mode=chunk_mode
                                )
                                manifest_dict["manifest_path"] = current_tts_manifest_path
                                # Save the updated manifest with manifest_path back to file
                                with open(current_tts_manifest_path, 'w') as f:
                                    json.dump(manifest_dict, f, indent=2)
                                all_tts_manifests.append(manifest_dict)
                                logging.info(f"Piper TTS successful for transcript {idx+1}. Manifest: {current_tts_manifest_path}")
                            finally:
                                os.unlink(temp_transcript_file_path) # Ensure temporary file is deleted

                        elif engine == "orpheus":
                            manifest_dict = core.wrappers.orpheus_tts(
                                input_json_data=transcript_data,
                                output_path=current_tts_manifest_path,
                                step_id=step_id, # Pass the main step_id
                                config_path=args.pipeline_config
                            )
                            manifest_dict["manifest_path"] = current_tts_manifest_path
                            # Save the updated manifest with manifest_path back to file
                            with open(current_tts_manifest_path, 'w') as f:
                                json.dump(manifest_dict, f, indent=2)
                            all_tts_manifests.append(manifest_dict)
                            logging.info(f"Orpheus TTS successful for transcript {idx+1}. Manifest: {current_tts_manifest_path}")
                        
                        else:
                            logging.warning(f"Unsupported TTS engine '{engine}' for transcript {idx+1}. Skipping.")

                    except RuntimeError as e:
                        logging.error(f"RuntimeError during TTS processing for transcript {idx+1} with {engine}: {e}", exc_info=True)
                    except Exception as e:
                        logging.error(f"Unexpected error during TTS processing for transcript {idx+1} with {engine}: {e}", exc_info=True)
                
                logging.info(f"TTS run step '{step_id}' completed. Generated {len(all_tts_manifests)} TTS manifests.")
                # The 'all_tts_manifests' list is now available for downstream steps if they are designed to use it.

            elif step_id == "qc_pronounce":
                if not all_tts_manifests:
                    logging.warning(f"Skipping QC step '{step_id}': No TTS manifests available.")
                    continue
                
                # Extract parameters from pipeline config (new format)
                step_params = step.get("parameters", {})
                
                # Set QC parameters with fallbacks from new configuration format
                mos_threshold = step_params.get("mos_threshold", 3.5)
                wer_threshold = step_params.get("wer_threshold", 0.10)
                max_attempts = step_params.get("max_attempts", 3)
                whisper_model = step_params.get("whisper_model", "large-v3")
                enable_transcription = step_params.get("enable_transcription", True)
                transcription_timeout = step_params.get("transcription_timeout", 30)
                retry_with_phonemes = step_params.get("retry_with_phonemes", True)
                retry_different_engine = step_params.get("retry_different_engine", True)
                preserve_original_on_failure = step_params.get("preserve_original_on_failure", False)
                detect_clipping = step_params.get("detect_clipping", True)
                detect_silence = step_params.get("detect_silence", True)
                silence_threshold = step_params.get("silence_threshold", -40)
                min_chunk_duration = step_params.get("min_chunk_duration", 0.5)
                max_chunk_duration = step_params.get("max_chunk_duration", 30.0)
                
                # Create QC output directory
                qc_output_dir = os.path.join(workspace_root, "02_qc_audio")
                os.makedirs(qc_output_dir, exist_ok=True)
                
                logging.info(f"Starting QC step '{step_id}' with thresholds: MOS >= {mos_threshold}, WER <= {wer_threshold}")
                
                # Process each TTS manifest through QC
                validated_manifests = []
                qc_summary = {"total": 0, "passed": 0, "failed": 0, "fixed": 0}
                
                for idx, tts_manifest in enumerate(all_tts_manifests):
                    try:
                        tts_manifest_path = tts_manifest.get("manifest_path")
                        if not tts_manifest_path:
                            logging.warning(f"TTS manifest {idx+1} missing manifest_path. Skipping QC.")
                            validated_manifests.append(tts_manifest)
                            continue
                            
                        logging.info(f"Running QC for TTS manifest {idx+1}: {tts_manifest_path}")
                        
                        qc_result = core.wrappers.run_audio_qc(
                            input_manifest=tts_manifest_path,
                            output_dir=qc_output_dir,
                            mos_threshold=mos_threshold,
                            wer_threshold=wer_threshold,
                            max_attempts=max_attempts,
                            whisper_model=whisper_model,
                            enable_transcription=enable_transcription,
                            transcription_timeout=transcription_timeout,
                            retry_with_phonemes=retry_with_phonemes,
                            retry_different_engine=retry_different_engine,
                            preserve_original_on_failure=preserve_original_on_failure,
                            detect_clipping=detect_clipping,
                            detect_silence=detect_silence,
                            silence_threshold=silence_threshold,
                            min_chunk_duration=min_chunk_duration,
                            max_chunk_duration=max_chunk_duration,
                            step_id=step_id
                        )
                        
                        # Update summary statistics
                        summary = qc_result.get("summary", {})
                        qc_summary["total"] += summary.get("chunks_processed", 0)
                        qc_summary["passed"] += summary.get("chunks_passed", 0)
                        qc_summary["failed"] += summary.get("chunks_failed", 0)
                        qc_summary["fixed"] += summary.get("chunks_fixed", 0)
                        
                        # Add QC results to original TTS manifest
                        validated_manifest = tts_manifest.copy()
                        validated_manifest["qc_results"] = qc_result
                        validated_manifest["qc_manifest_path"] = qc_result["manifest_path"]
                        
                        validated_manifests.append(validated_manifest)
                        
                        pass_rate = summary.get("pass_rate", 0.0)
                        logging.info(f"QC completed for manifest {idx+1}: {pass_rate:.1%} pass rate")
                        
                    except RuntimeError as e:
                        logging.error(f"QC failed for TTS manifest {idx+1}: {e}")
                        # On QC failure, include original manifest but mark as failed
                        failed_manifest = tts_manifest.copy()
                        failed_manifest["qc_status"] = "failed"
                        failed_manifest["qc_error"] = str(e)
                        validated_manifests.append(failed_manifest)
                    except Exception as e:
                        logging.error(f"Unexpected error during QC for manifest {idx+1}: {e}", exc_info=True)
                        failed_manifest = tts_manifest.copy()
                        failed_manifest["qc_status"] = "error" 
                        failed_manifest["qc_error"] = str(e)
                        validated_manifests.append(failed_manifest)
                
                # Replace all_tts_manifests with QC-validated versions
                all_tts_manifests = validated_manifests
                
                # Log overall QC summary
                total_chunks = qc_summary["total"]
                if total_chunks > 0:
                    overall_pass_rate = qc_summary["passed"] / total_chunks
                    logging.info(f"QC step '{step_id}' completed:")
                    logging.info(f"  Total chunks: {total_chunks}")
                    logging.info(f"  Passed: {qc_summary['passed']} ({overall_pass_rate:.1%})")
                    logging.info(f"  Failed: {qc_summary['failed']}")
                    logging.info(f"  Fixed: {qc_summary['fixed']}")
                    
                    # Warn if pass rate is low
                    if overall_pass_rate < 0.8:
                        logging.warning(f"QC pass rate ({overall_pass_rate:.1%}) is below 80%. Consider reviewing TTS settings.")
                else:
                    logging.warning("No audio chunks processed during QC step.")

            elif step_id == "apply_rvc":
                if not all_tts_manifests:
                    logging.warning(f"Skipping RVC step '{step_id}': No TTS manifests available from previous steps.")
                    continue
                rvc_output_dir = os.path.join(workspace_root, "03_rvc_audio")
                os.makedirs(rvc_output_dir, exist_ok=True)
                logging.info(f"RVC output directory: {rvc_output_dir}")

                for idx, tts_manifest in enumerate(all_tts_manifests):
                    # Generate input manifest path for RVC
                    tts_manifest_path = tts_manifest.get("manifest_path")
                    if not tts_manifest_path:
                        logging.warning(f"No manifest path found for TTS manifest {idx+1}, skipping RVC conversion.")
                        continue

                    try:
                        logging.info(f"Processing TTS manifest {idx+1}/{len(all_tts_manifests)} through RVC.")
                        
                        rvc_manifest = core.wrappers.run_rvc_convert(
                            input_manifest=tts_manifest_path,
                            output_dir=rvc_output_dir,
                            config_path=args.pipeline_config,
                            step_id=step_id
                        )
                        
                        # Add the manifest file path to the manifest dict for splice_audio step
                        rvc_manifest_path = Path(rvc_output_dir) / "rvc_conversion_manifest.json"
                        rvc_manifest["manifest_path"] = str(rvc_manifest_path)
                        
                        all_rvc_manifests.append(rvc_manifest)
                        logging.info(f"RVC conversion {idx+1} completed successfully.")
                        
                    except RuntimeError as e:
                        logging.error(f"RuntimeError during RVC conversion for manifest {idx+1}: {e}", exc_info=True)
                    except Exception as e:
                        logging.error(f"Unexpected error during RVC conversion for manifest {idx+1}: {e}", exc_info=True)

                logging.info(f"RVC step '{step_id}' completed. Generated {len(all_rvc_manifests)} RVC manifests.")

            elif step_id == "splice_audio":
                if not all_rvc_manifests:
                    logging.warning(f"Skipping audio splicing step '{step_id}': No RVC manifests available from previous steps.")
                    continue
                splice_output_dir = os.path.join(workspace_root, "04_spliced_audio")
                os.makedirs(splice_output_dir, exist_ok=True)
                logging.info(f"Audio splicing output directory: {splice_output_dir}")

                for idx, rvc_manifest in enumerate(all_rvc_manifests):
                    # Get RVC manifest path from the manifest itself
                    rvc_manifest_path = rvc_manifest.get("manifest_path")
                    if not rvc_manifest_path:
                        logging.warning(f"No manifest path in RVC manifest {idx+1}, skipping splice")
                        continue
                        
                    if not os.path.exists(rvc_manifest_path):
                        logging.warning(f"RVC manifest file not found at {rvc_manifest_path}, skipping splice")
                        continue
                    
                    try:
                        logging.info(f"Splicing RVC audio chunks {idx+1}/{len(all_rvc_manifests)}.")
                        
                        # Generate unique output filename
                        output_name = f"final_narration_{idx}.wav"
                        
                        splice_manifest = core.wrappers.run_splice_audio(
                            input_manifest=rvc_manifest_path,
                            output_dir=splice_output_dir,
                            output_name=output_name,
                            config_path=args.pipeline_config,
                            step_id=step_id
                        )
                        
                        all_splice_manifests.append(splice_manifest)
                        logging.info(f"Audio splicing {idx+1} completed successfully. Output: {output_name}")
                        
                    except RuntimeError as e:
                        logging.error(f"RuntimeError during audio splicing for manifest {idx+1}: {e}", exc_info=True)
                    except Exception as e:
                        logging.error(f"Unexpected error during audio splicing for manifest {idx+1}: {e}", exc_info=True)

                logging.info(f"Audio splicing step '{step_id}' completed. Generated {len(all_splice_manifests)} splice manifests.")

            elif step_id == "analyze_video":
                if not stems:
                    logging.warning(f"Skipping video analysis step '{step_id}': No stems available from check_datasets.")
                    continue
                video_analysis_output_dir = os.path.join(workspace_root, "05_video_analysis")
                os.makedirs(video_analysis_output_dir, exist_ok=True)
                logging.info(f"Video analysis output directory: {video_analysis_output_dir}")

                # Get step parameters
                step_params = step.get("parameters", {})
                scene_threshold = step_params.get("scene_threshold", 0.4)
                movement_threshold = step_params.get("movement_threshold", 0.1)
                keynote_delay = step_params.get("keynote_delay", 1.0)
                validate_transitions = step_params.get("validate_transitions", True)
                presentation_mode = step_params.get("presentation_mode", False)

                for stem in stems:
                    # Look for video file (.mov or .mp4)
                    video_path = None
                    for ext in ['.mov', '.mp4']:
                        candidate_path = os.path.join(data_dir, f"{stem}{ext}")
                        if os.path.exists(candidate_path):
                            video_path = candidate_path
                            break
                    
                    if not video_path:
                        logging.warning(f"No video file found for stem '{stem}' (checked .mov and .mp4)")
                        continue

                    try:
                        logging.info(f"Analyzing video for {stem}: {video_path}")
                        
                        # Generate output path for analysis manifest
                        analysis_manifest_path = os.path.join(video_analysis_output_dir, f"{stem}_video_analysis.json")
                        
                        # Look for transcript manifest for validation and transition count
                        transcript_path = None
                        expected_transitions = 0
                        if validate_transitions:
                            transcript_manifest_dir = os.path.join(workspace_root, "01_transcripts")
                            transcript_manifest_path = os.path.join(transcript_manifest_dir, f"{stem}_transcript_manifest.json")
                            if os.path.exists(transcript_manifest_path):
                                transcript_path = transcript_manifest_path
                                
                                # Extract expected transition count for Keynote optimization
                                if presentation_mode:
                                    try:
                                        with open(transcript_manifest_path, 'r') as f:
                                            transcript_data = json.load(f)
                                        
                                        # Count transition cues
                                        transition_cues = []
                                        if 'transcript' in transcript_data and 'slides' in transcript_data['transcript']:
                                            for slide in transcript_data['transcript']['slides']:
                                                if 'segments' in slide:
                                                    for segment in slide['segments']:
                                                        if segment.get('kind') == 'cue':
                                                            transition_cues.append(segment.get('cue', '[transition]'))
                                        
                                        expected_transitions = len(transition_cues)
                                        logging.info(f"Extracted {expected_transitions} expected transitions for {stem}")
                                    except (json.JSONDecodeError, KeyError) as e:
                                        logging.warning(f"Failed to extract transition count from {transcript_manifest_path}: {e}")
                            else:
                                logging.warning(f"Transcript manifest not found for validation: {transcript_manifest_path}")
                        
                        # Perform video analysis
                        video_analysis = core.wrappers.analyze_video(
                            video_path=video_path,
                            transcript_path=transcript_path,
                            output_path=analysis_manifest_path,
                            scene_threshold=scene_threshold,
                            movement_threshold=movement_threshold,
                            keynote_delay=keynote_delay,
                            presentation_mode=presentation_mode,
                            expected_transitions=expected_transitions,
                            step_id=step_id
                        )
                        
                        all_video_analyses.append(video_analysis)
                        logging.info(f"Video analysis for {stem} completed successfully. Manifest: {analysis_manifest_path}")
                        
                        # Log analysis summary
                        if 'video_analysis' in video_analysis:
                            analysis_data = video_analysis['video_analysis']
                            scene_count = analysis_data.get('total_scenes', 0)
                            movement_count = analysis_data.get('total_movements', 0)
                            validation_status = analysis_data.get('validation', {}).get('status', 'UNKNOWN')
                            logging.info(f"  {scene_count} scenes, {movement_count} movements, validation: {validation_status}")
                        
                    except RuntimeError as e:
                        logging.error(f"RuntimeError during video analysis for {stem}: {e}", exc_info=True)
                    except Exception as e:
                        logging.error(f"Unexpected error during video analysis for {stem}: {e}", exc_info=True)

                logging.info(f"Video analysis step '{step_id}' completed. Analyzed {len(all_video_analyses)} videos.")

            elif step_id == "sync_slides":
                # Check if we have the required inputs
                required_inputs = []
                missing_inputs = []
                
                # Check for video files from stems
                if stems:
                    video_files = []
                    for stem in stems:
                        # Look for video file (.mov or .mp4)
                        video_path = None
                        for ext in ['.mov', '.mp4']:
                            candidate_path = os.path.join(data_dir, f"{stem}{ext}")
                            if os.path.exists(candidate_path):
                                video_path = candidate_path
                                break
                        
                        if video_path:
                            video_files.append((stem, video_path))
                        else:
                            missing_inputs.append(f"Video file for stem: {stem}")
                    required_inputs.extend(video_files)
                else:
                    missing_inputs.append("No stems available from check_datasets")
                
                # Check for audio splice manifests
                if all_splice_manifests:
                    audio_manifests = all_splice_manifests
                else:
                    missing_inputs.append("Audio splice manifests from splice_audio step")
                
                # Check for video analysis manifests
                if all_video_analyses:
                    video_analyses = all_video_analyses
                else:
                    missing_inputs.append("Video analysis manifests from analyze_video step")
                
                if missing_inputs:
                    logging.warning(f"Skipping sync_slides step '{step_id}': Missing inputs: {missing_inputs}")
                    continue
                
                # Perform video synchronization for each stem
                all_sync_manifests = []
                sync_output_dir = os.path.join(workspace_root, "06_synchronized_videos")
                os.makedirs(sync_output_dir, exist_ok=True)
                logging.info(f"Video sync output directory: {sync_output_dir}")
                
                for idx, (stem, video_path) in enumerate(video_files):
                    try:
                        logging.info(f"Synchronizing video for {stem}: {video_path}")
                        
                        # Find corresponding audio splice manifest
                        audio_manifest_path = None
                        if idx < len(audio_manifests):
                            # Use the splice manifest if available
                            if isinstance(audio_manifests[idx], dict) and 'manifest_path' in audio_manifests[idx]:
                                audio_manifest_path = audio_manifests[idx]['manifest_path']
                            elif isinstance(audio_manifests[idx], str):
                                audio_manifest_path = audio_manifests[idx]
                        
                        if not audio_manifest_path or not os.path.exists(audio_manifest_path):
                            # Fallback: look for splice manifest by stem or generic name
                            splice_manifest_dir = os.path.join(workspace_root, "04_spliced_audio")
                            audio_manifest_path = os.path.join(splice_manifest_dir, f"{stem}_splice_manifest.json")
                            # If stem-based doesn't exist, try generic filename
                            if not os.path.exists(audio_manifest_path):
                                audio_manifest_path = os.path.join(splice_manifest_dir, "splice_manifest.json")
                        
                        # Find corresponding video analysis manifest
                        video_manifest_path = None
                        if idx < len(video_analyses):
                            if isinstance(video_analyses[idx], dict):
                                # Extract video analysis manifest path
                                video_analysis_dir = os.path.join(workspace_root, "05_video_analysis")
                                video_manifest_path = os.path.join(video_analysis_dir, f"{stem}_video_analysis.json")
                        
                        if not video_manifest_path or not os.path.exists(video_manifest_path):
                            logging.warning(f"Video analysis manifest not found for {stem}, using basic sync")
                            video_manifest_path = None
                        
                        # Find the spliced audio file
                        spliced_audio_path = None
                        if audio_manifest_path and os.path.exists(audio_manifest_path):
                            try:
                                with open(audio_manifest_path, 'r') as f:
                                    splice_data = json.load(f)
                                    if 'output_audio' in splice_data:
                                        spliced_audio_path = splice_data['output_audio']
                                    elif 'audio_splice' in splice_data and 'output_path' in splice_data['audio_splice']:
                                        spliced_audio_path = splice_data['audio_splice']['output_path']
                            except (json.JSONDecodeError, KeyError) as e:
                                logging.warning(f"Failed to parse splice manifest for {stem}: {e}")
                        
                        if not spliced_audio_path or not os.path.exists(spliced_audio_path):
                            # Fallback: look for spliced audio file by stem
                            splice_audio_dir = os.path.join(workspace_root, "04_spliced_audio")
                            spliced_audio_path = os.path.join(splice_audio_dir, f"{stem}_final.wav")
                            if not os.path.exists(spliced_audio_path):
                                spliced_audio_path = os.path.join(splice_audio_dir, f"final_narration_{idx}.wav")
                        
                        if not spliced_audio_path or not os.path.exists(spliced_audio_path):
                            logging.error(f"Spliced audio file not found for {stem}")
                            continue
                        
                        # Generate output path for synchronized video
                        output_video_path = os.path.join(sync_output_dir, f"{stem}_synchronized.mp4")
                        
                        # Perform synchronization
                        sync_manifest = core.wrappers.sync_slides(
                            video_path=video_path,
                            audio_path=spliced_audio_path,
                            output_path=output_video_path,
                            video_manifest=video_manifest_path,
                            audio_manifest=audio_manifest_path,
                            step_id=step_id,
                            config_path=args.pipeline_config
                        )
                        
                        all_sync_manifests.append(sync_manifest)
                        logging.info(f"Video synchronization for {stem} completed successfully. Output: {output_video_path}")
                        
                    except RuntimeError as e:
                        logging.error(f"RuntimeError during video synchronization for {stem}: {e}", exc_info=True)
                    except Exception as e:
                        logging.error(f"Unexpected error during video synchronization for {stem}: {e}", exc_info=True)
                
                logging.info(f"Video sync step '{step_id}' completed. Generated {len(all_sync_manifests)} synchronized videos.")

            # Add other steps here as needed
            # elif step_id == "make_srt":
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
