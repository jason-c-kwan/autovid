import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import io

# Add the parent directory to the sys.path to import autogen and core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the autogen_agentchat.agents before importing conductor
# This prevents errors if the actual agents require complex setup
sys.modules['autogen_agentchat.agents'] = MagicMock()

# Now import conductor
from autogen import conductor
# import core.wrappers # core.wrappers will be patched

# Import json for creating mock JSON strings
import json

class TestConductor(unittest.TestCase):

    @patch('os.makedirs') # Added
    @patch('core.wrappers.extract_transcript')
    @patch('core.wrappers.check_datasets')
    @patch('sys.exit', new_callable=MagicMock)
    @patch('builtins.open', new_callable=mock_open)
    @patch('autogen.conductor.argparse.ArgumentParser')
    @patch('autogen.conductor.load_dotenv')
    @patch('autogen.conductor.yaml.safe_load')
    @patch('autogen.conductor.AssistantAgent')
    @patch('autogen.conductor.UserProxyAgent')
    @patch('autogen.conductor.logging')
    def test_pipeline_execution(self, mock_logging, mock_user_proxy_agent, mock_assistant_agent,
                              mock_yaml_load, mock_load_dotenv, mock_argparse, mock_open,
                              mock_sys_exit, mock_check_datasets, mock_extract_transcript, mock_os_makedirs): # Added mock_os_makedirs
        # Setup mock for argparse
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        mock_args = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_args.agents_config = 'mock_agents.yaml'
        mock_args.pipeline_config = 'mock_pipeline.yaml'
        mock_args.data_dir = None
        mock_args.workspace_root = None

        # Setup mock for yaml.safe_load
        mock_yaml_load.side_effect = [
            { # First call (agents_config)
                'planner': {
                    'model_client': {
                        'init_args': {
                            'sk_client': {
                                'init_args': {
                                    'ai_model_id': 'gemini-pro',
                                    'api_key': 'test_api_key'
                                }
                            },
                            'prompt_settings': {
                                'init_args': {
                                    'temperature': 0.7
                                }
                            }
                        }
                    }
                    # Add other necessary planner configs if conductor.py uses them
                }
            },
            { # Second call (pipeline_config)
                'data_dir': 'config_data', # Moved out of 'globals'
                'workspace_root': 'config_workspace', # Moved out of 'globals'
                'steps': [
                    {'id': 'check_datasets'},
                    {'id': 'extract_transcript'}
                ]
            }
        ]
        
        # Mock the agent instantiation to capture the arguments
        def assistant_agent_side_effect(*args, **kwargs):
            # Store the arguments for later verification
            assistant_agent_side_effect.call_args = (args, kwargs)
            return MagicMock()
            
        assistant_agent_side_effect.call_args = None
        mock_assistant_agent.side_effect = assistant_agent_side_effect

        # Setup mock for open

        # Setup mock agent instances
        mock_planner_instance = MagicMock()
        mock_runner_instance = MagicMock()
        mock_assistant_agent.return_value = mock_planner_instance
        mock_user_proxy_agent.return_value = mock_runner_instance

        # Setup mock for runner.call
        def runner_call_side_effect(func, *args, **kwargs):
            if func == core.wrappers.check_datasets:
                return ['alpha', 'beta']
            elif func == core.wrappers.extract_transcript:
                return None
            return None

        # mock_runner_instance.call.side_effect = runner_call_side_effect # No longer using runner.call

        # Configure mocks for core.wrappers
        mock_check_datasets.return_value = json.dumps({"pairs": [{"stem": "alpha"}, {"stem": "beta"}]})
        mock_extract_transcript.return_value = json.dumps({"status": "success"})


        # Execute the main function
        conductor.main()

        # Verify file operations
        mock_open.assert_any_call('mock_agents.yaml', 'r')
        mock_open.assert_any_call('mock_pipeline.yaml', 'r')

        # Verify agent instantiation
        # mock_assistant_agent.assert_called_once_with(name="planner", model_client=...) # Keep commented for now
        mock_user_proxy_agent.assert_called_once_with(name="runner")

        # Verify calls to patched core.wrappers functions
        mock_check_datasets.assert_called_once_with('config_data', os.path.join('config_workspace', '00_check_datasets', 'pairs_manifest.json'))
        
        # Check calls to extract_transcript
        # Note: os.path.join might behave differently on different OS for the exact path string,
        # but the components should be correct.
        expected_alpha_transcript_path = os.path.join('config_workspace', '01_transcripts', 'alpha_transcript_manifest.json')
        expected_beta_transcript_path = os.path.join('config_workspace', '01_transcripts', 'beta_transcript_manifest.json')

        mock_extract_transcript.assert_any_call(os.path.join('config_data', 'alpha.pptx'), expected_alpha_transcript_path)
        mock_extract_transcript.assert_any_call(os.path.join('config_data', 'beta.pptx'), expected_beta_transcript_path)

        # Verify os.makedirs calls
        mock_os_makedirs.assert_any_call(os.path.join('config_workspace', '00_check_datasets'), exist_ok=True)
        mock_os_makedirs.assert_any_call(os.path.join('config_workspace', '01_transcripts'), exist_ok=True)

        # Verify logging
        mock_logging.info.assert_any_call("Executing step: check_datasets")
        mock_logging.info.assert_any_call("Executing step: extract_transcript")
        mock_logging.info.assert_any_call("Pipeline execution finished.")

        # Verify no errors occurred
        mock_sys_exit.assert_not_called()


    @patch('os.makedirs') # Added
    @patch('core.wrappers.extract_transcript')
    @patch('core.wrappers.check_datasets')
    @patch('sys.exit', new_callable=MagicMock)
    @patch('builtins.open', new_callable=mock_open)
    @patch('autogen.conductor.argparse.ArgumentParser')
    @patch('autogen.conductor.load_dotenv')
    @patch('autogen.conductor.yaml.safe_load')
    @patch('autogen.conductor.AssistantAgent')
    @patch('autogen.conductor.UserProxyAgent')
    @patch('autogen.conductor.logging')
    def test_extract_transcript_runtime_error(self, mock_logging, mock_user_proxy_agent,
                                            mock_assistant_agent, mock_yaml_load, mock_load_dotenv,
                                            mock_argparse, mock_open, mock_sys_exit,
                                            mock_check_datasets, mock_extract_transcript, mock_os_makedirs): # Added mock_os_makedirs
        # Setup mock for argparse
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        mock_args = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_args.agents_config = 'mock_agents.yaml'
        mock_args.pipeline_config = 'mock_pipeline.yaml'
        mock_args.data_dir = 'test_data'
        mock_args.workspace_root = 'test_workspace'

        # Setup mock for yaml.safe_load
        mock_yaml_load.side_effect = [
            { # First call (agents_config)
                'planner': {
                    'model_client': {
                        'init_args': {
                            'sk_client': {
                                'init_args': {
                                    'ai_model_id': 'gemini-pro',
                                    'api_key': 'test_api_key'
                                }
                            },
                            'prompt_settings': {
                                'init_args': {
                                    'temperature': 0.7
                                }
                            }
                        }
                    }
                }
            },
            { # Second call (pipeline_config)
                # 'globals': {}, # No longer using globals for these
                'data_dir': 'test_data', # Define at root
                'workspace_root': 'test_workspace', # Define at root
                'steps': [
                    {'id': 'check_datasets'},
                    {'id': 'extract_transcript'}
                ]
            }
        ]

        # Setup mock for open

        # Setup mock agent instances
        mock_planner_instance = MagicMock()
        # mock_runner_instance = MagicMock() # Not used directly for calls anymore
        mock_assistant_agent.return_value = mock_planner_instance
        mock_user_proxy_agent.return_value = MagicMock() # mock_runner_instance

        # Configure mocks for core.wrappers
        mock_check_datasets.return_value = json.dumps({"pairs": [{"stem": "alpha"}, {"stem": "beta"}]})

        def extract_transcript_side_effect(pptx_path, output_path):
            if 'beta.pptx' in pptx_path:
                raise RuntimeError("Simulated extraction error")
            return json.dumps({"status": "success"})
        mock_extract_transcript.side_effect = extract_transcript_side_effect


        # Execute the main function
        conductor.main()

        # Verify file operations
        mock_open.assert_any_call('mock_agents.yaml', 'r')
        mock_open.assert_any_call('mock_pipeline.yaml', 'r')

        # Verify agent instantiation
        # mock_assistant_agent.assert_called_once_with(name="planner", model_client=...) # Keep commented
        mock_user_proxy_agent.assert_called_once_with(name="runner")

        # Verify calls to patched core.wrappers functions
        expected_check_datasets_path = os.path.join('test_workspace', '00_check_datasets', 'pairs_manifest.json')
        mock_check_datasets.assert_called_once_with('test_data', expected_check_datasets_path)

        expected_alpha_transcript_path = os.path.join('test_workspace', '01_transcripts', 'alpha_transcript_manifest.json')
        # beta.pptx path for extract_transcript will also be constructed before the error
        expected_beta_transcript_path = os.path.join('test_workspace', '01_transcripts', 'beta_transcript_manifest.json')

        mock_extract_transcript.assert_any_call(os.path.join('test_data', 'alpha.pptx'), expected_alpha_transcript_path)
        mock_extract_transcript.assert_any_call(os.path.join('test_data', 'beta.pptx'), expected_beta_transcript_path)

        # Verify os.makedirs calls
        mock_os_makedirs.assert_any_call(os.path.join('test_workspace', '00_check_datasets'), exist_ok=True)
        mock_os_makedirs.assert_any_call(os.path.join('test_workspace', '01_transcripts'), exist_ok=True)

        # Verify error was logged
        mock_logging.error.assert_called()
        
        # Check that the error was logged with the correct message
        error_calls = [call for call in mock_logging.error.call_args_list 
                      if isinstance(call[0][0], str) and 'RuntimeError' in call[0][0]]
        self.assertGreater(len(error_calls), 0, "Expected error log not found")

        # Verify no unexpected system exit
        mock_sys_exit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
