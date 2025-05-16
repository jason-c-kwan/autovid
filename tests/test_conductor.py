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
import core.wrappers

class TestConductor(unittest.TestCase):

    @patch('sys.exit', new_callable=MagicMock)
    @patch('builtins.open', new_callable=mock_open)
    @patch('autogen.conductor.argparse.ArgumentParser')
    @patch('autogen.conductor.load_dotenv')
    @patch('autogen.conductor.yaml.safe_load')
    @patch('autogen.conductor.AssistantAgent')
    @patch('autogen.conductor.UserProxyAgent')
    @patch('autogen.conductor.logging')
    def test_pipeline_execution(self, mock_logging, mock_user_proxy_agent, mock_assistant_agent, 
                              mock_yaml_load, mock_load_dotenv, mock_argparse, mock_open, mock_sys_exit):
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
            {'planner_llm_config': {'config': 'planner'}}, # First call (agents_config)
            { # Second call (pipeline_config)
                'globals': {
                    'data_dir': 'config_data',
                    'workspace_root': 'config_workspace'
                },
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

        mock_runner_instance.call.side_effect = runner_call_side_effect

        # Execute the main function
        conductor.main()

        # Verify file operations
        mock_open.assert_any_call('mock_agents.yaml', 'r')
        mock_open.assert_any_call('mock_pipeline.yaml', 'r')

        # Verify agent instantiation
        mock_assistant_agent.assert_called_once_with(name="planner", llm_config={'config': 'planner'})
        mock_user_proxy_agent.assert_called_once_with(name="runner", human_input_mode="NEVER")

        # Verify runner calls
        mock_runner_instance.call.assert_any_call(core.wrappers.check_datasets, 'config_data')
        mock_runner_instance.call.assert_any_call(
            core.wrappers.extract_transcript, 
            os.path.join('config_data', 'alpha.pptx'), 
            'config_workspace'
        )
        mock_runner_instance.call.assert_any_call(
            core.wrappers.extract_transcript, 
            os.path.join('config_data', 'beta.pptx'), 
            'config_workspace'
        )

        # Verify logging
        mock_logging.info.assert_any_call("Executing step: check_datasets")
        mock_logging.info.assert_any_call("Executing step: extract_transcript")
        mock_logging.info.assert_any_call("Pipeline execution finished.")

        # Verify no errors occurred
        mock_sys_exit.assert_not_called()


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
                                            mock_argparse, mock_open, mock_sys_exit):
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
            {'planner_llm_config': {'config': 'planner'}}, # First call (agents_config)
            { # Second call (pipeline_config)
                'globals': {},
                'steps': [
                    {'id': 'check_datasets'},
                    {'id': 'extract_transcript'}
                ]
            }
        ]

        # Setup mock for open

        # Setup mock agent instances
        mock_planner_instance = MagicMock()
        mock_runner_instance = MagicMock()
        mock_assistant_agent.return_value = mock_planner_instance
        mock_user_proxy_agent.return_value = mock_runner_instance

        # Setup mock for runner.call to simulate an error during transcript extraction
        def runner_call_side_effect(*args, **kwargs):
            if args[0] == core.wrappers.check_datasets:
                return ['alpha', 'beta']
            elif args[0] == core.wrappers.extract_transcript:
                if 'beta.pptx' in args[1]:
                    raise RuntimeError("Simulated extraction error")
                return None
            return None

        mock_runner_instance.call.side_effect = runner_call_side_effect

        # Execute the main function
        conductor.main()

        # Verify file operations
        mock_open.assert_any_call('mock_agents.yaml', 'r')
        mock_open.assert_any_call('mock_pipeline.yaml', 'r')

        # Verify agent instantiation
        mock_assistant_agent.assert_called_once_with(name="planner", llm_config={'config': 'planner'})
        mock_user_proxy_agent.assert_called_once_with(name="runner", human_input_mode="NEVER")

        # Verify runner calls
        mock_runner_instance.call.assert_any_call(core.wrappers.check_datasets, 'test_data')
        mock_runner_instance.call.assert_any_call(
            core.wrappers.extract_transcript, 
            os.path.join('test_data', 'alpha.pptx'), 
            'test_workspace'
        )
        mock_runner_instance.call.assert_any_call(
            core.wrappers.extract_transcript, 
            os.path.join('test_data', 'beta.pptx'), 
            'test_workspace'
        )

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
